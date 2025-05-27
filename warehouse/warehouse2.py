"""
Main function to run FastAPI server.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from typing_extensions import override
from fastapi import HTTPException
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent # Added BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext # Added
from google.adk.tools.agent_tool import AgentTool 
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from google.adk.events import Event # Added
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect
from voice_agent import router as voice_agent_router
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Use this for consistency

# --- Configuration & Global Setup ---
load_dotenv()

APP_NAME = "ADK_MCP_App_Updated" # Changed to avoid potential conflicts if old app is running
MODEL_ID = "gemini-2.0-flash" # Updated to a generally available model, ensure you have access
GEMINI_PRO_MODEL_ID = "gemini-2.5-pro-preview-05-06" # For potentially more complex tasks like generation/revision

STATIC_DIR = "static"

# Initialize services (globally or via dependency injection)
session_service = InMemorySessionService()
artifacts_service = InMemoryArtifactService()

# Global variable to hold loaded MCP tools after lifespan startup
loaded_mcp_tools_global: Dict[str, Any] = {}

class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioServerParameters]

# --- Server Parameter Definitions ---
weather_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/weather_server.py"],
)
ct_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/cocktail.py"],
)
bnb_server_params = StdioServerParameters(
    command="npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
)
mlb_stats_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/mlb_stats_server.py"],
)
web_search_server_params = StdioServerParameters( # NEW
    command="python",
    args=["./mcp_server/web_search_server.py"],
)
bq_vector_search_server_params = StdioServerParameters( # NEW
    command="python",
    args=["./mcp_server/bq_vector_search_server.py"],
)
visual_asset_server_params = StdioServerParameters( # Your existing Imagen/Cloudflare server
    command="python",
    args=["./mcp_server/visual_asset_server.py"], # Path to your image generation server
)
static_retriever_server_params = StdioServerParameters( # NEW - for headshots
    command="python",
    args=["./mcp_server/static_asset_retriever_mcp_server.py"],
)
image_embedding_server_params = StdioServerParameters( # Your existing for vector search (logos)
    command="python",
    args=["./mcp_server/image_embedding_server.py"],
)
video_clip_server_params = StdioServerParameters( # NEW
    command="python",
    args=["./mcp_server/video_clip_server.py"],
)
audio_processing_server_params = StdioServerParameters( # NEW - for TTS and STT
    command="python",
    args=["./mcp_server/audio_processing_server.py"], # Path to your audio server
)


server_configs_instance = AllServerConfigs(
    configs={
        "weather": weather_server_params,
        "bnb": bnb_server_params,
        "ct": ct_server_params,
        "mlb": mlb_stats_server_params,
        "web_search": web_search_server_params,         # NEW
        "bq_search": bq_vector_search_server_params, # NEW
        "visual_assets": visual_asset_server_params, # MCP for Imagen/Cloudflare generation
        "static_retriever_mcp": static_retriever_server_params, # MCP for GCS headshot check
        "image_embedding_mcp": image_embedding_server_params, # MCP for logo vector search
        "video_clip_generator_mcp": video_clip_server_params, 
        "audio_processing_mcp": audio_processing_server_params, 
    }
)

# --- Agent Instructions ---
ROOT_AGENT_INSTRUCTION = """
**Role:** You are a Virtual Assistant acting as a Request Router.
**Primary Goal:** Analyze user requests and route them to the correct specialist sub-agent.
**Capabilities & Routing:**
* **Greetings:** If the user greets you, respond warmly and directly.
* **Cocktails:** Route requests about cocktails, drinks, recipes, or ingredients to `cocktail_assistant`.
* **Booking & Weather:** Route requests about booking accommodations or weather to `booking_assistant`.
* **MLB Information (General):** Route general requests concerning Major League Baseball (MLB) stats, scores, schedules, rosters, standings to the `mlb_assistant`.
    The `mlb_assistant` will handle obtaining any necessary IDs (like `game_pk`, `player_id`, `team_id`) if not provided by the user for these general queries.
* **MLB Game Recap:** If the user specifically asks for a "game recap", "recap of the game", "game summary" or similar, and a specific game can be identified (e.g., "recap of yesterday's Yankees game" or "recap for game PK 12345"), route the request to the `game_recap_assistant`.
    - If a `game_pk` is mentioned or easily derivable from the query (e.g. from a team name and date like "yesterday's Yankees game"), include it or the identifying information in the routing.
    - If the game for the recap is unclear, you can first delegate to `mlb_assistant` to help identify the `game_pk`, and then if `game_pk` is found, the user might be prompted to ask for the recap again, or you could try re-routing. (Simpler: for now, assume if routed to `game_recap_assistant`, the query contains enough info to derive game_pk, or `game_recap_assistant` will handle clarification if needed).
* **Out-of-Scope:** If the request is unrelated, state directly that you cannot assist.
**Key Directives:**
* **Delegate Immediately:** Once a suitable sub-agent is identified, route the request.
* **Do Not Answer Delegated Topics:** You must **not** attempt to answer questions for delegated topics yourself.
* **Formatting:** Format your final response using Markdown.
* **Game Recap Clarification:** If a user asks for a game recap but doesn't specify which game, ask them to specify the game (e.g., "Which game would you like a recap for? Please provide the teams and date, or the game ID if you know it.") before attempting to route to `game_recap_assistant`. If they provide details, then route.
"""


# --- Helper Function (from original code) ---
def _clean_json_string_from_llm(raw_json_string: Optional[str], default_if_empty: str = "[]") -> str:
    if not raw_json_string: return default_if_empty
    clean_string = raw_json_string.strip()
    if clean_string.startswith("```json"):
        clean_string = clean_string[7:]
        if clean_string.endswith("```"): clean_string = clean_string[:-3]
    elif clean_string.startswith("```"):
        clean_string = clean_string[3:]
        if clean_string.endswith("```"): clean_string = clean_string[:-3]
    return clean_string.strip() if clean_string.strip() else default_if_empty

################################################################################
# --- PHASE 1: NEW CUSTOM AGENT - StaticAssetPipelineAgent ---
################################################################################
class StaticAssetPipelineAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    entity_extractor: LlmAgent
    static_asset_query_generator: LlmAgent
    static_asset_retriever: LlmAgent

    def __init__(self, name: str,
                 entity_extractor: LlmAgent,
                 static_asset_query_generator: LlmAgent,
                 static_asset_retriever: LlmAgent):
        super().__init__(
            name=name,
            entity_extractor=entity_extractor,
            static_asset_query_generator=static_asset_query_generator,
            static_asset_retriever=static_asset_retriever,
            sub_agents=[entity_extractor, static_asset_query_generator, static_asset_retriever]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting static asset retrieval pipeline.")
        ctx.session.state["current_static_assets_list"] = [] # Initialize output

        if not ctx.session.state.get("current_recap"):
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping static asset pipeline.")
            return

        logger.info(f"[{self.name}] Running EntityExtractorForAssets...")
        async for event in self.entity_extractor.run_async(ctx): yield event
        
        raw_entities = ctx.session.state.get("extracted_entities_json", "{}")
        ctx.session.state["extracted_entities_json"] = _clean_json_string_from_llm(raw_entities, default_if_empty='{"players":[], "teams":[]}')
        logger.info(f"[{self.name}] Cleaned extracted_entities_json: {ctx.session.state['extracted_entities_json']}")

        logger.info(f"[{self.name}] Running StaticAssetQueryGenerator...")
        async for event in self.static_asset_query_generator.run_async(ctx): yield event
        logger.info(f"[{self.name}] Output of StaticAssetQueryGenerator (static_asset_search_queries_json): {ctx.session.state.get('static_asset_search_queries_json')}")

        if "player_lookup_dict_json" not in ctx.session.state:
            ctx.session.state["player_lookup_dict_json"] = "{}"
            logger.warning(f"[{self.name}] 'player_lookup_dict_json' was not in session state. Defaulted to empty.")
        logger.info(f"[{self.name}] player_lookup_dict_json for StaticAssetRetriever: {ctx.session.state.get('player_lookup_dict_json')}")

        logger.info(f"[{self.name}] Running StaticAssetRetriever...")
        async for event in self.static_asset_retriever.run_async(ctx): yield event
        
        retrieved_static_assets_json_raw = ctx.session.state.get("retrieved_static_assets_json", "[]")
        retrieved_static_assets_json_clean = _clean_json_string_from_llm(retrieved_static_assets_json_raw)
        
        current_static_assets_list = []
        try:
            parsed_static_assets = json.loads(retrieved_static_assets_json_clean)
            if isinstance(parsed_static_assets, list):
                current_static_assets_list = [item for item in parsed_static_assets if isinstance(item, dict) and item.get("image_uri")]
            else:
                 logger.warning(f"[{self.name}] Parsed static assets was not a list: {type(parsed_static_assets)}. Cleaned JSON: {retrieved_static_assets_json_clean}")
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse static assets JSON: {e}. Raw: '{retrieved_static_assets_json_raw}', Cleaned: '{retrieved_static_assets_json_clean}'")
        
        ctx.session.state["current_static_assets_list"] = current_static_assets_list
        logger.info(f"[{self.name}] Static asset retrieval pipeline finished. Found {len(current_static_assets_list)} static assets.")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"Static asset pipeline complete. Found {len(current_static_assets_list)} assets.")]))


################################################################################
# --- PHASE 2: NEW CUSTOM AGENT - IterativeImageGenerationAgent ---
################################################################################
class IterativeImageGenerationAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    generated_visual_prompts_generator: LlmAgent
    visual_generator_mcp_caller: LlmAgent
    visual_critic: LlmAgent
    new_visual_prompts_creator: LlmAgent
    max_visual_refinement_loops: int

    def __init__(self, name: str,
                 generated_visual_prompts_generator: LlmAgent,
                 visual_generator_mcp_caller: LlmAgent,
                 visual_critic: LlmAgent,
                 new_visual_prompts_creator: LlmAgent,
                 max_visual_refinement_loops: int = 1):
        super().__init__(
            name=name,
            generated_visual_prompts_generator=generated_visual_prompts_generator,
            visual_generator_mcp_caller=visual_generator_mcp_caller,
            visual_critic=visual_critic,
            new_visual_prompts_creator=new_visual_prompts_creator,
            max_visual_refinement_loops=max_visual_refinement_loops,
            sub_agents=[generated_visual_prompts_generator, visual_generator_mcp_caller, visual_critic, new_visual_prompts_creator]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting iterative image generation workflow.")
        ctx.session.state["all_generated_image_assets_details"] = [] # Initialize output

        if not ctx.session.state.get("current_recap"): # Basic prerequisite
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping image generation.")
            return

        logger.info(f"[{self.name}] Running GeneratedVisualPromptsGenerator (for initial prompts)...")
        async for event in self.generated_visual_prompts_generator.run_async(ctx): yield event
        
        current_prompts_json_for_next_iteration = _clean_json_string_from_llm(
            ctx.session.state.get("visual_generation_prompts_json", "[]")
        )
        logger.info(f"[{self.name}] Initial cleaned prompts JSON for visual gen loop: {current_prompts_json_for_next_iteration}")

        all_generated_assets_details_for_this_agent = []

        for i in range(self.max_visual_refinement_loops + 1):
            iteration_label = f"Iteration {i+1}/{self.max_visual_refinement_loops + 1}"
            logger.info(f"[{self.name}] Image generation/refinement {iteration_label}")
            
            prompts_to_use_this_iteration_str = current_prompts_json_for_next_iteration
            current_prompts_list_for_iter = []
            try:
                parsed_list = json.loads(prompts_to_use_this_iteration_str)
                if isinstance(parsed_list, list) and parsed_list:
                    current_prompts_list_for_iter = [str(p) for p in parsed_list if isinstance(p, str)]
                    if not current_prompts_list_for_iter: raise ValueError("Prompt list became empty.")
                else:
                    log_msg = f"[{self.name}] No valid visual prompts for {iteration_label}."
                    if i == 0: logger.warning(log_msg + " Ending image generation.")
                    else: logger.info(log_msg + " Ending image refinement.")
                    break 
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"[{self.name}] Invalid JSON or content for prompts in {iteration_label}. Error: {e}. JSON: '{prompts_to_use_this_iteration_str}'. Stopping loop.")
                break
            
            ctx.session.state['visual_generation_prompts_json_for_tool'] = prompts_to_use_this_iteration_str
            ctx.session.state['game_pk_str_for_tool'] = str(ctx.session.state.get("game_pk", "unknown_game"))
            
            logger.info(f"[{self.name}] {iteration_label}: Calling VisualGeneratorMCPCaller with {len(current_prompts_list_for_iter)} prompts.")
            async for event in self.visual_generator_mcp_caller.run_async(ctx): yield event
            
            tool_output_json_string_raw = ctx.session.state.get("generated_visual_assets_uris_json", '"[]"')
            tool_output_json_string_clean = _clean_json_string_from_llm(tool_output_json_string_raw, default_if_empty='"[]"')
            
            generated_uris_this_iter = []
            tool_had_error_flag = False # Add a flag
            try:
                parsed_tool_output_content = json.loads(tool_output_json_string_clean)
                logger.info(f"[{self.name}] {iteration_label}: Parsed tool output content: {parsed_tool_output_content} (Type: {type(parsed_tool_output_content)})")
                if isinstance(parsed_tool_output_content, list):
                    generated_uris_this_iter = [uri for uri in parsed_tool_output_content if isinstance(uri, str) and uri.startswith("gs://")]
                    logger.info(f"[{self.name}] {iteration_label}: Successfully parsed {len(generated_uris_this_iter)} URIs: {generated_uris_this_iter}")
                elif isinstance(parsed_tool_output_content, dict) and parsed_tool_output_content.get("error"):
                    logger.error(f"[{self.name}] {iteration_label}: MCP tool error: {parsed_tool_output_content['error']}")
                    tool_had_error_flag = True
                else:
                    logger.warning(f"[{self.name}] {iteration_label}: Parsed tool output was not a list or error dict: {parsed_tool_output_content} (Type: {type(parsed_tool_output_content)})")
                    if isinstance(parsed_tool_output_content, str):
                        logger.info(f"[{self.name}] {iteration_label}: Attempting secondary parse on string: {parsed_tool_output_content[:100]}...")

                        try:
                            secondary_parsed_output = json.loads(parsed_tool_output_content)
                            if isinstance(secondary_parsed_output, list):
                                generated_uris_this_iter = [uri for uri in secondary_parsed_output if isinstance(uri, str) and uri.startswith("gs://")]
                                logger.info(f"[{self.name}] {iteration_label}: Successfully parsed {len(generated_uris_this_iter)} URIs from secondary parse: {generated_uris_this_iter}")
                            else:
                                logger.warning(f"[{self.name}] {iteration_label}: Secondary parse did not yield a list: {secondary_parsed_output} (Type: {type(secondary_parsed_output)})")
                        except json.JSONDecodeError as e2:
                            logger.error(f"[{self.name}] {iteration_label}: Secondary JSON parse failed. Error: {e2}. String was: '{parsed_tool_output_content}'")
                            tool_had_error_flag = True
                # Simplified error handling for brevity; refer to original for more detailed secondary parsing
            except json.JSONDecodeError as e:
                logger.error(f"[{self.name}] {iteration_label}: JSON parse of tool output failed. Error: {e}. Output: '{tool_output_json_string_clean}'")
                tool_had_error_flag = True

            assets_for_critique_this_iteration = []
            if not current_prompts_list_for_iter:
                logger.warning(f"[{self.name}] {iteration_label}: No prompts were available for this iteration. Skipping asset detail appending.")
            elif tool_had_error_flag and not generated_uris_this_iter: # If tool errored AND we have no URIs
                logger.warning(f"[{self.name}] {iteration_label}: Tool reported an error or parsing failed, and no URIs were extracted. Associating null URIs with prompts.")
                for prompt_text in current_prompts_list_for_iter:
                    assets_for_critique_this_iteration.append({"prompt_origin": prompt_text, "image_uri": None, "type": "generated_image"})
            else:
                logger.info(f"[{self.name}] {iteration_label}: Populating assets. URIs for this iter ({len(generated_uris_this_iter)}): {generated_uris_this_iter}. Prompts ({len(current_prompts_list_for_iter)}).")

            for idx, prompt_text in enumerate(current_prompts_list_for_iter):
                asset_uri = generated_uris_this_iter[idx] if idx < len(generated_uris_this_iter) else None
                logger.info(f"[{self.name}] {iteration_label}: For prompt '{prompt_text[:50]}...', derived URI: {asset_uri}")
            
                assets_for_critique_this_iteration.append({"prompt_origin": prompt_text, "image_uri": asset_uri, "type": "generated_image"})
                if asset_uri:
                    logger.info(f"[{self.name}] {iteration_label}: Appending to all_generated_assets_details_for_this_agent - URI: {asset_uri}, Prompt: '{prompt_text[:50]}...'")
                    all_generated_assets_details_for_this_agent.append({
                        "prompt_origin": prompt_text, "image_uri": asset_uri,
                        "type": "generated_image", "iteration": i + 1
                    })
            
            ctx.session.state["assets_for_critique_json"] = json.dumps(assets_for_critique_this_iteration)
            ctx.session.state["prompts_used_for_critique_json"] = prompts_to_use_this_iteration_str

            if i >= self.max_visual_refinement_loops: break
            
            logger.info(f"[{self.name}] {iteration_label}: Running VisualCritic...")
            async for event in self.visual_critic.run_async(ctx): yield event
            
            critique_text = ctx.session.state.get("visual_critique_text", "")
            if "sufficient" in critique_text.lower():
                logger.info(f"[{self.name}] Visuals deemed sufficient. Ending refinement.")
                break

            logger.info(f"[{self.name}] {iteration_label}: Running NewVisualPromptsFromCritique...")
            async for event in self.new_visual_prompts_creator.run_async(ctx): yield event
            
            new_prompts_json_raw = ctx.session.state.get("new_visual_generation_prompts_json", "[]")
            current_prompts_json_for_next_iteration = _clean_json_string_from_llm(new_prompts_json_raw)
            
            try: # Check if new prompts are valid to continue
                if not json.loads(current_prompts_json_for_next_iteration): break
            except: break
        
        ctx.session.state["all_generated_image_assets_details"] = all_generated_assets_details_for_this_agent
        logger.info(f"[{self.name}] Iterative image generation finished. Generated {len(all_generated_assets_details_for_this_agent)} image assets details.")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"Iterative image generation complete. Generated {len(all_generated_assets_details_for_this_agent)} assets.")]))


################################################################################
# --- PHASE 3: NEW CUSTOM AGENT - VideoPipelineAgent ---
################################################################################
class VideoPipelineAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    veo_prompt_generator: LlmAgent
    video_generator_mcp_caller: LlmAgent

    def __init__(self, name: str,
                 veo_prompt_generator: LlmAgent,
                 video_generator_mcp_caller: LlmAgent):
        super().__init__(
            name=name,
            veo_prompt_generator=veo_prompt_generator,
            video_generator_mcp_caller=video_generator_mcp_caller,
            sub_agents=[veo_prompt_generator, video_generator_mcp_caller]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting video clip generation pipeline.")
        ctx.session.state["final_video_assets_list"] = [] # Initialize output

        if not ctx.session.state.get("current_recap"): # Basic prerequisite
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping video generation.")
            return

        async for event in self.veo_prompt_generator.run_async(ctx): yield event

        raw_veo_prompts_json = ctx.session.state.get("veo_generation_prompts_json", "[]")
        cleaned_veo_prompts_json_for_tool = _clean_json_string_from_llm(raw_veo_prompts_json)
        
        generated_video_uris_list = []
        try:
            parsed_veo_prompts = json.loads(cleaned_veo_prompts_json_for_tool)
            if not isinstance(parsed_veo_prompts, list) or not parsed_veo_prompts:
                logger.info(f"[{self.name}] No valid Veo prompts. Skipping video tool call.")
            else:
                ctx.session.state["veo_generation_prompts_json_for_tool"] = cleaned_veo_prompts_json_for_tool
                # game_pk_str_for_tool should already be set by image agent or VisualAssetWorkflowAgent
                if "game_pk_str_for_tool" not in ctx.session.state : # ensure it exists
                     ctx.session.state['game_pk_str_for_tool'] = str(ctx.session.state.get("game_pk", "unknown_game"))


                logger.info(f"[{self.name}] Calling VideoGeneratorMCPCaller with {len(parsed_veo_prompts)} Veo prompts.")
                async for event in self.video_generator_mcp_caller.run_async(ctx): yield event

                tool_output_video_uris_raw = ctx.session.state.get("generated_video_clips_uris_json", '"[]"')
                tool_output_video_uris_clean = _clean_json_string_from_llm(tool_output_video_uris_raw, default_if_empty='"[]"')
                
                try:
                    parsed_video_tool_output = json.loads(tool_output_video_uris_clean)
                    if isinstance(parsed_video_tool_output, list):
                        generated_video_uris_list = [uri for uri in parsed_video_tool_output if isinstance(uri, str) and uri.startswith("gs://")]
                    # Simplified error handling for brevity
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] Failed to parse JSON from VideoGeneratorMCPCaller. Error: {e}. Output: '{tool_output_video_uris_clean}'")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[{self.name}] Error processing Veo prompts: {e}. JSON: '{cleaned_veo_prompts_json_for_tool}'")
        
        final_video_assets_list_for_this_agent = []
        for idx, uri in enumerate(generated_video_uris_list):
            final_video_assets_list_for_this_agent.append({
                "video_uri": uri, "type": "generated_video", "source_prompt_index": idx
            })
        
        ctx.session.state["final_video_assets_list"] = final_video_assets_list_for_this_agent
        logger.info(f"[{self.name}] Video clip generation finished. Generated {len(final_video_assets_list_for_this_agent)} videos.")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"Video pipeline complete. Generated {len(final_video_assets_list_for_this_agent)} videos.")]))

################################################################################
# --- REFACTORED VisualAssetWorkflowAgent (Orchestrator) ---
# --- Attempt 2: Directly calling sub-agents' run_async ---
################################################################################
class VisualAssetWorkflowAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    # These are now direct references to the phase agent instances
    static_asset_pipeline_agent: StaticAssetPipelineAgent
    iterative_image_generation_agent: IterativeImageGenerationAgent
    video_pipeline_agent: VideoPipelineAgent

    def __init__(self, name: str,
                 static_asset_pipeline_agent: StaticAssetPipelineAgent,
                 iterative_image_generation_agent: IterativeImageGenerationAgent,
                 video_pipeline_agent: VideoPipelineAgent):
        
        super().__init__(
            name=name,
            # Store direct references
            static_asset_pipeline_agent=static_asset_pipeline_agent,
            iterative_image_generation_agent=iterative_image_generation_agent,
            video_pipeline_agent=video_pipeline_agent,
            # The sub_agents list for BaseAgent should include these phase agents
            sub_agents=[static_asset_pipeline_agent, iterative_image_generation_agent, video_pipeline_agent]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting Visual Asset Workflow orchestration.")

        if not ctx.session.state.get("current_recap"):
            logger.warning(f"[{self.name}] 'current_recap' missing. Visual workflow cannot proceed.")
            ctx.session.state["all_image_assets_list"] = []
            ctx.session.state["all_video_assets_list"] = []
            return

        # --- 1. Static Asset Retrieval Path ---
        # Directly call the run_async method of the phase agent instance
        logger.info(f"[{self.name}] Invoking StaticAssetPipelineAgent: {self.static_asset_pipeline_agent.name}...")
        async for event in self.static_asset_pipeline_agent.run_async(ctx): # Pass ctx
            yield event
        logger.info(f"[{self.name}] StaticAssetPipelineAgent finished. current_static_assets_list: {len(ctx.session.state.get('current_static_assets_list', []))}")

        # --- 2. Iterative Generative Visuals Workflow ---
        logger.info(f"[{self.name}] Invoking IterativeImageGenerationAgent: {self.iterative_image_generation_agent.name}...")
        async for event in self.iterative_image_generation_agent.run_async(ctx): # Pass ctx
            yield event
        logger.info(f"[{self.name}] IterativeImageGenerationAgent finished. all_generated_image_assets_details: {len(ctx.session.state.get('all_generated_image_assets_details', []))}")

        # --- 3. Video Clip Generation Path ---
        logger.info(f"[{self.name}] Invoking VideoPipelineAgent: {self.video_pipeline_agent.name}...")
        async for event in self.video_pipeline_agent.run_async(ctx): # Pass ctx
            yield event
        logger.info(f"[{self.name}] VideoPipelineTool finished. final_video_assets_list: {len(ctx.session.state.get('final_video_assets_list', []))}")

        # --- 4. Final Asset Aggregation (remains the same) ---
        current_static_assets_list = ctx.session.state.get("current_static_assets_list", [])
        all_generated_image_assets_details = ctx.session.state.get("all_generated_image_assets_details", [])
        final_video_assets_list = ctx.session.state.get("final_video_assets_list", [])
        
        final_generated_visuals_dict = {} 
        for asset in sorted(all_generated_image_assets_details, key=lambda x: x.get("iteration", 0)):
            if asset.get("image_uri"):
                final_generated_visuals_dict[asset["image_uri"]] = asset 
        
        all_image_assets_list = current_static_assets_list + list(final_generated_visuals_dict.values())
        ctx.session.state["all_image_assets_list"] = all_image_assets_list
        ctx.session.state["all_video_assets_list"] = final_video_assets_list

        num_static = len(current_static_assets_list)
        num_generated_unique_images = len(final_generated_visuals_dict)
        num_videos = len(final_video_assets_list)
        
        logger.info(f"[{self.name}] Visual asset workflow finished. Static: {num_static}, Generated Images: {num_generated_unique_images}, Videos: {num_videos}.")
        logger.info(f"[{self.name}] Total unique image assets in list: {len(all_image_assets_list)}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"Visual workflow complete. Static: {num_static}, Images: {num_generated_unique_images}, Videos: {num_videos}.")]))




################################################################################
# --- NEW CUSTOM AGENT - AudioProcessingPipelineAgent ---
################################################################################
class AudioProcessingPipelineAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    dialogue_to_speech_agent: LlmAgent
    audio_to_timestamps_agent: LlmAgent

    def __init__(self, name: str,
                 dialogue_to_speech_agent: LlmAgent,
                 audio_to_timestamps_agent: LlmAgent):
        super().__init__(
            name=name,
            dialogue_to_speech_agent=dialogue_to_speech_agent,
            audio_to_timestamps_agent=audio_to_timestamps_agent,
            sub_agents=[dialogue_to_speech_agent, audio_to_timestamps_agent]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting audio processing pipeline.")
        # Initialize outputs in session state
        ctx.session.state["generated_dialogue_audio_uri"] = None
        ctx.session.state["word_timestamps_list"] = []

        current_recap = ctx.session.state.get("current_recap")
        if not current_recap:
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping audio processing.")
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Audio processing skipped: no recap.")]))
            return

        game_pk_for_audio = str(ctx.session.state.get("game_pk", "unknown_game"))
        # Ensure game_pk is explicitly set for the DialogueToSpeechAgent if its instruction expects it directly
        # However, the instruction above for DialogueToSpeechAgent says it will retrieve 'game_pk' from session state.

        # 1. Generate Speech from Dialogue
        logger.info(f"[{self.name}] Running DialogueToSpeechAgent for game_pk: {game_pk_for_audio}...")
        async for event in self.dialogue_to_speech_agent.run_async(ctx):
            yield event # Yield events from the sub-agent

        raw_audio_details_json = ctx.session.state.get("generated_dialogue_audio_details_json", "{}")
        # _clean_json_string_from_llm might not be necessary if the LlmAgent output_key directly gets the tool's JSON string
        # but it's safer if the LLM ever wraps it.
        cleaned_audio_details_json = _clean_json_string_from_llm(raw_audio_details_json, default_if_empty='{}')
        
        audio_uri_for_stt = None 
        try:
            audio_details = json.loads(cleaned_audio_details_json)
            if isinstance(audio_details, dict):
                if audio_details.get("error"):
                    logger.error(f"[{self.name}] DialogueToSpeechAgent returned an error: {audio_details['error']}")
                else:    
                    potential_uri = audio_details.get("uri") or audio_details.get("audio_uri")
                    if potential_uri and isinstance(potential_uri, str) and potential_uri.startswith("gs://"):
                        audio_uri_for_stt = potential_uri
                        ctx.session.state["generated_dialogue_audio_uri"] = audio_uri_for_stt # Keep original key for other uses if any
                        logger.info(f"[{self.name}] Valid GCS URI for STT: {audio_uri_for_stt}")
                    else: 
                        logger.warning(f"[{self.name}] DialogueToSpeechAgent returned JSON with missing or invalid GCS URI: {audio_details}. Potential URI was: '{potential_uri}'")
            else:
                logger.error(f"[{self.name}] DialogueToSpeechAgent output was not a JSON dict: {cleaned_audio_details_json}")

        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse JSON from DialogueToSpeechAgent. Error: {e}. JSON string: '{cleaned_audio_details_json}'")

        # 2. Get Word Timestamps (if audio was generated)
        if audio_uri_for_stt:
            logger.info(f"[{self.name}] Running AudioToTimestampsAgent for audio: {audio_uri_for_stt}...")
            # The AudioToTimestampsAgent's instruction tells it to pick up 'generated_dialogue_audio_uri' from state.
            async for event in self.audio_to_timestamps_agent.run_async(ctx):
                yield event

            raw_timestamps_json = ctx.session.state.get("word_timestamps_json", "[]")
            cleaned_timestamps_json = _clean_json_string_from_llm(raw_timestamps_json, default_if_empty='[]')
            
            try:
                timestamps_data = json.loads(cleaned_timestamps_json)
                if isinstance(timestamps_data, list):
                    ctx.session.state["word_timestamps_list"] = timestamps_data
                    logger.info(f"[{self.name}] Successfully retrieved {len(timestamps_data)} word timestamps.")
                elif isinstance(timestamps_data, dict) and timestamps_data.get("error"):
                    logger.error(f"[{self.name}] AudioToTimestampsAgent returned an error: {timestamps_data['error']}")
                    ctx.session.state["word_timestamps_list"] = [] # Ensure it's a list for consistency
                else:
                    logger.warning(f"[{self.name}] AudioToTimestampsAgent returned unexpected JSON: {timestamps_data}")
                    ctx.session.state["word_timestamps_list"] = []

            except json.JSONDecodeError as e:
                logger.error(f"[{self.name}] Failed to parse JSON from AudioToTimestampsAgent. Error: {e}. JSON string: '{cleaned_timestamps_json}'")
                ctx.session.state["word_timestamps_list"] = []
        else:
            logger.warning(f"[{self.name}] Skipping timestamp generation as dialogue audio URI was not available.")
            ctx.session.state["word_timestamps_list"] = []


        final_message = f"Audio processing pipeline complete. Audio URI: {ctx.session.state['generated_dialogue_audio_uri']}, Timestamps found: {len(ctx.session.state.get('word_timestamps_list', []))}"
        logger.info(f"[{self.name}] {final_message}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=final_message)]))


################################################################################
# --- AssetValidationAndRetryAgent Definition ---
################################################################################
class AssetValidationAndRetryAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    static_asset_pipeline_agent: StaticAssetPipelineAgent
    iterative_image_generation_agent: IterativeImageGenerationAgent
    video_pipeline_agent: VideoPipelineAgent
    audio_processing_pipeline_agent: AudioProcessingPipelineAgent
    max_retries_per_pipeline: int

    def __init__(self, name: str,
                 static_asset_pipeline_agent: StaticAssetPipelineAgent,
                 iterative_image_generation_agent: IterativeImageGenerationAgent,
                 video_pipeline_agent: VideoPipelineAgent,
                 audio_processing_pipeline_agent: AudioProcessingPipelineAgent,
                 max_retries_per_pipeline: int = 1):
        super().__init__(
            name=name,
            # These are Pydantic fields for the agent's configuration/state
            static_asset_pipeline_agent=static_asset_pipeline_agent,
            iterative_image_generation_agent=iterative_image_generation_agent,
            video_pipeline_agent=video_pipeline_agent,
            audio_processing_pipeline_agent=audio_processing_pipeline_agent,
            max_retries_per_pipeline=max_retries_per_pipeline,
            # AssetValidationAndRetryAgent itself does not have exclusive sub-components
            # that are not already parented elsewhere. The agents it interacts with
            # are referenced via its fields but are not its children in the ADK agent tree.
            sub_agents=[]
        )
        # Store references to the agents it will manage/retry.
        # Pydantic already handles setting these as attributes if they are defined
        # in the class and passed to super().__init__(**kwargs).
        # Explicit assignment here is fine for clarity or if BaseAgent doesn't assign all kwargs.
        self.static_asset_pipeline_agent = static_asset_pipeline_agent
        self.iterative_image_generation_agent = iterative_image_generation_agent
        self.video_pipeline_agent = video_pipeline_agent
        self.audio_processing_pipeline_agent = audio_processing_pipeline_agent
        # self.max_retries_per_pipeline is already set by Pydantic
    
    # ... (rest of AssetValidationAndRetryAgent._run_async_impl method remains the same)
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting asset validation and retry process.")
        
        if "asset_retry_counts" not in ctx.session.state:
            ctx.session.state["asset_retry_counts"] = {
                "static": 0, "image_gen": 0, "video_gen": 0, "audio_gen": 0
            }
        retry_counts = ctx.session.state["asset_retry_counts"]

        if not ctx.session.state.get("current_recap"):
            logger.warning(f"[{self.name}] No current_recap. Skipping validation.")
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Asset validation skipped: no recap.")]))
            return

        # --- Static Assets Validation & Retry ---
        static_assets = ctx.session.state.get("current_static_assets_list", [])
        if not static_assets and retry_counts["static"] < self.max_retries_per_pipeline:
            logger.warning(f"[{self.name}] Static assets missing. Retrying (attempt {retry_counts['static'] + 1}).")
            retry_counts["static"] += 1
            ctx.session.state["current_static_assets_list"] = [] # Clear previous
            async for event in self.static_asset_pipeline_agent.run_async(ctx): yield event
            logger.info(f"[{self.name}] Static assets retry complete. Found: {len(ctx.session.state.get('current_static_assets_list', []))}")
        elif not static_assets:
            logger.warning(f"[{self.name}] Static assets still missing after max retries.")

        # --- Generated Images Validation & Retry ---
        generated_images = ctx.session.state.get("all_generated_image_assets_details", [])
        if not generated_images and retry_counts["image_gen"] < self.max_retries_per_pipeline:
            logger.warning(f"[{self.name}] Generated images missing. Retrying (attempt {retry_counts['image_gen'] + 1}).")
            retry_counts["image_gen"] += 1
            ctx.session.state["all_generated_image_assets_details"] = [] # Clear previous
            async for event in self.iterative_image_generation_agent.run_async(ctx): yield event
            logger.info(f"[{self.name}] Generated images retry complete. Found: {len(ctx.session.state.get('all_generated_image_assets_details', []))}")
        elif not generated_images:
            logger.warning(f"[{self.name}] Generated images still missing after max retries.")

        # --- Video Clips Validation & Retry ---
        generated_videos = ctx.session.state.get("final_video_assets_list", [])
        if not generated_videos and retry_counts["video_gen"] < self.max_retries_per_pipeline:
            logger.warning(f"[{self.name}] Generated videos missing. Retrying (attempt {retry_counts['video_gen'] + 1}).")
            retry_counts["video_gen"] += 1
            ctx.session.state["final_video_assets_list"] = [] # Clear previous
            async for event in self.video_pipeline_agent.run_async(ctx): yield event
            logger.info(f"[{self.name}] Generated videos retry complete. Found: {len(ctx.session.state.get('final_video_assets_list', []))}")
        elif not generated_videos:
             logger.warning(f"[{self.name}] Generated videos still missing after max retries.")

        # --- Audio Processing Validation & Retry ---
        if hasattr(self, 'audio_processing_pipeline_agent') and self.audio_processing_pipeline_agent:
            generated_audio = ctx.session.state.get("generated_dialogue_audio_uri")
            if not generated_audio and retry_counts["audio_gen"] < self.max_retries_per_pipeline:
                logger.warning(f"[{self.name}] Generated audio missing. Retrying (attempt {retry_counts['audio_gen'] + 1}).")
                retry_counts["audio_gen"] += 1
                ctx.session.state["generated_dialogue_audio_uri"] = None 
                ctx.session.state["word_timestamps_list"] = []
                async for event in self.audio_processing_pipeline_agent.run_async(ctx): yield event
                logger.info(f"[{self.name}] Generated audio retry complete. URI: {ctx.session.state.get('generated_dialogue_audio_uri')}")
            elif not generated_audio:
                 logger.warning(f"[{self.name}] Generated audio still missing after max retries.")
        else:
            logger.info(f"[{self.name}] Audio processing pipeline agent not available for validation/retry.")


        ctx.session.state["asset_retry_counts"] = retry_counts 

        current_static_assets = ctx.session.state.get("current_static_assets_list", [])
        all_generated_image_details = ctx.session.state.get("all_generated_image_assets_details", [])
        final_generated_visuals_dict = {}
        for asset in sorted(all_generated_image_details, key=lambda x: x.get("iteration", 0)):
            if asset.get("image_uri"):
                final_generated_visuals_dict[asset["image_uri"]] = asset
        ctx.session.state["all_image_assets_list"] = current_static_assets + list(final_generated_visuals_dict.values())
        
        logger.info(f"[{self.name}] Asset validation and retry process finished.")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Asset validation and retry efforts complete.")]))



################################################################################
# --- GameRecapAgent Definition (Adjusted to use refactored VisualAssetWorkflowAgent) ---
################################################################################
class GameRecapAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    initial_recap_generator: LlmAgent
    recap_critic: LlmAgent
    critique_processor: LlmAgent
    recap_reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent
    refinement_loop: LoopAgent
    post_processing_sequence: SequentialAgent
    visual_asset_workflow_orchestrator: VisualAssetWorkflowAgent # Changed variable name for clarity
    audio_processing_pipeline_agent: AudioProcessingPipelineAgent
    asset_validator: AssetValidationAndRetryAgent 

    def __init__(
        self,
        name: str,
        initial_recap_generator: LlmAgent,
        recap_critic: LlmAgent,
        critique_processor: LlmAgent,
        recap_reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
        visual_asset_workflow_orchestrator: VisualAssetWorkflowAgent, # Pass the orchestrator
        audio_processing_pipeline_agent: AudioProcessingPipelineAgent,
    ):
        refinement_loop = LoopAgent(
            name="RecapRefinementLoop",
            sub_agents=[recap_critic, critique_processor, recap_reviser],
            max_iterations=1
        )
        post_processing_sequence = SequentialAgent(
            name="RecapPostProcessing",
            sub_agents=[grammar_check, tone_check]
        )
        
        asset_validator_instance = AssetValidationAndRetryAgent(
            name="AssetValidationAndRetry",
            static_asset_pipeline_agent=visual_asset_workflow_orchestrator.static_asset_pipeline_agent,
            iterative_image_generation_agent=visual_asset_workflow_orchestrator.iterative_image_generation_agent,
            video_pipeline_agent=visual_asset_workflow_orchestrator.video_pipeline_agent,
            audio_processing_pipeline_agent=audio_processing_pipeline_agent, # Pass the audio agent instance
            max_retries_per_pipeline=1 # Set desired max retries, 0 to disable
)
        sub_agents_list = [
            initial_recap_generator,
            refinement_loop,
            post_processing_sequence,
            visual_asset_workflow_orchestrator, # This is the refactored orchestrator
            audio_processing_pipeline_agent,
            asset_validator_instance,
        ]
        super().__init__(
            name=name,
            initial_recap_generator=initial_recap_generator,
            recap_critic=recap_critic,
            critique_processor=critique_processor,
            recap_reviser=recap_reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            refinement_loop=refinement_loop,
            post_processing_sequence=post_processing_sequence,
            visual_asset_workflow_orchestrator=visual_asset_workflow_orchestrator,
            audio_processing_pipeline_agent=audio_processing_pipeline_agent, 
            sub_agents=sub_agents_list,
            asset_validator=asset_validator_instance,
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting full game recap and visual workflow.")

        # --- 0. Pre-requisites ---
        user_query = ctx.session.state.get("user_query") # Simplified, see original for full logic
        if not user_query:
            user_query = "generic recap request from GameRecapAgent" # Fallback
            ctx.session.state["user_query"] = user_query
            if ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text:
                 try:
                    data = json.loads(ctx.user_content.parts[0].text)
                    if isinstance(data, dict) and "message" in data: user_query = data["message"]
                    else: user_query = ctx.user_content.parts[0].text
                 except json.JSONDecodeError: user_query = ctx.user_content.parts[0].text
                 ctx.session.state["user_query"] = user_query
            logger.info(f"[{self.name}] User query set to: '{user_query}'")


        if "player_lookup_dict_json" not in ctx.session.state: # Ensure for visual workflow
            player_id_map = ctx.session.state.get("player_id_to_name_map")
            if player_id_map and isinstance(player_id_map, dict):
                ctx.session.state["player_lookup_dict_json"] = json.dumps(player_id_map)
            else:
                ctx.session.state["player_lookup_dict_json"] = "{}" # Default

        # --- 1. Dialogue Generation & Refinement ---
        logger.info(f"[{self.name}] Running InitialRecapGenerator (Dialogue)...")
        async for event in self.initial_recap_generator.run_async(ctx): yield event
        
        dialogue_after_initial_gen = ctx.session.state.get("current_recap", "")
        agent_should_exit = ctx.session.state.get("agent_should_exit_flag", False)
        if agent_should_exit or not dialogue_after_initial_gen or "game is not final" in dialogue_after_initial_gen.lower():
            message = dialogue_after_initial_gen or "Recap generation stopped early."
            logger.warning(f"[{self.name}] Initial recap phase indicated exit. Message: {message}")
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=message)]))
            return

        logger.info(f"[{self.name}] Running Dialogue RefinementLoop...")
        async for event in self.refinement_loop.run_async(ctx): yield event
        
        logger.info(f"[{self.name}] Running Dialogue PostProcessing Sequence...")
        async for event in self.post_processing_sequence.run_async(ctx): yield event

        final_dialogue_recap = ctx.session.state.get("current_recap", "")
        if not final_dialogue_recap:
            logger.error(f"[{self.name}] Dialogue recap is empty after refinement. Aborting.")
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Failed to produce dialogue recap.")]))
            return
        logger.info(f"[{self.name}] Dialogue workflow finished.")

        # --- 2. Visual Asset Workflow (using the orchestrator) ---
        logger.info(f"[{self.name}] Running VisualAssetWorkflow Orchestrator...")
        async for event in self.visual_asset_workflow_orchestrator.run_async(ctx):
            # Yield events from the orchestrator. It will internally call phase agents.
            yield event
        
        all_image_assets = ctx.session.state.get("all_image_assets_list", [])
        all_video_assets = ctx.session.state.get("all_video_assets_list", [])
        logger.info(f"[{self.name}] VisualAssetWorkflow Orchestrator finished. Images: {len(all_image_assets)}, Videos: {len(all_video_assets)}.")

        # --- 3. Audio Processing Workflow (NEW) ---
        if final_dialogue_recap: # Only run audio if dialogue exists
            logger.info(f"[{self.name}] Running AudioProcessingPipelineAgent...")
            async for event in self.audio_processing_pipeline_agent.run_async(ctx):
                yield event
            generated_audio_uri = ctx.session.state.get("generated_dialogue_audio_uri")
            word_timestamps = ctx.session.state.get("word_timestamps_list", [])
            logger.info(f"[{self.name}] AudioProcessingPipelineAgent finished. Audio URI: {generated_audio_uri}, Timestamps: {len(word_timestamps)}.")
        else:
            logger.warning(f"[{self.name}] Skipping Audio Processing as final dialogue recap is empty.")

        # --- Asset Validation and Retry Step ---
        if self.asset_validator.max_retries_per_pipeline > 0: # Conditionally run
           logger.info(f"[{self.name}] Running AssetValidationAndRetryAgent...")
           async for event in self.asset_validator.run_async(ctx):
              yield event
           logger.info(f"[{self.name}] AssetValidationAndRetryAgent finished.")
           # The session state (component asset lists and all_image_assets_list)
           # will be updated by the AssetValidationAndRetryAgent.
        
        # --- Update Final Asset Counts for Logging (using the potentially updated lists) ---
        all_image_assets = ctx.session.state.get("all_image_assets_list", [])
        all_video_assets = ctx.session.state.get("all_video_assets_list", [])
        # Potentially log other asset counts like static, generated images separately if needed
        num_static = len(ctx.session.state.get("current_static_assets_list", []))
        num_generated_unique_images = len(ctx.session.state.get("all_generated_image_assets_details", [])) # Or derive from all_image_assets
        num_videos = len(all_video_assets)
        generated_audio_uri = ctx.session.state.get("generated_dialogue_audio_uri")
        word_timestamps = ctx.session.state.get("word_timestamps_list", [])

        logger.info(f"[{self.name}] Post-Validation - Static: {num_static}, Gen.Images: {num_generated_unique_images}, Videos: {num_videos}, Audio: {'Yes' if generated_audio_uri else 'No'}, Timestamps: {len(word_timestamps)}")

        # --- 4. Final Output Event for GameRecapAgent ---
        logger.info(f"[{self.name}] Yielding final dialogue recap.")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=final_dialogue_recap)]))
        logger.info(f"[{self.name}] === GameRecapAgent processing complete. ===")


# --- Tool Collection (Assumed unchanged) ---
async def _collect_tools_stack(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    stack_needs_closing = False # Flag to manage closing only if successfully opened
    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(
            server_config_dict.configs, dict
        ):
            logging.error("server_config_dict does not have a valid '.configs' dictionary.")
            return {}, exit_stack # Return empty and non-closable stack

        for key, server_params in server_config_dict.configs.items():
            individual_exit_stack: Optional[contextlib.AsyncExitStack] = None
            try:
                tools, individual_exit_stack = await MCPToolset.from_server(
                    connection_params=server_params
                )
                if individual_exit_stack:
                    # Only enter context if stack is valid, making it closable
                    await exit_stack.enter_async_context(individual_exit_stack)
                    stack_needs_closing = True # Mark that the main stack now needs closing
                
                if tools:
                    all_tools[key] = tools
                    logger.info(f"Successfully collected tools for MCP server: {key}")
                else:
                    logging.warning("Connection successful for key '%s', but no tools returned.", key)

            except FileNotFoundError as file_error:
                logging.error("Command or script not found for key '%s': %s", key, file_error)
            except ConnectionRefusedError as conn_refused:
                logging.error("Connection refused for key '%s': %s", key, conn_refused)
            except Exception as e: # Catch more specific MCP connection errors if possible
                logging.error(f"Failed to connect or get tools for {key}: {e}", exc_info=True)
                # Do not attempt to close individual_exit_stack here if it failed to initialize

        if not all_tools:
            logging.warning("No tools were collected from any server.")
        
        # Ensure all expected keys exist, even if empty, for robust LlmAgent tool access
        expected_tool_keys = ["weather", "bnb", "ct", "mlb", "web_search", "bq_search", 
                              "visual_assets", "static_retriever_mcp", 
                              "image_embedding_mcp", "video_clip_generator_mcp", "audio_processing_mcp"]
        for k in expected_tool_keys:
            if k not in all_tools:
                logging.info("Tools for key '%s' were not collected. Ensuring key exists with empty list.", k)
                all_tools[k] = [] # Provide an empty list to prevent KeyErrors later

        return all_tools, exit_stack

    except Exception as e: # Catch errors in the outer loop of _collect_tools_stack
        logging.error("Unhandled exception in _collect_tools_stack: %s", e, exc_info=True)
        if stack_needs_closing: # Only close if it was successfully opened and entered
            await exit_stack.aclose()
        # Reraise or handle as appropriate for your application lifecycle
        raise # Or return {}, exit_stack to allow app to start in degraded state



# --- Agent Creation ---
async def create_agent_with_preloaded_tools(
    loaded_mcp_tools: Dict[str, Any],
) -> LlmAgent:
    booking_tools = loaded_mcp_tools.get("bnb", [])
    weather_tools = loaded_mcp_tools.get("weather", [])
    combined_booking_tools = list(booking_tools) + list(weather_tools) # Ensure they are lists
    ct_tools = loaded_mcp_tools.get("ct", [])
    mlb_tools = loaded_mcp_tools.get("mlb", [])
    web_search_tools = loaded_mcp_tools.get("web_search", [])
    bq_search_tools = loaded_mcp_tools.get("bq_search", [])

    # Tools for GameRecapAgent and its sub-agents
    # These LlmAgents will need specific tools from the broader set.
    game_recap_tool_list = list(mlb_tools) + list(web_search_tools) + list(bq_search_tools)


    booking_agent = LlmAgent(
        model=MODEL_ID,
        name="booking_assistant",
        instruction="""Use booking_tools to handle inquiries related to
        booking accommodations and weather information.
        Format your response using Markdown.
        If you don't know how to help, call "agent_exit".""",
        tools=combined_booking_tools,
    )

    cocktail_agent = LlmAgent(
        model=MODEL_ID,
        name="cocktail_assistant",
        instruction="""Use ct_tools to handle all inquiries related to cocktails.
        Format your response using Markdown.
        If you don't know how to help, call "agent_exit".""",
        tools=ct_tools,
    )

    mlb_assistant = LlmAgent(
         model=MODEL_ID,
         name="mlb_assistant",
         instruction="""You are an MLB Stats assistant.
         Use your tools (e.g., `mlb.get_live_game_score`, `mlb.get_team_schedule`) to answer questions about Major League Baseball.
         If the user does not provide all necessary IDs (like `game_pk`, `player_id`, `team_id`), you MUST ask for them before calling the tool.
         If a user asks for a game recap, and you identify a game_pk, you should state that you can provide stats and scores, but for a full recap, they might want to ask the main assistant to route them to the recap specialist. Or, provide the game_pk and use agent_exit.
         Format your responses clearly using Markdown.
         If you cannot help, use "agent_exit".""",
         tools=mlb_tools,
    )

    # --- Sub-Agents for GameRecapAgent ---
    # Ensure `game_pk` is consistently available in session state for these agents.
    # `user_query` or a `task_description` should also be in session state.
    
    # Note: Tool names used by LlmAgent instructions should match how ADK makes them available
    # e.g., if bq_search_tools contains a tool `search_past_critiques` from server "bq_search",
    # it might be callable as `bq_search.search_past_critiques` in the prompt.

    initial_recap_generator_agent = LlmAgent(
        name="InitialRecapGenerator",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an expert sports journalist tasked with generating an initial MLB game recap in a **two-host dialogue script format**. Your goal is to create a compelling narrative of the game, not just a list of events,  presented as a conversation.

Session State Expectations:
- `game_pk` (e.g., 717527): The unique ID for the game.
- `user_query`: The user's original request (e.g., "recap of Pirates last game").
- `parsed_game_pk_from_query` (Optional): A game_pk parsed directly from the user's query. Prioritize this.
- `pre_game_context_notes` (Optional): Web search findings about rivalry, storylines, etc.
- `past_critiques_feedback` (Optional): General feedback from past similar tasks to guide style and tone.

Your Multi-Step Process:

1.  **Game Identification (Critical):**
    *   If `game_pk` is not in session state: Analyze `user_query`.
    *   If the query implies a "latest" game (e.g., "Brewers last game"), use `mlb.get_team_schedule` (e.g., `days_range=-7`) to find the most recent *final* game. Extract its `game_pk`. If ambiguity, ask for clarification.
    *   Announce the identified game: "Okay, I've identified the game: Team A vs Team B on YYYY-MM-DD, Game PK: [game_pk]." Update `session.state.game_pk`.
    *   If no specific game can be confidently identified, state this and request clarification. Do NOT invent a `game_pk`. Use `agent_exit`.

2.  **Gather Core Game Data (if `game_pk` is now known):**
    *   Call `mlb.get_live_game_score` for `game_pk`.
    *   Call `mlb.get_game_play_by_play_summary` for `game_pk` (get enough plays, e.g., 15-20, to understand key moments).
    *   Call `mlb_stats.get_game_boxscore_and_details` for `game_pk`.
    *   Parse the data: game status, final score, winning/losing pitchers, key offensive performers (multi-hit, RBIs, HRs), inning-by-inning scores.
    *   **If Game Status is "Scheduled" or not "Final"**: Your output MUST state that a full recap is not yet available as the game is not final. Then use `agent_exit`.

3.  **Synthesize Initial Dialogue Script (Only if Game is "Final"):**
    *   **Dialogue Format:**
        *   The entire output MUST be a conversation between two hosts (e.g., Host A, Host B).
        *   **Strict Alternation:** Each line of the script MUST represent one host speaking, alternating strictly.
        *   **NO Speaker Labels:** CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:", or any character names. Just write the raw dialogue line for each speaker's turn.
    *   **Storytelling First:** Your primary goal is to tell the story of the game through this dialogue.
        *   Identify a potential "story of the game" (e.g., a pitcher's duel, an offensive breakout, a key player's heroics, a specific turning point).
        *   The dialogue should start with an engaging lead, perhaps one host setting the scene and the other reacting or adding initial thoughts, summarizing the game's outcome and main storyline.
    *   **Pitching Narrative:** The hosts should discuss the performance of the starting pitchers, especially the winner and loser.
    *   **Offensive Highlights & Progression:**
        *   The dialogue should cover how the scoring unfolded, focusing on the most impactful plays.
        *   Hosts should name the players involved in these key offensive moments and discuss their contributions.
    *   **Integrate Context:**
        *   If `pre_game_context_notes` are available, one host might bring it up, and the other can elaborate or connect it to game events.
    *   **Guidance from Past Critiques:** Use `past_critiques_feedback` (if available) for general guidance on narrative structure, tone, and conversational style.
    *   **Language:** Use vivid, active language. The dialogue should sound natural and engaging. Avoid one host just listing stats for the other to react to; make it a genuine discussion.
    *   **Acknowledge Limitations:** If specific details are unavailable, one host might pose it as a question the other can't fully answer, or they might acknowledge the gap.

Output ONLY the generated recap text. Do not add conversational fluff like "Here is the recap..." or "I have gathered the data...".
        """,
        tools=[ # Ensure all necessary tools are listed
            tool for toolset_name in ["bq_search", "mlb", "web_search"] 
            for tool in loaded_mcp_tools.get(toolset_name, [])
        ],
        output_key="current_recap",
    )

    recap_critic_agent = LlmAgent(
        name="RecapCritic",
        model=MODEL_ID, # Can be a faster model
        instruction="""
You are a sharp, demanding MLB analyst and broadcast producer acting as a writing critic.
Expected in session state: `current_recap` (which is a two-host dialogue script), `game_pk`, `user_query`.

Review the `current_recap` dialogue script. Provide constructive, actionable criticism. Focus on:

- **Dialogue Flow & Engagement:**
    - Does the conversation between the hosts sound natural? Is the back-and-forth engaging?
    - Do the hosts have distinct enough 'voices' or perspectives, or do they sound too similar?
    - Is it a real discussion, or does one host merely set up the other?
- **Accuracy & Completeness (within the dialogue):**
    - Are scores, key player actions, and game sequence correctly and sufficiently detailed *as discussed by the hosts*?
- **Narrative & Engagement (of the dialogue itself):**
    - Does the *conversation* tell a compelling story? Does it have a clear narrative arc?
    - Does the dialogue capture tension, excitement, or the "story" of the game?
    - Is the language used by the hosts engaging, vivid, and journalistic?
- **Journalistic Style (of the dialogue):**
    - Does the dialogue sound like a professional sports podcast or broadcast segment?
- **Information Gaps & Opportunities for Enrichment (within the dialogue):**
    - What key information are the hosts *not* discussing that would enhance the story?
    - Are there opportunities for one host to introduce more stats or context for the other to react to?
- **Clarity & Flow (of the dialogue):**
    - Is the conversation easy to follow? Are the hosts' lines clear and concise?
- **Data Usage (by the hosts):**
    - Are stats used effectively by the hosts to support their points, or just dropped in?

If the dialogue script is excellent and requires no changes (rare!), respond ONLY with "The recap is excellent."
Otherwise, provide **specific, bulleted feedback** with clear examples of what needs improvement in the dialogue or what specific information the hosts should discuss.
        """,
        output_key="current_critique",
    )

    critique_processor_agent = LlmAgent(
        name="CritiqueProcessor",
        model=GEMINI_PRO_MODEL_ID, # Use a capable model for this multi-step reasoning
        instruction="""You are a specialized research assistant and data coordinator.
        Expected in session state: `current_critique`, `game_pk`, `user_query`.

        Your multi-step task is to process the `current_critique` and gather information for revision:

        1.  **Store the Critique:**
            *   Call the `bq_search.store_new_critique` tool.
            *   Use parameters:
                *   `critique_text` = {session.state.current_critique}
                *   `task_text` = "recap for game_pk {session.state.game_pk} based on user query '{session.state.user_query}'" (ensure `game_pk` and `user_query` are from session state).
                *   `game_pk_str` = If `session.state.game_pk` has an integer value, convert it to its string representation (e.g., if game_pk is 777930, pass "777930") and/or If `session.state.game_pk` is null/None or not present, pass an **empty string.
                *   `revision_number_str` = string representation of {session.state.revision_number} (if available, otherwise empty string).
            *   Let the result of this tool call be `critique_storage_status_json`.

        2.  **Generate Targeted Web Search Queries from Critique:**
            *   Carefully analyze the `current_critique`. Identify 1-3 key questions or information gaps highlighted by the critique that could be addressed with a web search (e.g., specific missing actual details, player's recent performance trends, injury news before the game, context of a rivalry).
            *   Formulate these as concise, effective search queries suitable for the Tavily search engine.
            *   Broader context if the critique implies it's missing (e.g., "series implications for [Team A]", "historical significance of [Team A] vs [Team B] matchup", "player [Player Name] recent performance trend").
            *   Let this list of query strings be `generated_tavily_queries`. (This is an internal thought process; you will use these queries in the next step).

        3.  **Perform Web Searches:**
            *   If you generated any `generated_tavily_queries` in step 2:
                *   For each query in `generated_tavily_queries` (max 3 queries total):
                    *   Call the `web_search.perform_web_search` tool with the `query` and `max_results=1` (or 2 if more context is needed).
                *   Collect all results. Let the combined list of web search result strings be `web_search_findings_list`. If no results, this should be an empty list or a list with "No relevant web results found."

        4.  **Perform RAG Document Search (Contextual Game Info):**
            *   Call `bq_search.search_rag_documents` using:
                *   `query_text` = {session.state.current_critique} (to find RAG docs relevant to the critique points)
                *   `game_pk_str` = string representation of {session.state.game_pk}
                *   `top_n` = 2.
            *   Let the result (a JSON string list of document contents) be `rag_findings_json_list`.

        5.  **Assemble Final JSON Output:**
            *   Construct a single JSON string as your output. This JSON object must have the following keys:
                - `"critique_storage_status"`: (string) The `critique_storage_status_json` obtained from step 1.
                - `"web_search_queries_generated"`: (list of strings) The `generated_tavily_queries` you formulated in step 2.                
                - `"web_search_findings"`: (list of strings) The `web_search_findings_list` from step 3.
                - `"rag_findings"`: (list of strings) Parse `rag_findings_json_list` from step 4 into a Python list of strings.
                - `"overall_status_message"`: (string) A brief confirmation, e.g., "Critique processed. Web and RAG searches performed based on critique."

            Example JSON output format:
            ```json
            {{
              "critique_storage_status": "{{\"status\": \"success\", \"critique_id\": \"xyz123\"}}",
              "web_search_findings": ["Tavily: Player X was indeed recovering from a minor injury before the game.", "Tavily: The rivalry dates back to a controversial playoff series in 2010."],
              "rag_findings": ["RAG: The game summary highlighted the manager's post-game comments on the team's resilience.", "RAG: Detailed play analysis shows the turning point was the 7th inning double play."],
              "overall_status_message": "Critique processed. Web and RAG searches performed to address critique points."
            }}
            ```
        Ensure your entire output is ONLY this single, valid JSON string. Do not add any explanatory text before or after the JSON.
        If a step yields no data (e.g., no web queries generated, or searches return nothing), represent that with empty lists (e.g., `"web_search_findings": []`) in the final JSON.
        """,
        tools=[
             tool for toolset_name in ["bq_search", "web_search"] for tool in loaded_mcp_tools.get(toolset_name, [])
        ],
        output_key="critique_processor_results_json"
    )

    recap_reviser_agent = LlmAgent(
        name="RecapReviser",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an expert sports story editor and reviser, tasked with transforming a game recap **dialogue script** into a polished, engaging piece of sports journalism, maintaining the two-host conversational format.

Session State Expectations:
- `current_recap`: The existing version of the game recap.
- `current_critique`: The critique to address, focusing on narrative, detail, and journalistic style.
- `critique_processor_results_json`: A JSON string containing:
    - `critique_storage_status` (string)
    - `web_search_findings` (list of strings from web searches based on the critique)
    - `rag_findings` (list of strings from RAG document searches based on the critique)
- `user_query`: The original user request.
- `past_critiques_feedback` (Optional): General learnings from similar past tasks.
- `live_score_data`, `pbp_summary_data`, `comprehensive_data_json`: Core game data from the InitialRecapGenerator phase, which might be needed for cross-referencing or extracting further detail if the critique pointed to a specific factual gap not covered by new web/RAG findings.

Your Task:
1.  **Parse Research:** Parse the JSON string in `session.state.critique_processor_results_json`. Extract `web_search_findings` and `rag_findings`.
2.  **Address Critique Holistically (Maintaining Dialogue Format):**
    *   Thoroughly revise the `current_recap` (which is a dialogue script) to address *every actionable point* in `current_critique`.
    *   **Integrate Research Narratively:** Seamlessly weave in relevant information from `web_search_findings` and `rag_findings` into the *dialogue*. For example, one host might present a new finding, and the other can react or build upon it.
    *   **Enhance Storytelling & Dialogue Flow:** Elevate the language within the hosts' lines. Ensure the back-and-forth is natural and engaging.
    *   **Refine Narrative Arc:** Ensure the dialogue has a clear lead, develops the game's key moments and turning points through the hosts' discussion, and concludes effectively.
    *   **Contextualize Performances:** The hosts should discuss stats and their significance within the conversation.
3.  **Apply Stylistic Guidance (for Dialogue):**
    *   Incorporate general stylistic advice from `session.state.past_critiques_feedback` applicable to conversational sports commentary.
    *   Ensure the tone of the dialogue is appropriate.
4.  **Fulfill Original Request:** Double-check that the revised dialogue script comprehensively and engagingly addresses the `user_query`.
5.  **Clarity and Conciseness within Dialogue:** Ensure each host's lines are clear and the overall conversation flows well.
6.  **Maintain Dialogue Format:**
    *   The entire revised output MUST remain a conversation between two hosts.
    *   **Strict Alternation:** Each line of the script MUST represent one host speaking, alternating strictly.
    *   **NO Speaker Labels:** CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:".

Your final output MUST BE ONLY the revised and improved game dialogue script text, with each speaker's line on a new line. Do not include any conversational intros, outros, or explanations about the changes you made.
        """,
        output_key="current_recap",
    )


    grammar_check_agent = LlmAgent(
        name="RecapGrammarCheck",
        model=MODEL_ID,
        instruction="""
You are a grammar and style checker for sports journalism.
Expected in session state: `current_recap`.

Review the `current_recap` for grammatical errors, awkward phrasing, and areas where the language could be more impactful or active, fitting for a professional sports recap.
Output only a JSON list of concise, actionable suggestions. If the grammar and style are excellent, output an empty list `[]` or a list containing the string "Grammar and style are good."

Example of a suggestion:
"In paragraph 2, sentence 3: 'He allowed just two hits' could be more active, e.g., 'He yielded only two hits' or 'He surrendered just two hits.'"
""",
        output_key="grammar_suggestions",
    )

    tone_check_agent = LlmAgent(
        name="RecapToneCheck",
        model=MODEL_ID,
        instruction="""
You are a tone analyzer.
Expected in session state: `current_recap`.

Analyze the tone of the `current_recap` from the perspective of a fan of the winning team or a neutral sports journalist reporting on the game's outcome.
- A dominant win (like the 8-0 example) should generally be 'positive' or at least 'neutral-positive'.
- A close, hard-fought win might be 'positive' or 'exciting'.
- A straightforward loss would likely be 'neutral' or 'negative' for the losing team's perspective, but the recap itself should aim for objective reporting where appropriate.

Consider if the language used effectively conveys the significance of the win/loss and the performances.

Output ONLY one word that best describes the overall tone: 'positive', 'negative', or 'neutral'.
""",
        output_key="tone_check_result",
    )


# --- Sub-Agents for VisualAssetWorkflowAgent ---

    entity_extractor_agent = LlmAgent(
        name="EntityExtractorForAssets",
        model=MODEL_ID, # Can be a fast model like gemini-2.0-flash
        instruction="""
You are a text analysis assistant.
Read the dialogue script provided in session state key 'current_recap'.
Identify all unique full player names (e.g., "Willy Adames", "Shohei Ohtani") and unique full MLB team names (e.g., "Los Angeles Angels", "Milwaukee Brewers") mentioned in the script.
The team names should be the full official names if present (e.g. "Los Angeles Dodgers", not just "Dodgers" if the full name appears). Prioritize full names.

Output a JSON object with two keys:
- "players": A list of unique player full name strings.
- "teams": A list of unique MLB team full name strings.

Example Input Script (from 'current_recap'):
"Host A: The Los Angeles Angels fought hard, but the New York Yankees were too strong today.
 Host B: Definitely, and Shohei Ohtani had a great game for the Angels, even in a loss. Aaron Judge, on the other hand, was unstoppable for the Yankees."

Example Output (as a JSON string):
{
  "players": ["Shohei Ohtani", "Aaron Judge"],
  "teams": ["Los Angeles Angels", "New York Yankees"]
}

If no players or teams are found, output empty lists within the JSON object (e.g., {"players": [], "teams": []}).
Ensure your entire output is a single, valid JSON string.
        """,
        output_key="extracted_entities_json" # This will be a JSON string
    )

    # Now, update StaticAssetQueryGenerator to use this output
    static_asset_query_generator_agent = LlmAgent(
        name="StaticAssetQueryGenerator",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an asset planner.
Expected in session state: 'extracted_entities_json', which is a JSON string like '{"players": ["Player A Full Name"], "teams": ["Full Team Name X"]}'.

Your task:
1. Parse the JSON string from 'extracted_entities_json' to get lists of player names and team names.
2. For each full team name in the "teams" list, generate a query string: "[Full Team Name] logo".
3. For each player full name in the "players" list, generate a query string: "[Player Full Name] headshot".
4. Combine all these generated query strings into a single list.
5. Output this final list of query strings as a single, valid JSON string.

Example Input ('extracted_entities_json'):
'{"players": ["Shohei Ohtani", "Aaron Judge"], "teams": ["Los Angeles Angels", "New York Yankees"]}'

Example Output (as a JSON string):
"[\\"Los Angeles Angels logo\\", \\"New York Yankees logo\\", \\"Shohei Ohtani headshot\\", \\"Aaron Judge headshot\\"]"

If the input 'extracted_entities_json' is empty or represents no entities (e.g., '{"players": [], "teams": []}'),
then output an empty JSON list string: "[]".
        """,
        # No tools needed for this agent, it just transforms data from session state
        output_key="static_asset_search_queries_json", # Still outputs a JSON string
    )

    # ... (rest of your LlmAgent definitions for visual_generator_mcp_caller_agent, etc.)

    static_asset_retriever_agent = LlmAgent(
    name="StaticAssetRetriever",
    model=GEMINI_PRO_MODEL_ID, # Needs to be good at instruction following and JSON manipulation
    instruction="""
You are an asset retrieval coordinator.
Expected in session state:
- 'static_asset_search_queries_json': A JSON string list of queries (e.g., '["Los Angeles Angels logo", "Willy Adames headshot"]').
- 'player_lookup_dict_json': A JSON string of a dictionary mapping player IDs to full names (e.g., '{"12345": "Willy Adames", ...}').

Your task is to process each query from 'static_asset_search_queries_json':
1.  Parse 'player_lookup_dict_json' into a Python dictionary (player_id: player_name). Create an inverse mapping (player_name_lower_case: player_id) for efficient lookup.
2.  Initialize an empty list called `found_assets_list`.
3.  For each `original_query_string` in the parsed list from 'static_asset_search_queries_json':
    a.  If the `original_query_string` contains "logo" (e.g., "Los Angeles Angels logo"):
        i.  Carefully extract the full team name (e.g., "Los Angeles Angels"). This might involve removing " logo" and trimming whitespace.
        ii. Call the `image_embedding_mcp.search_similar_images_by_text` tool with:
            - `query_text` = the extracted full team name.
            - `top_k` = 1.
            - `filter_image_type` = "logo".
        iii.The tool returns a JSON string representing a list of results. Parse this JSON string. If results are present, take the first result dictionary.
        iv. If a valid logo asset dictionary is retrieved, add `{"search_term_origin": original_query_string, **logo_asset_dict}` to `found_assets_list`. (Ensure the final dict has image_uri, type, entity_name, etc.)
    b.  If the `original_query_string` contains "headshot" (e.g., "Willy Adames headshot"):
        i.  Extract the player's full name (e.g., "Willy Adames").
        ii. Convert the extracted name to lowercase. Look up this lowercase name in your inverted player lookup map to get the `player_id`.
        iii.If a `player_id` is found:
            1. Retrieve the original casing player name using the `player_id` from the initial `player_lookup_dict`.
            2. Call the `static_retriever_mcp.get_headshot_uri_if_exists` tool with:
                - `player_id_str` = the found `player_id` (as a string).
                - `player_name_for_log` = the original casing player name.
        iv. The tool returns a JSON string. Parse it. If `image_uri` is present in the parsed dictionary, add `{"search_term_origin": original_query_string, **headshot_asset_dict}` to `found_assets_list`.
    c.  If a query is unrecognized, skip it.
4.  After processing all queries, convert `found_assets_list` into a JSON string. This is your final output.
    If 'static_asset_search_queries_json' was initially empty or no assets were successfully found and added to `found_assets_list`, output an empty JSON list string: "[]".
    """,
    tools=[
        *loaded_mcp_tools_global.get("static_retriever_mcp", []),
        *loaded_mcp_tools_global.get("image_embedding_mcp", [])
    ],
    output_key="retrieved_static_assets_json",
)

    generated_visual_prompts_agent = LlmAgent(
    name="GeneratedVisualPrompts",
    model=GEMINI_PRO_MODEL_ID,
    instruction="""
You are an assistant director analyzing an MLB game dialogue script (in session state 'current_recap') to plan visual shots for an image generation model like Imagen 3 (which filters specific names).
Identify 3-5 key moments, scenes, or actions described in the dialogue that need a generated visual.

**Critical Imagen Compatibility Rules:**
1.  **NO Player Names:** Use generic descriptions ("an MLB player", "the batter", "the pitcher", "a fielder", "the runner").
2.  **NO Team Names.**
3.  **Uniforms:** Describe generically based on home/away context if implied by the dialogue ("a player in a white home uniform", "the batter in a home jersey", "a player in a colored away uniform", "the pitcher in a gray away jersey"), or neutrally ("an MLB player's uniform"). If specific colors are mentioned in the dialogue for a generic player (e.g. "the batter in blue and orange"), use those.

**Prompt Generation Guidelines:**
*   For actions (home run, double play, strikeout), generate 1-2 distinct visual prompts representing the sequence if applicable.
*   For descriptive moments (e.g., stadium shot, manager looking tense), generate a single detailed prompt.
*   Focus on creating descriptive prompts suitable for Imagen 3. Emphasize action, emotion, setting, and relevant details like uniform descriptions based on the rules above.

Output ONLY a JSON list of 3-5 prompt strings, formatted AS A JSON STRING.
Example JSON Output (as a string): "[\\"Prompt 1...\\", \\"Prompt 2...\\"]"
If no clear visual moments, output an empty JSON list string: "[]".
If the dialogue is too short or no clear visual moments are identifiable, output an empty JSON list string: "[]".
    """,
    output_key="visual_generation_prompts_json", # Expects a JSON STRING
)

    visual_generator_mcp_caller_agent = LlmAgent( # Renamed for clarity
    name="VisualGeneratorMCPCaller",
    model=GEMINI_PRO_MODEL_ID,
    instruction="""
You are an image generation coordination robot.
Expected in session state:

- 'game_pk': The current game_pk (as a string or number).
- 'visual_generation_prompts_json_for_tool': A string that IS a valid JSON list of image generation prompts (e.g., "[\\"prompt1 text\\", \\"prompt2 text\\"]").
Your ONLY function is to execute the `visual_assets.generate_images_from_prompts` tool and return its raw JSON string output.
You will receive `prompts_json_string` (this is already a JSON formatted string list of prompts) from session state key 'visual_generation_prompts_json_for_tool'.
You will receive `game_pk_string` (this is already a string) from session state key 'game_pk_str_for_tool'.

Immediately call the `visual_assets.generate_images_from_prompts` tool.
Use the exact `prompts_json_string` you received for the tool's `prompts_json` parameter.
Use the exact `game_pk_string` you received for the tool's `game_pk_str` parameter.
Your entire response MUST be ONLY the direct, verbatim JSON string output from the `visual_assets_mcp.generate_images_from_prompts` tool.
Do not add any other text, explanation, or formatting.
    """,
    tools=[tool for tool in loaded_mcp_tools_global.get("visual_assets", [])],
    output_key="generated_visual_assets_uris_json", # Expects JSON string
)

    visual_critic_agent = LlmAgent(
    name="VisualCritic",
    model=MODEL_ID, # Or GEMINI_PRO_MODEL_ID
    instruction="""
You are a demanding visual producer reviewing generated images for an MLB highlight.
The primary image generator (Imagen 3) cannot use specific player/team names.

Expected in session state:
- 'current_recap': The dialogue script for context.
- 'assets_for_critique_json': JSON string list of dicts: `[{"prompt_origin": "...", "image_uri": "gs://..." or null}]`.
- 'prompts_used_for_critique_json': JSON string list of all prompts that WERE ATTEMPTED for generating the current set of images.

Critique the set of generated images based on 'assets_for_critique_json' and their corresponding 'prompts_used_for_critique_json':
1.  **Relevance to Dialogue & Prompts:** For each prompt in 'prompts_used_for_critique_json', check if its corresponding image in 'assets_for_critique_json' (if generation was successful, i.e., image_uri is not null) covers the key actions/scenes from 'current_recap' that the prompt targeted. Are there action gaps for successfully generated images?
2.  **Failures:** Note any prompts for which image generation failed (image_uri is null).
3.  **Quality/Action/Composition (for successful generations):** Are the images clear? Do they effectively convey the intended action, mood, or composition described in their 'prompt_origin', even if generic?
4.  **Suggestions for Improvement (Action/Scene Focused & Generator-Safe):** If improvements are needed (either due to failed generations or poor quality/relevance of successful ones), suggest **specific new prompts** focusing on missing *actions* or improving *composition/mood*. **Ensure suggested prompts follow the generator limitations: NO specific player names, NO specific team names.** Use descriptive generic terms.

If ALL attempted prompts resulted in successful, high-quality, relevant images OR if no prompts were provided initially, respond ONLY with "Visuals look sufficient."
Otherwise, provide concise, bullet-point feedback and **specific, generator-safe prompt suggestions** for the *next* round of image generation.
    """,
    # NO TOOLS for the critic. It only generates text.
    output_key="visual_critique_text",
)

    new_visual_prompts_from_critique_agent = LlmAgent(
    name="NewVisualPromptsFromCritique",
    model=GEMINI_PRO_MODEL_ID,
    instruction="""
You are an assistant director refining visual plans based on a critique.
Expected in session state: 'visual_critique_text'.

**Strict Image Generator Limitations (Enforce These):** NO Player Names, NO Team Names, Generic Uniforms.

Task:
Analyze `visual_critique_text`.
If critique is "Visuals look sufficient." or empty, output an empty JSON list string: `"[]"`.
Otherwise, identify visual concepts needing new/better images from the critique.
Generate a JSON list of **2-4 NEW, concise, specific prompt strings** for these concepts.
Prompts MUST be generator-safe and adhere to limitations.
Translate specific player/team mentions from critique into compliant generic descriptions. Focus on action, setting, emotion.

Example Critique: "Missing a shot of the double play. Home run image needs more excitement."
Example Output (as a JSON string):
"[\\"Dynamic action shot of two MLB fielders turning a double play...\\", \\"MLB batter celebrating enthusiastically after a home run...\\"]"

Output ONLY a JSON list string.
    """,
    output_key="new_visual_generation_prompts_json", # JSON string
)

    veo_prompt_generator_agent = LlmAgent(
        name="VeoPromptGenerator",
        model=GEMINI_PRO_MODEL_ID, # Needs good reasoning capability
        instruction="""
You are a creative video director planning concise and compelling video shots for an MLB highlight reel using a text-to-video model like Veo, which generates short clips (approximately 5-8 seconds). Your prompts must be safe and adhere to common content guidelines.

Expected in session state:
- 'current_recap': The final dialogue script for the game.
- 'visual_critique_text' (Optional): Critique from the *image* generation phase.
- 'all_image_assets_list' (Optional): List of all image assets (static and generated so far). This includes dicts with 'image_uri' and 'type'.
- 'all_video_assets_list' (Optional): List of video assets generated so far.

Your Task:
1.  **Review 'current_recap'**: Identify 2-3 distinct, key moments suitable for short video clips.
    *   **Prioritize single, impactful actions or brief sequences** that can be fully conveyed in 5-8 seconds (e.g., a pitch and swing, a diving catch, a runner sliding, a batter's reaction, a brief fan celebration).
    *   Avoid overly complex sequences that require more time (e.g., a full at-bat with multiple pitches, a long running play with multiple parts).
2.  **Consider Existing Visuals**:
    *   Review 'all_image_assets_list' and 'all_video_assets_list'.
    *   **Avoid Duplication**: Do not generate prompts for moments already well-covered by existing high-quality images or videos, unless a video clip would add *significant unique dynamism or a different perspective* not possible with a static image.
    *   **Complement, Don't Repeat**: If an image shows the *result* of an action (e.g., player celebrating after a hit), a video might show the *action itself* (e.g., the swing connecting).
3.  **Generate Veo-Optimized Prompts (2-3 prompts maximum):**
    *   **Conciseness for Time**: Each prompt should describe a scene or action that can realistically unfold in about 5-8 seconds.
    *   **Focus and Clarity**: Target one primary subject and action per prompt.
    *   **Dynamic but Safe Language**:
        *   Emphasize motion, camera work (e.g., "slow-motion of...", "dynamic low-angle shot of...", "close-up tracking..."), and visual details.
        *   Describe sports actions factually and visually (e.g., "batter swings and connects with the baseball," "fielder dives to catch the ball," "runner slides into the base").
        *   **AVOID**: Overly aggressive or potentially misinterpretable "action" words if they could trigger safety filters (e.g., instead of "batter crushes the ball," try "batter hits a long fly ball"). Be mindful of terms that might imply harm even in a sports context. Focus on the athleticism and skill.
    *   **Adherence to Safety Rules (Strictly Enforced by Veo):**
        *   **NO specific player names.**
        *   **NO specific team names.**
        *   Use generic descriptions: "an MLB player," "the home team batter," "a fielder in a blue uniform," "a pitcher in a gray away jersey."
    *   **Prompt Length**: Keep prompts reasonably short and to the point.

Output ONLY a JSON list of 2-3 Veo prompt strings, formatted AS A JSON STRING.
Example Output (as a string):
"[\\"Slow-motion close-up of a baseball hitting the sweet spot of a wooden bat, a spray of dirt kicking up from home plate.\\", \\"Dynamic low-angle shot of an MLB batter in a white home uniform making contact and watching the ball fly, stadium lights in the background.\\", \\"A fielder in a blue uniform makes a diving catch on a line drive in the outfield grass.\\"]"

If no highly suitable moments for distinct, short video clips are identified (considering existing visuals and time constraints), or if the recap is very short, output an empty JSON list string: "[]".
        """,
        # No tools needed for this agent, it just generates prompts
        output_key="veo_generation_prompts_json", # JSON string list of prompts for Veo
    )

    video_generator_mcp_caller_agent = LlmAgent(
        name="VideoGeneratorMCPCaller",
        model=MODEL_ID, # Simpler model for a direct tool call
        instruction="""
You are a video generation robot.
Expected in session state:
- 'veo_generation_prompts_json_for_tool': A string that IS a valid JSON list of video generation prompts (e.g., "[\\"prompt1 text\\", \\"prompt2 text\\"]").
- 'game_pk_str_for_tool': The current game_pk as a string (e.g., "777866").

Your SOLE task is to execute the `video_clip_generator_mcp.generate_video_clips_from_prompts` tool and return its exact JSON string output.
1.  Retrieve the string value from session state key 'veo_generation_prompts_json_for_tool'.
2.  Parse this JSON string into an actual Python LIST of prompt strings. Let's call this `parsed_prompt_list`.
3.  Retrieve the string value from session state key 'game_pk_str_for_tool'. Let's call this `game_pk_value`.
4.  If `parsed_prompt_list` is empty or not a list, your output MUST be the JSON string "[]" and you DO NOT call the tool.
5.  Otherwise, call the `video_clip_generator_mcp.generate_video_clips_from_prompts` tool:
    - For the tool's `prompts` parameter, pass the `parsed_prompt_list` (the Python LIST of strings).
    - For the tool's `game_pk_str` parameter, pass the `game_pk_value`.
6.  Your entire response MUST be ONLY the direct, verbatim JSON string output from the `video_clip_generator_mcp.generate_video_clips_from_prompts` tool. This output will be a JSON string representing either a list of GCS URIs for videos or an error object. Do not add any other text, explanation, or formatting.
        """,
        tools=[tool for tool in loaded_mcp_tools_global.get("video_clip_generator_mcp", [])],
        output_key="generated_video_clips_uris_json", # JSON string list of video URIs
    )

    # --- LlmAgents for the NEW AudioProcessingPipelineAgent ---
    audio_processing_mcp_tools = loaded_mcp_tools.get("audio_processing_mcp", [])

    dialogue_to_speech_agent = LlmAgent(
        name="DialogueToSpeechAgent",
        model=GEMINI_PRO_MODEL_ID, # Can be a simpler model if just calling a tool
        instruction="""
Your SOLE TASK is to execute a tool call. You MUST call the tool named `audio_processing_mcp.synthesize_multi_speaker_speech`.
Use the value of `{session.state.current_recap}` for the `script` parameter.
Use the value of `{session.state.game_pk}` for the `game_pk_str` parameter.
The result of your execution of this tool will be a JSON string.
Your output MUST be ONLY that JSON string. DO NOT output any other text or explanation.
Example of expected tool output if successful: `{"audio_uri": "gs://some-bucket/some-file.mp3"}`
Example of expected tool output if the tool reports an error: `{"error": "Details from the tool"}`
Invoke the tool now.
        """,
        tools=audio_processing_mcp_tools,
        output_key="generated_dialogue_audio_details_json", # e.g., '{"audio_uri": "gs://..."}' or '{"error": "..."}'
    )

    audio_to_timestamps_agent = LlmAgent(
        name="AudioToTimestampsAgent",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an audio transcription analyst.
Expected in session state:
- 'generated_dialogue_audio_uri': A GCS URI string (e.g., "gs://bucket/audio.mp3") for the audio to be transcribed.

Your task:
1. Retrieve the 'generated_dialogue_audio_uri' string from session state.
2. **Validate Input:** Check if 'generated_dialogue_audio_uri' starts with "gs://".
   - If it does not start with "gs://", or is missing/empty, you MUST immediately output the JSON string: `{"error": "Invalid or missing audio GCS URI provided for transcription. Expected gs:// path."}`. Do NOT proceed to call any tool.
3. **Call Tool (if input is valid):** If the URI is valid, call the `audio_processing_mcp.get_word_timestamps_from_audio` tool with:
    - `audio_gcs_uri` = the value from 'generated_dialogue_audio_uri'.
4. **Handle Tool Output:**
   - If the tool call is successful, it will return a JSON string representing a list of word timestamp objects. Your entire response MUST be ONLY this direct, verbatim JSON string from the tool.
   - If the tool call itself returns a JSON string indicating an error (e.g., `{"error": "STT tool failed"}`), your response MUST be that exact error JSON string.
   - If the tool call fails for any other reason before returning a JSON string, you MUST output: `{"error": "Failed to execute the STT tool or received unexpected non-JSON response."}`.
        """,
        tools=audio_processing_mcp_tools,
        output_key="word_timestamps_json", # e.g., '[{"word": "Hello", ...}]' or '{"error": "..."}'
    )

    # --- Instantiate NEW Phase Agents ---
    static_asset_pipeline_agent_instance = StaticAssetPipelineAgent(
        name="StaticAssetPipeline",
        entity_extractor=entity_extractor_agent,
        static_asset_query_generator=static_asset_query_generator_agent,
        static_asset_retriever=static_asset_retriever_agent
    )
    iterative_image_generation_agent_instance = IterativeImageGenerationAgent(
        name="IterativeImageGeneration",
        generated_visual_prompts_generator=generated_visual_prompts_agent,
        visual_generator_mcp_caller=visual_generator_mcp_caller_agent,
        visual_critic=visual_critic_agent,
        new_visual_prompts_creator=new_visual_prompts_from_critique_agent,
        max_visual_refinement_loops=1
    )
    video_pipeline_agent_instance = VideoPipelineAgent(
        name="VideoPipeline",
        veo_prompt_generator=veo_prompt_generator_agent,
        video_generator_mcp_caller=video_generator_mcp_caller_agent
    )

    # --- Instantiate REFACTORED VisualAssetWorkflowAgent ---
    visual_asset_workflow_orchestrator_instance = VisualAssetWorkflowAgent(
        name="VisualAssetWorkflowOrchestrator",
        static_asset_pipeline_agent=static_asset_pipeline_agent_instance,
        iterative_image_generation_agent=iterative_image_generation_agent_instance,
        video_pipeline_agent=video_pipeline_agent_instance
    )

    # --- Instantiate NEW AudioProcessingPipelineAgent ---
    audio_processing_pipeline_agent_instance = AudioProcessingPipelineAgent(
        name="AudioProcessingPipeline",
        dialogue_to_speech_agent=dialogue_to_speech_agent,
        audio_to_timestamps_agent=audio_to_timestamps_agent
    )

    # --- Instantiate GameRecapAgent with the refactored Visual Orchestrator ---
    game_recap_assistant = GameRecapAgent(
        name="game_recap_assistant",
        initial_recap_generator=initial_recap_generator_agent,
        recap_critic=recap_critic_agent,
        critique_processor=critique_processor_agent,
        recap_reviser=recap_reviser_agent,
        grammar_check=grammar_check_agent,
        tone_check=tone_check_agent,
        visual_asset_workflow_orchestrator=visual_asset_workflow_orchestrator_instance,
        audio_processing_pipeline_agent=audio_processing_pipeline_agent_instance,
    )

    root_agent = LlmAgent(
        model=MODEL_ID,
        name="ai_assistant", # This is the app_name for the runner sometimes
        instruction=ROOT_AGENT_INSTRUCTION,
        sub_agents=[cocktail_agent, booking_agent, mlb_assistant, game_recap_assistant], # Added game_recap_assistant
    )
    return root_agent

# --- Agent Execution Helpers ---
async def _run_agent_and_get_response(
    runner: Runner,
    session_id: str,
    content: types.Content,
) -> List[str]:
    """
    Runs the ADK agent asynchronously for a given session and content,
    collecting and returning textual responses from the model.

    Args:
        runner: An instance of the ADK Runner.
        session_id: The unique identifier for the current session.
        content: The user's message/content to send to the agent.

    Returns:
        A list of strings, where each string is a part of the model's response.
    """
    logging.info("Running agent for session %s", session_id)
    events_async = runner.run_async(
        session_id=session_id, user_id=session_id, new_message=content
    )

    response_parts: List[str] = []
    async for event in events_async:
        try:
            if hasattr(event, "content") and event.content.role == "model":
                if hasattr(event.content, "parts") and event.content.parts:
                    part_text = getattr(event.content.parts[0], "text", None)
                    if isinstance(part_text, str) and part_text:
                        response_parts.append(part_text)
        except AttributeError as e:
            logging.warning("Could not process event attribute during agent run: %s", e)
    logging.info("Agent run finished for session %s.", session_id)
    return response_parts


# NEW Helper specifically for the voice query path, ensuring user_id is passed
async def _run_voice_agent_and_get_response(
    runner: Runner,
    session_id: str,
    user_id: str, # Explicitly take user_id
    content: types.Content,
) -> List[str]:
    logging.info(f"VOICE AGENT HELPER: Running agent for app '{runner.app_name}', session '{session_id}', user '{user_id}'.")
    events_async = runner.run_async(
        session_id=session_id,
        user_id=user_id,     # Pass user_id explicitly
        new_message=content
        # app_name is known by the runner instance
    )
    response_parts: List[str] = []
    async for event in events_async:
        try:
            if hasattr(event, "content") and event.content.role == "model":
                if hasattr(event.content, "parts") and event.content.parts:
                    if event.content.parts[0] and hasattr(event.content.parts[0], "text"):
                        part_text = getattr(event.content.parts[0], "text", None)
                        if isinstance(part_text, str) and part_text:
                            response_parts.append(part_text)
        except AttributeError as e:
            logging.warning("Could not process event attribute during voice agent helper run: %s. Event: %s", e, event)
    logging.info(f"VOICE AGENT HELPER: Agent run finished. Response parts: {response_parts}")
    return response_parts


# main.py
# ... (imports)
_run_agent_and_get_response
# --- Global variable for monkey-patching ---
original_get_session = None # Defined globally

# ... (APP_NAME, session_service, loaded_mcp_tools_global, etc.) ...

async def process_voice_query_with_adk(session_id: str, user_query: str) -> str:
    global original_get_session # <<< --- ADD THIS LINE ---
    
    logging.info(f"ADK processing voice query for session {session_id}: '{user_query}'")
    user_id_for_adk = session_id
    session_key_tuple = (APP_NAME, user_id_for_adk, session_id)

    if not loaded_mcp_tools_global: return "Core tools not available."
    if not session_service: return "Session service not available."

    try:
        # 1. Ensure ADK session exists (create if not)
        current_session_obj = None
        retrieved_session_from_get = None

        logging.info(f"VOICE_ADK: Attempting to GET session with key components: app='{APP_NAME}', user='{user_id_for_adk}', session='{session_id}'")
        try:
            retrieved_session_from_get = session_service.get_session(
                app_name=APP_NAME, user_id=user_id_for_adk, session_id=session_id
            )
        except KeyError:
            logging.info(f"VOICE_ADK: get_session raised KeyError for {session_key_tuple}. Session does not exist yet.")
            # retrieved_session_from_get remains None

        if retrieved_session_from_get is not None:
            current_session_obj = retrieved_session_from_get
            logging.info(f"VOICE_ADK: Session GET successful. Session ID: {current_session_obj.id}, State: {current_session_obj.state}")
        else:
            logging.info(f"VOICE_ADK: Session not found/None via GET. Attempting to CREATE session with key: {session_key_tuple}")
            current_session_obj = session_service.create_session(
                app_name=APP_NAME, user_id=user_id_for_adk, session_id=session_id, state={"source": "voice_adk_create_after_get_fail"}
            )
            logging.info(f"VOICE_ADK: Session CREATED successfully. Session ID: {current_session_obj.id}, State: {current_session_obj.state}")
        
        if hasattr(session_service, '_sessions') and isinstance(session_service._sessions, dict):
            logging.info(f"VOICE_ADK: Keys in session_service._sessions before Runner: {list(session_service._sessions.keys())}")
            if session_key_tuple in session_service._sessions:
                actual_obj_in_dict = session_service._sessions[session_key_tuple]
                obj_id = actual_obj_in_dict.id if actual_obj_in_dict else 'None Object in Dict'
                logging.info(f"VOICE_ADK: CONFIRMED key {session_key_tuple} IS in _sessions. Object type: {type(actual_obj_in_dict)}, ID: {obj_id}")
            else:
                logging.warning(f"VOICE_ADK: WARNING - key {session_key_tuple} IS NOT in _sessions dict before Runner.")
        
        content_for_adk = types.Content(role="user", parts=[types.Part(text=user_query)])
        root_adk_agent = await create_agent_with_preloaded_tools(loaded_mcp_tools_global)

        logging.info(f"VOICE_ADK: Initializing Runner with app_name='{APP_NAME}', agent_type={type(root_adk_agent)}, artifact_service_type={type(artifacts_service)}, session_service_type={type(session_service)}")
        
        # --- RUNNER INITIALIZATION ---
        adk_runner = Runner(
            app_name=APP_NAME,
            agent=root_adk_agent,
            artifact_service=artifacts_service,
            session_service=session_service
            # memory_service is optional in your Runner's signature, so omitting it is fine
        )
        logging.info(f"VOICE_ADK: Runner initialized successfully. Instance ID: {id(adk_runner)}")
        # --- END OF CORRECTION ---
        
        logging.info(f"VOICE_ADK: Calling _run_voice_agent_and_get_response with session_id='{session_id}', user_id='{user_id_for_adk}'")
        response_parts = await _run_voice_agent_and_get_response(
            adk_runner, session_id, user_id_for_adk, content_for_adk
        )
        # The restoration of original_get_session will happen in the finally block

        if response_parts:
            final_response = " ".join(response_parts)
            return final_response
        else:
            return "ADK processed (voice) but no text parts."

    # ... (except ValueError, except Exception as before) ...
    except ValueError as ve:
        # ...
        logging.error(f"VOICE_ADK: ValueError: {ve}", exc_info=True)
        return "I'm having trouble with conversation data (Runner)." if "Session not found" in str(ve) else "Data problem (voice)."
    except Exception as e:
        logging.error(f"VOICE_ADK: General error: {e}", exc_info=True)
        return "Unexpected problem (voice)."
    finally:
        # Ensure restoration of the original get_session method
        # Check if it was patched in THIS call by checking if original_get_session (the global) has a value
        if original_get_session is not None and hasattr(session_service, 'get_session') and session_service.get_session.__name__ == 'patched_get_session':
            session_service.get_session = original_get_session
            logging.info("VOICE_ADK: Restored original session_service.get_session in finally block.")
            original_get_session = None # Reset global for the next potential call to process_voice_query_with_adk

async def _get_runner_async(
    loaded_mcp_tools: Dict[str, Any], session_id: str, query: str
) -> List[str]:
    """
    Sets up and runs the root agent for a given query using preloaded tools.

    This function creates a root agent, initializes a runner, and then
    executes the agent with the user's query, returning the response parts.

    Args:
        loaded_mcp_tools: A dictionary of pre-loaded MCP tools.
        session_id: The unique identifier for the user's session.
        query: The user's input query as a string.

    Returns:
        A list of strings representing the parts of the agent's textual response.
        Returns an error message list if critical services are unavailable.
    """
    content = types.Content(role="user", parts=[types.Part(text=query)])

    if artifacts_service is None or session_service is None:
        logging.error(
            "Artifact or Session service is not initialized for _get_runner_async."
        )
        return ["Error: Core services not available."]

    if not loaded_mcp_tools:
        logging.error("MCP tools are not available for _get_runner_async.")
        return ["Error: Essential tools not loaded, cannot process request."]

    root_agent = await create_agent_with_preloaded_tools(loaded_mcp_tools)
    logging.info(f"TEXT_PATH: About to create Runner. Using session_service instance ID: {id(session_service)}")
    logging.info(f"TEXT_PATH: Session service type: {type(session_service)}")
    if hasattr(session_service, '_sessions'):
        logging.info(f"TEXT_PATH: Keys in session_service._sessions before Runner: {list(session_service._sessions.keys())}")
    else:
        logging.info("TEXT_PATH: session_service does not have _sessions attribute.")
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )
    logging.info(f"TEXT_PATH: Runner created. Runner's session_service instance ID: {id(runner.session_service)}")
    response = await _run_agent_and_get_response(runner, session_id, content)
    return response


# --- FastAPI Application ---


@asynccontextmanager
async def app_lifespan(app_instance: FastAPI) -> Any: 
    """
    Manages application startup and shutdown operations for the FastAPI app.
    Args:
        app_instance: The FastAPI application instance.
    """
    global loaded_mcp_tools_global # To store tools for the voice agent call
    logging.info("Application Lifespan: Startup initiated.")
    app_instance.state.mcp_tools = {}
    app_instance.state.mcp_tool_exit_stack = None

    try:
        collected_tools, tool_stack = await _collect_tools_stack(
            server_configs_instance
        )
        app_instance.state.mcp_tools = collected_tools
        loaded_mcp_tools_global = collected_tools     # For voice agent function
        app_instance.state.mcp_tool_exit_stack = tool_stack
        logging.info(
            "Application Lifespan: MCP Toolset initialized. Tools: %s",
            list(app_instance.state.mcp_tools.keys()),
        )
    except FileNotFoundError as file_error:
        logging.error("Command or script not found for key: %s", file_error)
    except ConnectionRefusedError as conn_refused:
        logging.error("Connection refused for key: %s", conn_refused)
    except Exception as e: # Catch a broader range of exceptions during startup
        logging.error(f"Critical error during MCP Toolset initialization: {e}", exc_info=True)
    yield

    logging.info("Application Lifespan: Shutdown initiated.")
    if app_instance.state.mcp_tool_exit_stack:
        logging.info("Application Lifespan: Closing MCP Toolset connections.")
        try:
            await app_instance.state.mcp_tool_exit_stack.aclose()
            logging.info(
                "Application Lifespan: MCP Toolset connections closed successfully."
            )
        except ConnectionRefusedError as conn_refused:
            logging.error("Connection refused for key: %s", conn_refused)
    else:
        logging.warning(
            "Application Lifespan: No MCP Toolset exit stack found to close."
        )


# Instantiate FastAPI with the lifespan manager
app = FastAPI(lifespan=app_lifespan)

# Include the Voice Agent Router
app.include_router(voice_agent_router) # The paths from voice_agent.py will be registered

# --- WebSocket Communication ---
async def run_adk_agent_async(
    websocket: WebSocket, loaded_mcp_tools: Dict[str, Any], session_id: str
) -> None:
    """
    Handles the continuous WebSocket communication loop for a connected client.

    Receives text messages from the client, processes them using the ADK agent
    (via `_get_runner_async`), and sends the agent's responses back to the client.

    Args:
        websocket: The WebSocket connection object for the client.
        loaded_mcp_tools: Pre-loaded MCP tools for the agent.
        session_id: The unique identifier for the client's session.
    """
    try:
        while True:
            text = await websocket.receive_text()
            response_parts = await _get_runner_async(loaded_mcp_tools, session_id, text)

            if not response_parts:
                logging.info(
                    "Agent for session %s did not produce a direct text response for input: '%s'",
                    session_id,
                    text[:50],
                )
                # Consider if a specific message should be sent or just wait for next input.
                # For now, we assume if response_parts is empty, no direct message to user.
                continue

            ai_message = "\n".join(response_parts)
            await websocket.send_text(json.dumps({"message": ai_message}))
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logging.info("Client %s disconnected from run_adk_agent_async.", session_id)
    finally:
        logging.info("Agent WebSocket task ending for session %s.", session_id)

 
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    FastAPI WebSocket endpoint for client connections.

    Accepts new WebSocket connections, creates a user session, and starts
    the `run_adk_agent_async` task to handle communication with the ADK agent.
    Ensures that MCP tools are loaded before starting the agent task.

    Args:
        websocket: The WebSocket connection object.
        session_id: The unique session identifier passed in the URL path.
    """
    await websocket.accept()
    logging.info("Client %s connected to WebSocket endpoint.", session_id)
    try:
        session_service.create_session(
            app_name=APP_NAME, user_id=session_id, session_id=session_id, state={}
        )

        # Access tools from app.state (set by the lifespan manager)
        loaded_mcp_tools = websocket.app.state.mcp_tools
        mcp_stack_exists = websocket.app.state.mcp_tool_exit_stack is not None

        if not loaded_mcp_tools or not mcp_stack_exists:
            logging.error(
                "MCP Tools not properly initialized. Cannot serve requests for session %s.",
                session_id,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "message": "Error: Server is not fully initialized. Please try again later."
                    }
                )
            )
            await websocket.close(code=1011)
            return

        await run_adk_agent_async(websocket, loaded_mcp_tools, session_id)

    except WebSocketDisconnect:
        logging.info(
            "Client %s disconnected from websocket_endpoint (early).", session_id
        )
    finally:
        logging.info("WebSocket endpoint cleanup for session %s.", session_id)


# --- TEMPORARY TEST ROUTE (Remove or comment out for production) ---
class TestADKQuery(BaseModel):
    session_id: str
    query: str

@app.post("/test_adk_voice_logic")
async def test_adk_processing(payload: TestADKQuery):
    """
    Temporary HTTP endpoint to directly test process_voice_query_with_adk.
    """
    logging.info(f"Received test request for /test_adk_voice_logic with payload: {payload}")
    if not loaded_mcp_tools_global: # Check if lifespan has run and tools are loaded
         raise HTTPException(status_code=503, detail="ADK Tools not loaded yet. Wait for app startup.")
    try:
        response_text = await process_voice_query_with_adk(
            session_id=payload.session_id,
            user_query=payload.query
        )
        return {"session_id": payload.session_id, "query": payload.query, "adk_response": response_text}
    except Exception as e:
        logging.error(f"Error in /test_adk_voice_logic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# --- END OF TEMPORARY TEST ROUTE ---


# Mount static files (e.g., for a web UI)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")