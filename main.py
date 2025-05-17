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
GEMINI_PRO_MODEL_ID = "gemini-2.5-flash-preview-04-17" # For potentially more complex tasks like generation/revision

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

server_configs_instance = AllServerConfigs(
    configs={
        "weather": weather_server_params,
        "bnb": bnb_server_params,
        "ct": ct_server_params,
        "mlb": mlb_stats_server_params,
        "web_search": web_search_server_params,         # NEW
        "bq_search": bq_vector_search_server_params, # NEW
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

# --- GameRecapAgent Definition (Patterned after StoryFlowAgent) ---

class GameRecapAgent(BaseAgent):
    """
    Custom agent for generating and refining an MLB game recap.
    Orchestrates fetching data, generation, critique, enrichment, and revision.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Define sub-agents that will be passed during initialization
    initial_recap_generator: LlmAgent
    recap_critic: LlmAgent
    critique_processor: LlmAgent # Processes critique, stores it, generates search queries, runs searches
    recap_reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    refinement_loop: LoopAgent
    post_processing_sequence: SequentialAgent

    def __init__(
        self,
        name: str,

        initial_recap_generator: LlmAgent,
        recap_critic: LlmAgent,
        critique_processor: LlmAgent,
        recap_reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
    ):
        # Create internal composite agents
        # The loop will run: Critic -> CritiqueProcessor -> Reviser
        refinement_loop = LoopAgent(
            name="RecapRefinementLoop",
            sub_agents=[recap_critic, critique_processor, recap_reviser],
            max_iterations=2 # Configurable number of refinement cycles
        )
        post_processing_sequence = SequentialAgent(
            name="RecapPostProcessing",
            sub_agents=[grammar_check, tone_check]
        )

        # Define the sub_agents list for the framework
        # These are the agents that GameRecapAgent directly invokes in its _run_async_impl
        sub_agents_list = [
            initial_recap_generator,
            refinement_loop,
            post_processing_sequence,
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
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting game recap workflow.")

        # Correctly get user_query:
        # 1. Try to get it from session state (if set by a preceding agent or handler)
        # 2. If not in state, get it from ctx.user_content (the initial message for this invocation)
        # Get user_query PRIMARILY from session state, as it should have been parsed and placed there by the entry handlers
        user_query = ctx.session.state.get("user_query")

        if not user_query:
            # Fallback only if not set by entry handlers (e.g. direct test of GameRecapAgent)
            logger.warning(f"[{self.name}] 'user_query' not found in session state. Attempting to use ctx.user_content.")
            if ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text:
                raw_text_from_content = ctx.user_content.parts[0].text
                # Check if raw_text_from_content is a JSON string like {"message":"..."}
                try:
                    data = json.loads(raw_text_from_content)
                    if isinstance(data, dict) and "message" in data:
                        user_query = data["message"]
                        logger.info(f"[{self.name}] Parsed user_query from ctx.user_content (JSON): '{user_query}'")
                    else:
                        user_query = raw_text_from_content # Assume it's plain text
                        logger.info(f"[{self.name}] Used user_query directly from ctx.user_content (plain text): '{user_query}'")
                except json.JSONDecodeError:
                    user_query = raw_text_from_content # Not JSON, use as is
                    logger.info(f"[{self.name}] Used user_query directly from ctx.user_content (not JSON): '{user_query}'")
                
                # Store it back into session state if derived here
                if user_query:
                    ctx.session.state["user_query"] = user_query
            else:
                user_query = "generic recap request" # Fallback
                logger.warning(f"[{self.name}] user_query also not in ctx.user_content. Using fallback: '{user_query}'")
        else:
            logger.info(f"[{self.name}] Found user_query in session state: '{user_query}'")
        
        # Ensure user_query is definitely in state for sub-agents if it was just derived
        if "user_query" not in ctx.session.state and user_query != "generic recap request":
             ctx.session.state["user_query"] = user_query
        if "game_pk" not in ctx.session.state:
            logger.warning(f"[{self.name}] game_pk not found in session state. InitialRecapGenerator needs to handle this or an error may occur. Current user_query for context: '{user_query}'")
            # If user_query was just fetched from ctx.user_content and game_pk extraction is desired here:
            # (This is a simplified example; robust extraction is complex)
            if user_query: # Ensure user_query is available
                text_query_lower = user_query.lower()
                if "game pk" in text_query_lower:
                    try:
                        pk_str = text_query_lower.split("game pk")[-1].strip().split(" ")[0]
                        ctx.session.state["game_pk"] = pk_str # Save to session state
                        logger.info(f"[{self.name}] Extracted game_pk: {pk_str} from user_query and saved to session state.")
                    except Exception as e:
                        logger.info(f"[{self.name}] Could not parse game_pk from '{text_query_lower}': {e}")
                elif "brewers last game" in text_query_lower: # Example for your query
                    # This is where you'd need a tool call to figure out the "brewers last game" game_pk
                    # For now, logging it or setting a placeholder if this agent should handle it:
                    logger.info(f"[{self.name}] User asked for 'brewers last game', game_pk needs resolution.")
                    # ctx.session.state["game_pk"] = "NEEDS_RESOLUTION_FOR_BREWERS_LAST_GAME"


        logger.info(f"[{self.name}] Running InitialRecapGenerator with game_pk: {ctx.session.state.get('game_pk')}, user_query: {ctx.session.state.get('user_query')}")
        async for event in self.initial_recap_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from InitialRecapGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        
        # 2. Refinement Loop (Critic -> CritiqueProcessor -> Reviser)
        logger.info(f"[{self.name}] Running RecapRefinementLoop...")
        async for event in self.refinement_loop.run_async(ctx):
            logger.info(f"[{self.name}] Event from RecapRefinementLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        
        logger.info(f"[{self.name}] Recap state after loop: {ctx.session.state.get('current_recap')[:100]}...")

        # 3. Sequential Post-Processing (Grammar and Tone Check)
        logger.info(f"[{self.name}] Running RecapPostProcessing...")
        async for event in self.post_processing_sequence.run_async(ctx):
            logger.info(f"[{self.name}] Event from RecapPostProcessing: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # 4. Final Output
        final_recap = ctx.session.state.get("current_recap", "No recap was finalized.")
        grammar_sugg = ctx.session.state.get("grammar_suggestions", "")
        tone_result = ctx.session.state.get("tone_check_result", "")

        logger.info(f"[{self.name}] Workflow finished. Final Recap: {final_recap[:100]}..., Grammar: {grammar_sugg}, Tone: {tone_result}")
        
        # The last agent in the sequence or loop should ideally yield the final response.
        # If not, we can construct and yield one here.
        # However, ADK typically expects the final agent in a chain to produce the final output event.
        # We assume current_recap is the main output. The LlmAgent 'reviser' or 'tone_check'
        # if it's the last one modifying content, should output it correctly.
        # If the final response is not automatically yielded by the last sub-agent, uncomment below:
        # yield self.create_final_response_event(ctx, final_recap)


# --- Tool Collection ---
async def _collect_tools_stack(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    stack_needs_closing = False
    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(
            server_config_dict.configs, dict
        ):
            logging.error("server_config_dict does not have a valid '.configs' dictionary.")
            return {}, exit_stack
        for key, server_params in server_config_dict.configs.items():
            individual_exit_stack: Optional[contextlib.AsyncExitStack] = None
            try:
                tools, individual_exit_stack = await MCPToolset.from_server(
                    connection_params=server_params
                )
                if individual_exit_stack:
                    await exit_stack.enter_async_context(individual_exit_stack)
                    stack_needs_closing = True
                if tools:
                    all_tools[key] = tools
                    logger.info(f"Successfully collected tools for MCP server: {key}")
                else:
                    logging.warning("Connection successful for key '%s', but no tools returned.", key)
            except FileNotFoundError as file_error:
                logging.error("Command or script not found for key '%s': %s", key, file_error)
            except ConnectionRefusedError as conn_refused:
                logging.error("Connection refused for key '%s': %s", key, conn_refused)
            except Exception as e:
                logging.error(f"Failed to connect or get tools for {key}: {e}", exc_info=True)


        if not all_tools:
            logging.warning("No tools were collected from any server.")

        # Ensure all expected keys exist, even if empty
        expected_keys = ["weather", "bnb", "ct", "mlb", "web_search", "bq_search"]
        for k in expected_keys:
            if k not in all_tools:
                logging.info("Tools for key '%s' were not collected. Ensuring key exists with empty list.", k)
                all_tools[k] = []
        return all_tools, exit_stack
    except Exception as e:
        logging.error("Unhandled exception in _collect_tools_stack: %s", e, exc_info=True)
        if stack_needs_closing:
            await exit_stack.aclose()
        raise

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
You are an expert sports journalist tasked with generating an initial MLB game recap. Your goal is to create a compelling narrative of the game, not just a list of events.

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

3.  **Synthesize Initial Narrative Recap (Only if Game is "Final"):**
    *   **Storytelling First:** Your primary goal is to tell the story of the game.
        *   Identify a potential "story of the game" (e.g., a pitcher's duel, an offensive breakout, a key player's heroics, a specific turning point).
        *   Start with an engaging lead paragraph that summarizes the game's outcome and main storyline.
    *   **Pitching Narrative:** Describe the performance of the starting pitchers, especially the winner and loser. Go beyond stats – how did they look? Were they dominant, struggling, etc.?
    *   **Offensive Highlights & Progression:**
        *   Describe how the scoring unfolded inning by inning, focusing on the most impactful plays (key hits, home runs, RBI moments).
        *   Name the players involved in these key offensive moments.
        *   Weave in details from `pbp_summary_data` for crucial moments to add color.
    *   **Integrate Context:**
        *   If `pre_game_context_notes` are available, subtly weave them into the narrative where relevant (e.g., "The rivalry continued as...", "Coming into the series, Player X was on a hot streak and proved it by...").
    *   **Guidance from Past Critiques:** Use `past_critiques_feedback` (if available) for general guidance on narrative structure, tone, and journalistic style.
    *   **Language:** Use vivid, active language. Avoid just listing stats.
    *   **Acknowledge Limitations:** If specific details are unavailable from the provided data (e.g., a very niche stat), do not invent them.

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
Expected in session state: `current_recap`, `game_pk`, `user_query`.

Review the `current_recap`. Provide constructive, actionable criticism. Focus on:

- **Accuracy & Completeness:**
    - Are scores, key player actions (pitchers, hitters), and game sequence correct and sufficiently detailed?
    - Are there any factual errors or significant omissions based on typical game data expectations?
- **Narrative & Engagement:**
    - Does the recap tell a compelling story of the game, or is it just a dry list of events?
    - Does it have a clear narrative arc (beginning, rising action, climax/key moments, resolution)?
    - Does it capture the tension, excitement, or any specific "story" of the game (e.g., a pitcher's duel, an offensive explosion, a key turning point)?
    - Is the language engaging, vivid, and journalistic? Or is it flat and robotic?
- **Journalistic Style:**
    - Does it sound like a professional sports recap a fan would read on a sports website?
    - Are there clichés or awkward phrasings that should be improved?
    - Is there a good balance between stats and narrative description?
- **Information Gaps & Opportunities for Enrichment:**
    - Identify specific missing information that would enhance the story or context (e.g., "How did Player X get on base before scoring?", "What was the specific hit that drove in the go-ahead run in the 7th?", "Was there a critical defensive play not mentioned?", "What was the impact of this win/loss on standings or morale, if discernible?").
    - Are there opportunities to better integrate existing stats or add highly relevant ones to support the narrative?
- **Clarity & Flow:**
    - Is the language clear, concise, and does the recap flow logically from one point to the next?
    - Are there run-on sentences or overly complex paragraphs?
- **Data Usage:**
    - Are stats (if any) used effectively to support the narrative, or do they feel tacked on?

If the draft is excellent and requires no changes (rare!), respond ONLY with "The recap is excellent."
Otherwise, provide **specific, bulleted feedback** with clear examples of what needs improvement or what specific information is missing that could be researched.
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
You are an expert sports story editor and reviser, tasked with transforming a game recap into a polished, engaging piece of sports journalism.

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
2.  **Address Critique Holistically:** Thoroughly revise the `current_recap` to address *every actionable point* in `current_critique`.
    *   **Integrate Research Narratively:** Seamlessly weave in relevant information from `web_search_findings` and `rag_findings` to fill information gaps, add depth, provide context, or correct inaccuracies. Don't just append facts; integrate them into the story. If these findings are sparse or don't fully address a critique point, use the core game data (`live_score_data`, `pbp_summary_data`, `comprehensive_data_json`) to find the necessary details.
    *   **Enhance Storytelling:** Elevate the language. Use stronger action verbs, more descriptive adjectives, and craft sentences that build excitement or highlight the significance of events.
    *   **Refine Narrative Arc:** Ensure the recap has a clear lead, develops the game's key moments and turning points, and concludes effectively.
    *   **Contextualize Performances:** Go beyond just stating stats. Explain their significance in the context of the game.
3.  **Apply Stylistic Guidance:**
    *   Incorporate general stylistic advice from `session.state.past_critiques_feedback` where applicable, but prioritize addressing the `current_critique` and its associated research.
    *   Ensure the tone is appropriate for the game's outcome and the user's request.
4.  **Fulfill Original Request:** Double-check that the revised recap comprehensively and engagingly addresses the `user_query`.
5.  **Clarity and Conciseness:** Ensure the final recap is clear, flows well, and avoids unnecessary jargon or overly complex sentences.

Your final output MUST BE ONLY the revised and improved game recap text. Do not include any conversational intros, outros, or explanations about the changes you made.
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

    game_recap_assistant = GameRecapAgent(
        name="game_recap_assistant",
        initial_recap_generator=initial_recap_generator_agent,
        recap_critic=recap_critic_agent,
        critique_processor=critique_processor_agent,
        recap_reviser=recap_reviser_agent,
        grammar_check=grammar_check_agent,
        tone_check=tone_check_agent,
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