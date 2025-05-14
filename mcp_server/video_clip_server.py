# mcp_server/video_clip_server.py
import logging
import json
import os
import re
import time
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, UTC

from google.cloud import storage
import google.generativeai as genai # Use the specific client for Veo
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Needed for Veo client init
VERTEX_VEO_MODEL_ID = os.getenv("VERTEX_VEO_MODEL_ID", "veo-2.0-generate-001")
GCS_BUCKET_GENERATED_VIDEOS = os.getenv("GCS_BUCKET_GENERATED_VIDEOS", "mlb_generated_videos")
GCS_VIDEO_OUTPUT_PREFIX = "generated/videos/"

# Veo Generation Parameters
VIDEO_GENERATION_ASPECT_RATIO = "16:9"
VIDEO_GENERATION_PERSON_ALLOW = "allow_adult"
VIDEO_DURATION_SECONDS = int(os.getenv("VIDEO_DURATION_SECONDS", 7))
VIDEO_ENHANCE_PROMPT = True # Or getenv with bool conversion

# Sleep/Retry Config
VIDEO_GENERATION_SLEEP_SECONDS = int(os.getenv("VIDEO_GENERATION_SLEEP_SECONDS", 15))
VIDEO_POLLING_INTERVAL_SECONDS = int(os.getenv("VIDEO_POLLING_INTERVAL_SECONDS", 30))
VIDEO_GENERATION_QUOTA_SLEEP_SECONDS = int(os.getenv("VIDEO_GENERATION_QUOTA_SLEEP_SECONDS", 90))
VIDEO_GENERATION_ERROR_SLEEP_SECONDS = int(os.getenv("VIDEO_GENERATION_ERROR_SLEEP_SECONDS", 20))
MAX_PROMPTS_TO_ANIMATE_PER_CALL = int(os.getenv("MAX_PROMPTS_TO_ANIMATE_PER_CALL", 3)) # Limit per MCP call

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("video_clip_generator")

# --- Initialize Clients ---
storage_client = None
genai_client = None # Specific client for Veo

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    # Initialize google-genai client, pointing to Vertex backend
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logger.info(f"Storage client and google-genai Client for Veo initialized (Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION})")
except Exception as e:
    logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
    # Allow server to start but tools will fail

# --- MCP Tool ---

@mcp.tool()
async def generate_video_clips_from_prompts(prompts_json: str, game_pk_str: str = "unknown_game") -> str:
    """
    Generates short video clips from text prompts using Veo.
    Expects prompts_json as a JSON string list. Returns a JSON string list of GCS URIs.
    game_pk_str is used for GCS path naming.
    """
    if not genai_client:
        return json.dumps({"error": "Veo (google-genai) client not initialized."})
    if not storage_client: # Should not happen if init succeeds, but check
        return json.dumps({"error": "GCS client not initialized."})

    try:
        prompts: List[str] = json.loads(prompts_json)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Input must be a JSON list of strings.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid prompts_json input: {e}")
        return json.dumps({"error": f"Invalid input: {e}"})

    if not prompts:
        logger.info("No prompts provided for video generation.")
        return json.dumps([]) # Return empty list

    prompts_to_animate = prompts[:MAX_PROMPTS_TO_ANIMATE_PER_CALL]
    logger.info(f"VIDEO_MCP: generate_video_clips - Processing {len(prompts_to_animate)} prompts (max {MAX_PROMPTS_TO_ANIMATE_PER_CALL}) for game {game_pk_str}.")

    generated_uris = []
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    for i, prompt_text in enumerate(prompts_to_animate):
        logger.info(f"Attempting text-to-video generation {i+1}/{len(prompts_to_animate)} for prompt: '{prompt_text[:80]}...'")
        operation_lro = None
        operation_name_str = "UNKNOWN_OPERATION"

        try:
            # Create unique output URI for the video
            prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_')
            video_blob_name = f"{GCS_VIDEO_OUTPUT_PREFIX}game_{game_pk_str}/vid_{prompt_slug}_{timestamp}_{i:02d}.mp4"
            output_video_gcs_uri = f"gs://{GCS_BUCKET_GENERATED_VIDEOS}/{video_blob_name}"

            # Prepare config
            veo_config = genai_types.GenerateVideosConfig(
                output_gcs_uri=output_video_gcs_uri,
                aspect_ratio=VIDEO_GENERATION_ASPECT_RATIO,
                number_of_videos=1,
                duration_seconds=VIDEO_DURATION_SECONDS,
                person_generation=VIDEO_GENERATION_PERSON_ALLOW,
                enhance_prompt=VIDEO_ENHANCE_PROMPT,
            )

            # Start Veo operation (blocking I/O in thread)
            logger.info(f"Starting Veo text-to-video operation...")
            operation_lro = await asyncio.to_thread(
                genai_client.models.generate_videos,
                model=VERTEX_VEO_MODEL_ID,
                prompt=prompt_text,
                config=veo_config,
            )
            operation_name_str = getattr(operation_lro, 'name', 'UNKNOWN_OPERATION_NAME')
            logger.info(f"Veo operation started: {operation_name_str}. Polling for completion...")

            # Poll the operation
            polling_start_time = time.time()
            current_op_state = operation_lro

            while not current_op_state.done:
                await asyncio.sleep(VIDEO_POLLING_INTERVAL_SECONDS)
                try:
                    # Refresh state (blocking I/O in thread)
                    current_op_state = await asyncio.to_thread(genai_client.operations.get, current_op_state)
                    elapsed = time.time() - polling_start_time
                    logger.debug(f"Polling Veo operation {operation_name_str} (elapsed: {elapsed:.0f}s)... Done: {current_op_state.done}")
                except Exception as poll_err:
                    logger.error(f"Error refreshing Veo operation {operation_name_str}: {poll_err}. Stopping polling.")
                    current_op_state = None # Mark as failed
                    break

            # Process result
            if current_op_state and current_op_state.done and not current_op_state.error:
                if current_op_state.response and current_op_state.result.generated_videos:
                     video_uri = current_op_state.result.generated_videos[0].video.uri
                     logger.info(f"Veo text-to-video successful for prompt '{prompt_text[:80]}...'. URI: {video_uri}")
                     generated_uris.append(video_uri)
                else:
                     logger.warning(f"Veo operation {operation_name_str} finished but has no response/result.")
            elif current_op_state and current_op_state.error:
                logger.error(f"Veo operation {operation_name_str} finished WITH FAILED status. Error: {current_op_state.error}")
            elif current_op_state is None:
                 logger.error(f"Veo operation FAILED during polling for prompt '{prompt_text[:80]}...'.")
            else:
                 logger.warning(f"Veo operation {operation_name_str} finished in unexpected state.")

            logger.debug(f"Sleeping {VIDEO_GENERATION_SLEEP_SECONDS}s before next video generation.")
            await asyncio.sleep(VIDEO_GENERATION_SLEEP_SECONDS)

        except google_exceptions.ResourceExhausted as quota_error:
            logger.error(f"Veo Quota Exceeded starting generation for prompt '{prompt_text[:80]}...': {quota_error}. Sleeping {VIDEO_GENERATION_QUOTA_SLEEP_SECONDS}s.")
            await asyncio.sleep(VIDEO_GENERATION_QUOTA_SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"Unexpected error during Veo text-to-video setup/start for prompt '{prompt_text[:80]}...': {e}", exc_info=True)
            await asyncio.sleep(VIDEO_GENERATION_ERROR_SLEEP_SECONDS)

    logger.info(f"VIDEO_MCP: Finished video generation. Generated {len(generated_uris)} URIs.")
    return json.dumps(generated_uris)

if __name__ == "__main__":
    logger.info("Starting Video Clip Generation MCP Server (Veo)...")
    mcp.run(transport="stdio")