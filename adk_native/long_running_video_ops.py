# adk_native/long_running_video_ops.py
import logging
import json
import os
import re
import time
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, UTC
import hashlib

from google.cloud import storage # type: ignore
from google import genai as google_genai_sdk # type: ignore # Use the specific client for Veo
from google.genai import types as genai_types # type: ignore
from google.api_core import exceptions as google_exceptions # type: ignore
from google.adk.tools.tool_context import ToolContext # type: ignore

# --- Configuration (Copied from video_clip_server.py) ---
GCP_PROJECT_ID_VIDEO = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021") # Added _VIDEO to avoid clash if main.py also has it
GCP_LOCATION_VIDEO = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_VEO_MODEL_ID_VIDEO = os.getenv("VERTEX_VEO_MODEL_ID", "veo-2.0-generate-001") # Corrected
GCS_BUCKET_GENERATED_VIDEOS = os.getenv("GCS_BUCKET_GENERATED_VIDEOS", "mlb_generated_videos")
GCS_VIDEO_OUTPUT_PREFIX = "adk_native/videos/" # Changed prefix

VIDEO_GENERATION_ASPECT_RATIO = "16:9"
VIDEO_GENERATION_PERSON_ALLOW = "allow_adult"
VIDEO_DURATION_SECONDS = int(os.getenv("VIDEO_DURATION_SECONDS_NATIVE", 7))
VIDEO_ENHANCE_PROMPT = True

VIDEO_POLLING_INTERVAL_SECONDS = int(os.getenv("VIDEO_POLLING_INTERVAL_SECONDS_NATIVE", 20))
VIDEO_GENERATION_QUOTA_SLEEP_SECONDS = int(os.getenv("VIDEO_GENERATION_QUOTA_SLEEP_SECONDS_NATIVE", 80))
VIDEO_GENERATION_ERROR_SLEEP_SECONDS = int(os.getenv("VIDEO_GENERATION_ERROR_SLEEP_SECONDS_NATIVE", 15))
MAX_PROMPTS_TO_ANIMATE_PER_CALL = int(os.getenv("MAX_PROMPTS_TO_ANIMATE_PER_CALL_NATIVE", 2)) # Limit for native ADK tool

logger = logging.getLogger(__name__)

# --- Global Task Tracking ---
VIDEO_GENERATION_TASKS: Dict[str, Dict[str, Any]] = {}

# --- Client Initialization ---
storage_client_video_instance = None
genai_client_video_instance = None

def ensure_video_clients_initialized():
    global storage_client_video_instance, genai_client_video_instance
    if not storage_client_video_instance:
        storage_client_video_instance = storage.Client(project=GCP_PROJECT_ID_VIDEO)
    if not genai_client_video_instance:
        try:
            # google-genai client for Veo (Vertex backend)
            genai_client_video_instance = google_genai_sdk.Client(vertexai=True, project=GCP_PROJECT_ID_VIDEO, location=GCP_LOCATION_VIDEO)
        except Exception as e:
            logger.error(f"Failed to initialize Veo (google-genai) client: {e}")
            genai_client_video_instance = None


async def _perform_video_generation_work(task_id: str, prompts: List[str], game_pk_str: str):
    """The actual video generation logic, runs as a background asyncio task."""
    ensure_video_clients_initialized()
    if not genai_client_video_instance:
        VIDEO_GENERATION_TASKS[task_id].update({"status": "failed", "error": "Veo client not initialized."})
        return

    VIDEO_GENERATION_TASKS[task_id]["status"] = "processing"
    generated_uris: List[str] = []
    errors: List[str] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    prompts_to_animate = prompts[:MAX_PROMPTS_TO_ANIMATE_PER_CALL]

    for i, prompt_text in enumerate(prompts_to_animate):
        logger.info(f"[Task {task_id}] Processing video prompt {i+1}/{len(prompts_to_animate)}: '{prompt_text[:80]}...'")
        operation_lro = None
        operation_name_str = "UNKNOWN_OPERATION"

        try:
            prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_')
            video_blob_name = f"{GCS_VIDEO_OUTPUT_PREFIX}game_{game_pk_str}/vid_{prompt_slug}_{timestamp}_{i:02d}.mp4"
            output_video_gcs_uri = f"gs://{GCS_BUCKET_GENERATED_VIDEOS}/{video_blob_name}"

            veo_config = genai_types.GenerateVideosConfig(
                output_gcs_uri=output_video_gcs_uri,
                aspect_ratio=VIDEO_GENERATION_ASPECT_RATIO,
                number_of_videos=1,
                duration_seconds=VIDEO_DURATION_SECONDS,
                person_generation=VIDEO_GENERATION_PERSON_ALLOW,
                enhance_prompt=VIDEO_ENHANCE_PROMPT,
            )

            logger.info(f"[Task {task_id}] Starting Veo text-to-video operation...")
            operation_lro = await asyncio.to_thread(
                genai_client_video_instance.models.generate_videos, # type: ignore
                model=VERTEX_VEO_MODEL_ID_VIDEO,
                prompt=prompt_text,
                config=veo_config,
            )
            operation_name_str = getattr(operation_lro, 'name', 'UNKNOWN_OPERATION_NAME')
            logger.info(f"[Task {task_id}] Veo operation started: {operation_name_str}. Polling...")

            polling_start_time = time.time()
            current_op_state = operation_lro

            while not current_op_state.done: # type: ignore
                await asyncio.sleep(VIDEO_POLLING_INTERVAL_SECONDS)
                try:
                    current_op_state = await asyncio.to_thread(genai_client_video_instance.operations.get, current_op_state) # type: ignore
                    elapsed = time.time() - polling_start_time
                    logger.debug(f"[Task {task_id}] Polling Veo operation {operation_name_str} (elapsed: {elapsed:.0f}s)... Done: {current_op_state.done}") # type: ignore
                except Exception as poll_err:
                    logger.error(f"[Task {task_id}] Error refreshing Veo operation {operation_name_str}: {poll_err}. Stopping polling.")
                    current_op_state = None # Mark as failed
                    break
            
            if current_op_state and current_op_state.done and not current_op_state.error: # type: ignore
                if current_op_state.response and current_op_state.result.generated_videos: # type: ignore
                    video_uri = current_op_state.result.generated_videos[0].video.uri # type: ignore
                    logger.info(f"[Task {task_id}] Veo successful for prompt '{prompt_text[:80]}...'. URI: {video_uri}")
                    generated_uris.append(video_uri)
                else:
                    err_msg = f"Veo operation {operation_name_str} finished but no response/result."
                    logger.warning(f"[Task {task_id}] {err_msg}")
                    errors.append(err_msg)
            elif current_op_state and current_op_state.error: # type: ignore
                err_msg = f"Veo operation {operation_name_str} failed. Error: {current_op_state.error}" # type: ignore
                logger.error(f"[Task {task_id}] {err_msg}")
                errors.append(err_msg)
            elif current_op_state is None:
                err_msg = f"Veo operation FAILED during polling for prompt '{prompt_text[:80]}...'."
                logger.error(f"[Task {task_id}] {err_msg}")
                errors.append(err_msg)
            else:
                err_msg = f"Veo operation {operation_name_str} finished in unexpected state."
                logger.warning(f"[Task {task_id}] {err_msg}")
                errors.append(err_msg)
            
            await asyncio.sleep(1) # Brief pause between prompts

        except google_exceptions.ResourceExhausted as quota_error:
            err_msg = f"Veo Quota Exceeded for prompt '{prompt_text[:80]}...': {quota_error}"
            logger.error(f"[Task {task_id}] {err_msg}")
            errors.append(err_msg)
            await asyncio.sleep(VIDEO_GENERATION_QUOTA_SLEEP_SECONDS)
        except Exception as e:
            err_msg = f"Unexpected error during Veo text-to-video for prompt '{prompt_text[:80]}...': {e}"
            logger.error(f"[Task {task_id}] {err_msg}", exc_info=True)
            errors.append(err_msg)
            await asyncio.sleep(VIDEO_GENERATION_ERROR_SLEEP_SECONDS)

    if not generated_uris and errors:
        VIDEO_GENERATION_TASKS[task_id].update({
            "status": "failed",
            "error": "; ".join(errors)
        })
    else:
        VIDEO_GENERATION_TASKS[task_id].update({
            "status": "completed",
            "result": json.dumps(generated_uris), # Result should be JSON serializable
            "errors": errors
        })
    logger.info(f"[Task {task_id}] Video generation work finished. Status: {VIDEO_GENERATION_TASKS[task_id]['status']}")


def initiate_video_generation(prompts: List[str], game_pk_str: str = "unknown_game", tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Initiates background video generation for a list of prompts.
    This function is intended to be wrapped by ADK's LongRunningFunctionTool.
    """
    ensure_video_clients_initialized() # Ensure this is called
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        return {"status": "error", "message": "Invalid prompts input. Must be a list of strings."}

    if not prompts:
        return {"status": "completed", "result": json.dumps([])}

    prompt_hash_part = hashlib.md5(json.dumps(sorted(prompts)).encode()).hexdigest()[:8]
    task_id = f"vid_task_{prompt_hash_part}_{int(time.time())}"
    
    agent_name = tool_context.agent_name if tool_context and hasattr(tool_context, 'agent_name') else "UnknownAgentFromToolContext"
    logger.info(f"LRFT:init_video_gen (Agent: {agent_name}): Initiating task {task_id} for {len(prompts)} prompts, game {game_pk_str}.")

    VIDEO_GENERATION_TASKS[task_id] = {
        "status": "submitted",
        "prompts": prompts,
        "game_pk_str": game_pk_str,
        "start_time": time.time(),
        # REMOVED: "tool_call_id": tool_context.id if tool_context else None
    }
    asyncio.create_task(_perform_video_generation_work(task_id, prompts, game_pk_str))

    return {
        "status": "pending_agent_client_action",
        "task_id": task_id,
        "tool_name": "initiate_video_generation", # This name is used by the client polling logic
        "message": f"Video generation task {task_id} initiated for {len(prompts)} prompts. Awaiting client polling."
    }