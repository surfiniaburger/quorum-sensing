# adk_native/long_running_image_ops.py
import logging
import json
import os
import re
import io
import time
import random
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, UTC
import hashlib

import requests # type: ignore
from PIL import Image as PilImage # type: ignore
from google.cloud import storage, secretmanager # type: ignore
from google.api_core import exceptions as google_exceptions # type: ignore
from vertexai.preview.vision_models import ImageGenerationModel # type: ignore
from google.adk.tools.tool_context import ToolContext # type: ignore


# --- Configuration (Copied from visual_asset_server.py, ensure these are accessible) ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_IMAGEN_MODEL_ID = os.getenv("VERTEX_IMAGEN_MODEL", "imagen-3.0-generate-002")
GCS_BUCKET_GENERATED_ASSETS = os.getenv("GCS_BUCKET_GENERATED_ASSETS", "mlb_generated_assets")

CLOUDFLARE_ACCOUNT_ID_SECRET = os.getenv("CLOUDFLARE_ACCOUNT_ID_SECRET", "cloudflare-account-id")
CLOUDFLARE_API_TOKEN_SECRET = os.getenv("CLOUDFLARE_API_TOKEN_SECRET", "cloudflare-api-token")
CLOUDFLARE_FALLBACK_MODEL = os.getenv("CLOUDFLARE_FALLBACK_MODEL", "@cf/bytedance/stable-diffusion-xl-lightning")
CLOUDFLARE_API_ENDPOINT_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"

IMAGE_GENERATION_SEED = None
IMAGE_GENERATION_WATERMARK = False
IMAGE_GENERATION_NEGATIVE_PROMPT = "text, words, letters, blurry, low quality, cartoonish, illustration, drawing, sketch, unrealistic, watermark, signature, writing"
IMAGE_GENERATION_ASPECT_RATIO = "16:9"
IMAGE_GENERATION_NUMBER_PER_PROMPT = 1
IMAGE_GENERATION_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_SLEEP_SECONDS_NATIVE", 5))
IMAGE_GENERATION_ERROR_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_ERROR_SLEEP_SECONDS_NATIVE", 2))
IMAGE_GENERATION_QUOTA_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_QUOTA_SLEEP_SECONDS_NATIVE", 65))
CLOUDFLARE_FALLBACK_SLEEP_SECONDS = int(os.getenv("CLOUDFLARE_FALLBACK_SLEEP_SECONDS_NATIVE", 2))

MAX_API_RETRIES = 3  # Max retries for API calls (like Imagen) per prompt
BASE_RETRY_SLEEP_SECONDS = 2  

logger = logging.getLogger(__name__)

IMAGE_GENERATION_TASKS: Dict[str, Dict[str, Any]] = {}

storage_client_instance = None
imagen_model_instance = None
secret_manager_client_instance = None
cloudflare_account_id_global = None
cloudflare_api_token_global = None

def ensure_clients_initialized():
    global storage_client_instance, imagen_model_instance, secret_manager_client_instance
    global cloudflare_account_id_global, cloudflare_api_token_global

    if not storage_client_instance:
        storage_client_instance = storage.Client(project=GCP_PROJECT_ID)
    if not imagen_model_instance:
        try:
            import vertexai # type: ignore
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            imagen_model_instance = ImageGenerationModel.from_pretrained(VERTEX_IMAGEN_MODEL_ID)
        except Exception as e:
            logger.error(f"Failed to initialize Imagen model: {e}")
            imagen_model_instance = None
    if not secret_manager_client_instance:
        secret_manager_client_instance = secretmanager.SecretManagerServiceClient()

    if not cloudflare_account_id_global and secret_manager_client_instance:
        cloudflare_account_id_global = _access_secret_version_sync(GCP_PROJECT_ID, CLOUDFLARE_ACCOUNT_ID_SECRET)
    if not cloudflare_api_token_global and secret_manager_client_instance:
        cloudflare_api_token_global = _access_secret_version_sync(GCP_PROJECT_ID, CLOUDFLARE_API_TOKEN_SECRET)

    if not imagen_model_instance and not (cloudflare_account_id_global and cloudflare_api_token_global):
        logger.warning("Neither Imagen nor Cloudflare fallback is available for image generation.")

def _access_secret_version_sync(project_id: str, secret_id: str, version_id: str = "latest") -> Optional[str]:
    if not secret_manager_client_instance: return None
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    try:
        response = secret_manager_client_instance.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        return payload
    except Exception as e:
        logger.error(f"Error accessing secret {secret_id} synchronously: {e}", exc_info=False)
        return None

async def _save_image_to_gcs_native(image_bytes: bytes, bucket_name: str, blob_name: str, content_type='image/png') -> Optional[str]:
    if not storage_client_instance:
        logger.error("GCS storage client not initialized for _save_image_to_gcs_native.")
        return None
    try:
        bucket = storage_client_instance.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(
            blob.upload_from_string,
            image_bytes,
            content_type=content_type,
            timeout=180  # Increased timeout to 3 minutes
        )
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved image to {gcs_uri}")
        return gcs_uri
    except google_exceptions.TimeoutError as e: # More specific timeout exception from google-api-core
        logger.error(f"GCS upload timeout for gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None
    except Exception as e: # Catch other potential exceptions during upload
        logger.error(f"Error saving image to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

async def _generate_image_cloudflare_native(prompt: str, width: int = 768, height: int = 768, num_steps: int = 20) -> Optional[bytes]:
    if not cloudflare_account_id_global or not cloudflare_api_token_global:
        return None
    url = CLOUDFLARE_API_ENDPOINT_TEMPLATE.format(account_id=cloudflare_account_id_global, model_id=CLOUDFLARE_FALLBACK_MODEL)
    headers = {"Authorization": f"Bearer {cloudflare_api_token_global}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "width": width, "height": height, "num_steps": num_steps}
    try:
        response = await asyncio.to_thread(
            requests.post, url, headers=headers, json=data, timeout=60
        )
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Cloudflare fallback failed. Status: {response.status_code}, Response: {response.text[:200]}...")
            return None
    except Exception as e:
        logger.error(f"Error during Cloudflare API request: {e}", exc_info=True)
        return None

async def _perform_image_generation_work(task_id: str, prompts: List[str], game_pk_str: str):
    ensure_clients_initialized()
    IMAGE_GENERATION_TASKS[task_id]["status"] = "processing"
    generated_uris: List[str] = []
    errors: List[str] = [] # Collect errors for each prompt
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    generation_params = {
        "number_of_images": IMAGE_GENERATION_NUMBER_PER_PROMPT,
        "aspect_ratio": IMAGE_GENERATION_ASPECT_RATIO,
        "add_watermark": IMAGE_GENERATION_WATERMARK,
        "negative_prompt": IMAGE_GENERATION_NEGATIVE_PROMPT,
    }

    for i, prompt_text in enumerate(prompts):
        logger.info(f"[Task {task_id}] Processing image prompt {i+1}/{len(prompts)}: '{prompt_text[:80]}...'")
        image_bytes = None
        model_used = None
        content_type = 'image/png'
        imagen_succeeded_this_prompt = False

        if imagen_model_instance:
            for attempt in range(MAX_API_RETRIES):
                try:
                    current_seed = IMAGE_GENERATION_SEED
                    if current_seed is None and not IMAGE_GENERATION_WATERMARK:
                        current_seed = random.randint(1, 2**31 - 1)

                    logger.info(f"[Task {task_id}] Prompt {i+1}, Attempt {attempt+1}/{MAX_API_RETRIES} for Imagen.")
                    response = await asyncio.to_thread(
                        imagen_model_instance.generate_images,
                        prompt=prompt_text,
                        seed=current_seed,
                        number_of_images=generation_params["number_of_images"],
                        aspect_ratio=generation_params["aspect_ratio"],
                        add_watermark=generation_params["add_watermark"],
                        negative_prompt=generation_params["negative_prompt"]
                    )

                    if response and response.images:
                        img_object = response.images[0]
                        if hasattr(img_object, '_image_bytes') and img_object._image_bytes:
                            image_bytes = img_object._image_bytes
                            content_type = getattr(img_object, 'mime_type', 'image/png')
                        else: # Fallback for PIL-only response
                            pil_img = img_object._pil_image
                            buffer = io.BytesIO()
                            pil_img.save(buffer, format="PNG")
                            image_bytes = buffer.getvalue()
                            content_type = 'image/png'
                        
                        model_used = VERTEX_IMAGEN_MODEL_ID
                        imagen_succeeded_this_prompt = True
                        logger.info(f"[Task {task_id}] Imagen generation successful for prompt {i+1} on attempt {attempt+1}.")
                        break # Success, break from retry loop for this prompt

                    else:
                        logger.warning(f"[Task {task_id}] Imagen returned no image for prompt {i+1} on attempt {attempt+1}.")
                        # This isn't necessarily a retryable API error, might be prompt issue.
                        # For now, let's treat it as a failure for this attempt; if all attempts fail, it's an error for the prompt.
                        if attempt == MAX_API_RETRIES - 1:
                             errors.append(f"Imagen returned no image for prompt: {prompt_text[:50]}...")


                except google_exceptions.ResourceExhausted as quota_error:
                    err_msg = f"Imagen Quota Exceeded for prompt {i+1} (attempt {attempt+1}): {quota_error}"
                    logger.error(f"[Task {task_id}] {err_msg}")
                    if attempt < MAX_API_RETRIES - 1:
                        # Exponential backoff with jitter
                        sleep_time = (BASE_RETRY_SLEEP_SECONDS ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"[Task {task_id}] Retrying prompt {i+1} in {sleep_time:.2f}s due to quota error.")
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"[Task {task_id}] Quota error for prompt {i+1} after {MAX_API_RETRIES} attempts. Adding to errors and sleeping longer before next prompt (if any).")
                        errors.append(f"Quota exceeded for prompt: {prompt_text[:50]}... after retries.")
                        await asyncio.sleep(IMAGE_GENERATION_QUOTA_SLEEP_SECONDS) # Longer sleep after repeated quota failure for one prompt
                        break # Break from attempt loop, move to next prompt
                
                except google_exceptions.GoogleAPICallError as api_call_error: # Catch other Google API errors
                    err_msg = f"Imagen API Call Error for prompt {i+1} (attempt {attempt+1}): {api_call_error}"
                    logger.error(f"[Task {task_id}] {err_msg}")
                    if attempt < MAX_API_RETRIES - 1:
                        sleep_time = (BASE_RETRY_SLEEP_SECONDS ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"[Task {task_id}] Retrying prompt {i+1} in {sleep_time:.2f}s due to API call error.")
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"[Task {task_id}] API Call error for prompt {i+1} after {MAX_API_RETRIES} attempts.")
                        errors.append(f"API call error for prompt: {prompt_text[:50]}... ({api_call_error})")
                        await asyncio.sleep(IMAGE_GENERATION_ERROR_SLEEP_SECONDS)
                        break # Break from attempt loop

                except Exception as e: # Catch other unexpected errors during Imagen call
                    err_msg = f"Unexpected Imagen Error for prompt {i+1} (attempt {attempt+1}): {e}"
                    logger.error(f"[Task {task_id}] {err_msg}", exc_info=True) # exc_info for full traceback
                    if attempt < MAX_API_RETRIES - 1:
                        sleep_time = (BASE_RETRY_SLEEP_SECONDS ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"[Task {task_id}] Retrying prompt {i+1} in {sleep_time:.2f}s due to unexpected error.")
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"[Task {task_id}] Unexpected error for prompt {i+1} after {MAX_API_RETRIES} attempts.")
                        errors.append(f"Unexpected error for prompt: {prompt_text[:50]}... ({e})")
                        await asyncio.sleep(IMAGE_GENERATION_ERROR_SLEEP_SECONDS)
                        break # Break from attempt loop
            # End of retry loop for a single prompt with Imagen
        else: # Imagen model instance not available
            logger.warning(f"[Task {task_id}] Imagen model not available for prompt {i+1}.")
            # errors.append("Imagen model not available.") # Not an error for this prompt if fallback exists

        # --- Cloudflare Fallback ---
        if not imagen_succeeded_this_prompt:
            if cloudflare_account_id_global and cloudflare_api_token_global:
                logger.info(f"[Task {task_id}] Attempting Cloudflare fallback for prompt {i+1} as Imagen failed or was unavailable.")
                # No complex retry for Cloudflare in this example, but could be added
                cf_image_bytes = await _generate_image_cloudflare_native(prompt_text)
                if cf_image_bytes:
                    image_bytes = cf_image_bytes
                    model_used = CLOUDFLARE_FALLBACK_MODEL
                    content_type = 'image/png' # Assume PNG for Cloudflare
                    logger.info(f"[Task {task_id}] Cloudflare fallback successful for prompt {i+1}.")
                else:
                    err_msg_cf = f"Cloudflare fallback also failed for prompt: {prompt_text[:50]}..."
                    logger.warning(f"[Task {task_id}] {err_msg_cf}")
                    errors.append(err_msg_cf) # Add Cloudflare failure to errors
                await asyncio.sleep(CLOUDFLARE_FALLBACK_SLEEP_SECONDS) # Pause after Cloudflare attempt
            elif imagen_model_instance: # Log only if Imagen was tried and failed, and CF is unavailable
                logger.warning(f"[Task {task_id}] Imagen failed for prompt {i+1} and Cloudflare fallback is disabled/unavailable.")
                # The error from Imagen attempts should already be in 'errors' list
            else: # Both Imagen and Cloudflare are unavailable
                err_msg_no_model = f"No image generation model (Imagen or Cloudflare) available for prompt: {prompt_text[:50]}..."
                logger.error(f"[Task {task_id}] {err_msg_no_model}")
                errors.append(err_msg_no_model)


        # --- Save Result if image_bytes were obtained ---
        if image_bytes and model_used:
            try:
                prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_')
                file_ext = "png" if content_type == 'image/png' else "jpg"
                final_content_type = content_type if content_type.startswith('image/') else 'image/jpeg'
                blob_name = f"adk_native/game_{game_pk_str}/img_{timestamp}_{i+1:02d}_{prompt_slug}.{file_ext}"
                
                gcs_uri = await _save_image_to_gcs_native(image_bytes, GCS_BUCKET_GENERATED_ASSETS, blob_name, final_content_type)
                if gcs_uri:
                    generated_uris.append(gcs_uri)
                else: # _save_image_to_gcs_native already logs its error
                    err_msg_save = f"Failed to save image from {model_used} to GCS for prompt {i+1} (GCS URI was None after save attempt)."
                    logger.warning(f"[Task {task_id}] {err_msg_save}")
                    errors.append(err_msg_save)
            except Exception as save_err: # Catch any other error during GCS save prep or call
                err_msg_save_exc = f"Exception during GCS save for image from {model_used} (prompt {i+1}): {save_err}"
                logger.error(f"[Task {task_id}] {err_msg_save_exc}", exc_info=True)
                errors.append(err_msg_save_exc)
        
        # General pause between processing different prompts if previous prompt processing didn't already sleep due to quota/error
        if i < len(prompts) - 1 and not (imagen_model_instance and not imagen_succeeded_this_prompt): # Avoid double sleep if fallback was used
            logger.debug(f"[Task {task_id}] General pause of {IMAGE_GENERATION_SLEEP_SECONDS}s before next prompt.")
            await asyncio.sleep(IMAGE_GENERATION_SLEEP_SECONDS)

    # --- Final Task Status Update ---
    if not generated_uris and errors:
        IMAGE_GENERATION_TASKS[task_id].update({
            "status": "failed",
            "error": "; ".join(errors) if errors else "Image generation task failed with no specific errors but no URIs."
        })
    elif errors: # Some URIs might have been generated, but some errors also occurred
        IMAGE_GENERATION_TASKS[task_id].update({
            "status": "completed_with_errors",
            "result": json.dumps(generated_uris), # Still provide successful URIs
            "errors": errors
        })
    else: # All successful
        IMAGE_GENERATION_TASKS[task_id].update({
            "status": "completed",
            "result": json.dumps(generated_uris),
            "errors": [] # No errors
        })
    logger.info(f"[Task {task_id}] Image generation work finished. Overall Status: {IMAGE_GENERATION_TASKS[task_id]['status']}. Generated URIs: {len(generated_uris)}. Errors: {len(errors)}")
def initiate_image_generation(prompts: List[str], game_pk_str: str = "unknown_game", tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    ensure_clients_initialized()
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        logger.error(f"LRFT:init_image_gen: Invalid 'prompts' type. Expected List[str], got {type(prompts)}. Value: {prompts}")
        return {"status": "error", "message": "Invalid prompts input. Must be a list of strings."}

    if not prompts:
        return {"status": "completed", "result": json.dumps([])}

    prompt_hash_part = hashlib.md5(json.dumps(sorted(prompts)).encode()).hexdigest()[:8]
    task_id = f"img_task_{prompt_hash_part}_{int(time.time())}"
    agent_name = tool_context.agent_name if tool_context and hasattr(tool_context, 'agent_name') else "UnknownAgentFromToolContext"
    logger.info(f"LRFT:init_image_gen (Agent: {agent_name}): Initiating task {task_id} for {len(prompts)} prompts, game {game_pk_str}.")

    IMAGE_GENERATION_TASKS[task_id] = {
        "status": "submitted",
        "prompts": prompts,
        "game_pk_str": game_pk_str,
        "start_time": time.time(),
    }
    asyncio.create_task(_perform_image_generation_work(task_id, prompts, game_pk_str))

    return {
        "status": "pending_agent_client_action",
        "task_id": task_id,
        "tool_name": "initiate_image_generation",
        "message": f"Image generation task {task_id} initiated for {len(prompts)} prompts. Awaiting client polling."
    }