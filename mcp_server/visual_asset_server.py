# mcp_server/visual_asset_server.py
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
import requests
from PIL import Image as PilImage
from google.cloud import storage, secretmanager
from google.api_core import exceptions as google_exceptions
from vertexai.preview.vision_models import ImageGenerationModel
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Needed for Imagen client
VERTEX_IMAGEN_MODEL = os.getenv("VERTEX_IMAGEN_MODEL", "imagen-3.0-generate-002")
GCS_BUCKET_GENERATED_ASSETS = os.getenv("GCS_BUCKET_GENERATED_ASSETS", "mlb_generated_assets")

# Cloudflare Fallback Config
CLOUDFLARE_ACCOUNT_ID_SECRET = os.getenv("CLOUDFLARE_ACCOUNT_ID_SECRET", "cloudflare-account-id")
CLOUDFLARE_API_TOKEN_SECRET = os.getenv("CLOUDFLARE_API_TOKEN_SECRET", "cloudflare-api-token")
CLOUDFLARE_FALLBACK_MODEL = os.getenv("CLOUDFLARE_FALLBACK_MODEL", "@cf/bytedance/stable-diffusion-xl-lightning")
CLOUDFLARE_API_ENDPOINT_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"

# Generation Parameters
IMAGE_GENERATION_SEED = None # Or getenv with int conversion
IMAGE_GENERATION_WATERMARK = False # Or getenv with bool conversion
IMAGE_GENERATION_NEGATIVE_PROMPT = "text, words, letters, blurry, low quality, cartoonish, illustration, drawing, sketch, unrealistic, watermark, signature, writing"
IMAGE_GENERATION_ASPECT_RATIO = "16:9"
IMAGE_GENERATION_NUMBER_PER_PROMPT = 1

# Sleep/Retry Config
IMAGE_GENERATION_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_SLEEP_SECONDS", 35))
IMAGE_GENERATION_ERROR_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_ERROR_SLEEP_SECONDS", 15))
IMAGE_GENERATION_QUOTA_SLEEP_SECONDS = int(os.getenv("IMAGE_GENERATION_QUOTA_SLEEP_SECONDS", 70))
CLOUDFLARE_FALLBACK_SLEEP_SECONDS = int(os.getenv("CLOUDFLARE_FALLBACK_SLEEP_SECONDS", 5))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("visual_assets")

# --- Initialize Clients ---
storage_client = None
imagen_model = None
secret_manager_client = None
cloudflare_account_id = None
cloudflare_api_token = None

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    # Vertex AI SDK needs initialization for the project/location
    import vertexai
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    imagen_model = ImageGenerationModel.from_pretrained(VERTEX_IMAGEN_MODEL)
    secret_manager_client = secretmanager.SecretManagerServiceClient()
    logger.info(f"Storage, Imagen ({VERTEX_IMAGEN_MODEL}), Secret Manager clients initialized.")

    # --- Helper Function to Access Secret Manager ---
    def access_secret_version(project_id: str, secret_id: str, version_id: str = "latest") -> Optional[str]:
        if not secret_manager_client: return None
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        try:
            response = secret_manager_client.access_secret_version(request={"name": name})
            payload = response.payload.data.decode("UTF-8")
            logger.info(f"Successfully accessed secret: {secret_id}")
            return payload
        except Exception as e:
            logger.error(f"Error accessing secret {secret_id}: {e}", exc_info=False) # Keep log cleaner
            return None

    # Fetch Cloudflare credentials
    cloudflare_account_id = access_secret_version(GCP_PROJECT_ID, CLOUDFLARE_ACCOUNT_ID_SECRET)
    cloudflare_api_token = access_secret_version(GCP_PROJECT_ID, CLOUDFLARE_API_TOKEN_SECRET)
    if not cloudflare_account_id or not cloudflare_api_token:
        logger.warning("Cloudflare credentials not found/accessible. Fallback generation disabled.")
    else:
        logger.info("Cloudflare credentials loaded.")

except Exception as e:
    logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
    # Allow server to start but tools will fail

# --- Helper Functions ---

async def _save_image_to_gcs_async(image_bytes: bytes, bucket_name: str, blob_name: str, content_type='image/png') -> Optional[str]:
    """Saves image bytes to GCS asynchronously."""
    if not storage_client:
        logger.error("GCS storage client not initialized.")
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Use asyncio.to_thread for the blocking upload_from_string call
        await asyncio.to_thread(
            blob.upload_from_string,
            image_bytes,
            content_type=content_type
        )
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved generated image to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving image to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

async def _generate_image_cloudflare_async(prompt: str, width: int = 768, height: int = 768, num_steps: int = 20) -> Optional[bytes]:
    """Generates image using Cloudflare Worker AI asynchronously."""
    if not cloudflare_account_id or not cloudflare_api_token:
        logger.warning("Cloudflare credentials not available, skipping fallback.")
        return None

    url = CLOUDFLARE_API_ENDPOINT_TEMPLATE.format(account_id=cloudflare_account_id, model_id=CLOUDFLARE_FALLBACK_MODEL)
    headers = {"Authorization": f"Bearer {cloudflare_api_token}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "width": width, "height": height, "num_steps": num_steps}

    logger.info(f"Attempting Cloudflare fallback ({CLOUDFLARE_FALLBACK_MODEL}) for prompt: '{prompt[:80]}...'")

    try:
        # Use requests library within asyncio.to_thread for blocking I/O
        response = await asyncio.to_thread(
            requests.post, url, headers=headers, json=data, timeout=60
        )
        if response.status_code == 200:
            logger.info("Cloudflare fallback generation successful.")
            return response.content
        else:
            logger.error(f"Cloudflare fallback failed. Status: {response.status_code}, Response: {response.text[:200]}...")
            return None
    except Exception as e:
        logger.error(f"Error during Cloudflare API request: {e}", exc_info=True)
        return None

# --- MCP Tool ---

@mcp.tool()
async def generate_images_from_prompts(prompts: List[str], game_pk_str: str = "unknown_game") -> str:
    """
    Generates images for a list of prompts using Imagen, with Cloudflare fallback.
    Expects prompts_json as a JSON string list. Returns a JSON string list of GCS URIs.
    game_pk_str is used for GCS path naming.
    """
    if not imagen_model and not (cloudflare_account_id and cloudflare_api_token):
        return json.dumps({"error": "No image generation model (Imagen or Cloudflare) is available."})
    if not storage_client:
        return json.dumps({"error": "GCS client not initialized."})

    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        logger.error(f"MCP Tool received invalid 'prompts' type. Expected List[str], got {type(prompts)}. Value: {prompts}")
        return json.dumps({"error": f"MCP Tool received invalid 'prompts' type. Expected List[str], got {type(prompts)}."})
    if not prompts:
        logger.info("No prompts provided for image generation.")
        return json.dumps([]) # Return empty list

    logger.info(f"VISUAL_MCP: generate_images_from_prompts - Processing {len(prompts)} prompts for game {game_pk_str}.")

    generated_uris = []
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    imagen_available = bool(imagen_model)

    generation_params = {
        "number_of_images": IMAGE_GENERATION_NUMBER_PER_PROMPT,
        "aspect_ratio": IMAGE_GENERATION_ASPECT_RATIO,
        "add_watermark": IMAGE_GENERATION_WATERMARK,
        "negative_prompt": IMAGE_GENERATION_NEGATIVE_PROMPT,
        # Seed handling per prompt
    }

    for i, prompt_text in enumerate(prompts):
        logger.info(f"Processing visual prompt {i+1}/{len(prompts)}: '{prompt_text[:80]}...'")
        image_bytes = None
        model_used = None
        content_type = 'image/png' # Default for Cloudflare, Imagen might vary but PNG is safe

        # --- Attempt 1: Imagen ---
        imagen_succeeded = False
        if imagen_available:
            try:
                current_seed = IMAGE_GENERATION_SEED
                if current_seed is None and not IMAGE_GENERATION_WATERMARK:
                    current_seed = random.randint(1, 2**31 - 1)

                # Use asyncio.to_thread for the blocking SDK call
                response = await asyncio.to_thread(
                    imagen_model.generate_images,
                    prompt=prompt_text,
                    seed=current_seed,
                    number_of_images=generation_params["number_of_images"],
                    aspect_ratio=generation_params["aspect_ratio"],
                    add_watermark=generation_params["add_watermark"],
                    negative_prompt=generation_params["negative_prompt"]
                )

                if response and response.images:
                    # Access image bytes directly (assuming SDK provides this)
                    # Check SDK documentation for exact attribute name
                    # Example: response.images[0]._image_bytes
                    # If it only gives PIL, convert back to bytes
                    img_object = response.images[0]
                    # Assuming img_object has ._image_bytes attribute
                    if hasattr(img_object, '_image_bytes') and img_object._image_bytes:
                         image_bytes = img_object._image_bytes
                         # Try to get mime type if available, else keep default
                         content_type = getattr(img_object, 'mime_type', 'image/png')
                         model_used = VERTEX_IMAGEN_MODEL
                         imagen_succeeded = True
                         logger.info(f" --> Imagen generation successful.")
                    else: # Fallback: Try converting PIL image if bytes aren't direct
                        try:
                            pil_img = img_object._pil_image # Access PIL object
                            buffer = io.BytesIO()
                            pil_img.save(buffer, format="PNG") # Save as PNG bytes
                            image_bytes = buffer.getvalue()
                            content_type = 'image/png'
                            model_used = VERTEX_IMAGEN_MODEL
                            imagen_succeeded = True
                            logger.info(f" --> Imagen generation successful (converted from PIL).")
                        except Exception as pil_err:
                             logger.warning(f"Could not get bytes from Imagen response: {pil_err}")


                else:
                    logger.warning(f"Imagen returned no image for prompt {i+1}.")

                if imagen_succeeded:
                    await asyncio.sleep(IMAGE_GENERATION_SLEEP_SECONDS / 1000.0) # Convert ms to s if needed


            except google_exceptions.ResourceExhausted as quota_error:
                logger.error(f"Imagen Quota Exceeded for prompt {i+1}: {quota_error}. Sleeping {IMAGE_GENERATION_QUOTA_SLEEP_SECONDS}s.")
                await asyncio.sleep(IMAGE_GENERATION_QUOTA_SLEEP_SECONDS)
                continue # Skip to next prompt, no fallback on quota error

            except Exception as e:
                logger.error(f"Unexpected Imagen Error for prompt {i+1}: {e}", exc_info=True)
                logger.warning("Will attempt fallback.")
                await asyncio.sleep(IMAGE_GENERATION_ERROR_SLEEP_SECONDS)

        # --- Attempt 2: Cloudflare Fallback ---
        if not imagen_succeeded:
             if cloudflare_account_id and cloudflare_api_token:
                 cf_image_bytes = await _generate_image_cloudflare_async(prompt_text)
                 if cf_image_bytes:
                     image_bytes = cf_image_bytes
                     model_used = CLOUDFLARE_FALLBACK_MODEL
                     content_type = 'image/png' # Cloudflare usually returns PNG
                     logger.info(f" --> Cloudflare fallback successful.")
                 else:
                     logger.warning(f" --> Cloudflare fallback also failed for prompt {i+1}.")
                 await asyncio.sleep(CLOUDFLARE_FALLBACK_SLEEP_SECONDS)
             elif imagen_available: # Only log warning if Imagen was available but failed
                 logger.warning("Imagen failed and Cloudflare fallback is disabled.")


        # --- Save Result ---
        if image_bytes and model_used:
            try:
                prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_')
                # Use JPEG for storage efficiency unless PNG transparency is needed
                file_ext = "png" if content_type == 'image/png' else "jpg"
                final_content_type = content_type if content_type.startswith('image/') else 'image/jpeg' # Default if type unknown

                blob_name = f"generated/game_{game_pk_str}/img_{timestamp}_{i+1:02d}_{prompt_slug}.{file_ext}"
                gcs_uri = await _save_image_to_gcs_async(image_bytes, GCS_BUCKET_GENERATED_ASSETS, blob_name, final_content_type)
                if gcs_uri:
                    generated_uris.append(gcs_uri)
                else:
                    logger.warning(f"Failed to save generated image to GCS for prompt {i+1} (model: {model_used}).")
            except Exception as save_err:
                 logger.error(f"Failed to save image from {model_used} for prompt {i+1}: {save_err}", exc_info=True)

    logger.info(f"VISUAL_MCP: Finished image generation. Generated {len(generated_uris)} URIs.")
    return json.dumps(generated_uris)


if __name__ == "__main__":
    logger.info("Starting Visual Asset Generation MCP Server...")
    mcp.run(transport="stdio")