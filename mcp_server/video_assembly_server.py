# mcp_server/video_assembly_server.py
import logging
import json
import os
import tempfile
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime, UTC
from google.cloud import storage
from moviepy import (VideoFileClip, ImageClip, AudioFileClip,
                            CompositeVideoClip, concatenate_videoclips) # Check imports
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
OUTPUT_GCS_BUCKET = os.getenv("OUTPUT_GCS_BUCKET", "mlb_final_videos")
OUTPUT_VIDEO_PREFIX = "final_recap_synced/"
DEFAULT_VISUAL_GCS_URI = os.getenv("DEFAULT_VISUAL_GCS_URI", "gs://mlb_logos/Major_League_Baseball_MLB_transparent_logo.png") # Example Default

# MoviePy Config
TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", 1280))
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 720))
TARGET_FPS = int(os.getenv("TARGET_FPS", 24))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("video_assembly")

# --- Initialize Clients ---
storage_client = None
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Storage client initialized.")
    # Verify moviepy is installed
    import moviepy
except ImportError:
     logger.critical("moviepy library not found. Video assembly will fail. Install with `pip install moviepy`.")
     # Exit or allow server to run with failures? For now, allow run.
except Exception as e:
    logger.critical(f"Failed to initialize storage client: {e}", exc_info=True)
    # Allow server to start but tools will fail

# --- Helper Functions ---
async def _download_gcs_blob_async(gcs_uri: str, local_dir: str) -> Optional[str]:
    """Downloads a blob from GCS to a local directory asynchronously."""
    if not storage_client or not gcs_uri or not gcs_uri.startswith("gs://"): return None
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        if not blob_name: return None # Cannot download bucket only
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_filename = os.path.join(local_dir, blob_name.replace("/", "_"))
        await asyncio.to_thread(blob.download_to_filename, local_filename)
        logger.info(f"Downloaded {gcs_uri} to {local_filename}")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download {gcs_uri}: {e}", exc_info=True)
        return None

async def _upload_blob_async(local_file_path: str, gcs_uri: str):
    """Uploads a local file to GCS asynchronously."""
    if not storage_client or not gcs_uri or not gcs_uri.startswith("gs://"): return None
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(blob.upload_from_filename, local_file_path)
        logger.info(f"Uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to {gcs_uri}: {e}", exc_info=True)
        return None

def _cleanup_temp_dir_sync(dir_path: str):
    """Synchronous helper to clean up temporary directory and files."""
    import shutil
    if dir_path and os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Removed temporary directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory {dir_path}: {e}")

# --- MCP Tool ---

@mcp.tool()
async def assemble_video_from_timeline(audio_gcs_uri: str, timeline_json: str, game_pk_str: str = "unknown_game") -> str:
    """
    Assembles a final video based on an audio track and a visual timeline JSON.
    Returns the GCS URI of the final video as a JSON string, or an error object.
    """
    if not storage_client:
        return json.dumps({"error": "GCS client not initialized."})
    if not audio_gcs_uri or not timeline_json:
        return json.dumps({"error": "Missing audio URI or timeline JSON."})
    try:
        import moviepy # Ensure it's loadable here too
    except ImportError:
         return json.dumps({"error": "moviepy library not installed."})


    logger.info(f"ASSEMBLY_MCP: assemble_video_from_timeline - Assembling video for game {game_pk_str}...")

    try:
        visual_timeline: List[Dict] = json.loads(timeline_json)
        if not isinstance(visual_timeline, list): raise ValueError("Timeline JSON must be a list.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid timeline_json input: {e}")
        return json.dumps({"error": f"Invalid timeline input: {e}"})

    if not visual_timeline:
        return json.dumps({"error": "Visual timeline is empty."})

    temp_dir = None
    local_files_to_clean = []
    processed_moviepy_clips = []
    opened_video_sources = []
    main_audio_clip = None
    final_video_no_audio = None
    final_video = None
    final_video_gcs_uri = None

    try:
        temp_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="video_assembly_")
        logger.info(f"Created temporary directory: {temp_dir}")

        # --- Download Assets ---
        logger.info("Downloading audio and visual assets...")
        download_tasks = []
        unique_uris = set(item['visual_uri'] for item in visual_timeline if item.get('visual_uri'))
        unique_uris.add(audio_gcs_uri)
        unique_uris.add(DEFAULT_VISUAL_GCS_URI) # Ensure default is downloaded

        uri_to_local_path = {}

        async def download_and_map(uri):
            if uri and uri.startswith("gs://"):
                local_path = await _download_gcs_blob_async(uri, temp_dir)
                if local_path:
                    uri_to_local_path[uri] = local_path
                    local_files_to_clean.append(local_path)
                else:
                    logger.warning(f"Failed to download asset {uri}.")
            else:
                logger.warning(f"Skipping invalid URI: {uri}")

        await asyncio.gather(*(download_and_map(uri) for uri in unique_uris))
        logger.info(f"Downloaded {len(uri_to_local_path)} assets.")

        # Check required assets
        local_audio_path = uri_to_local_path.get(audio_gcs_uri)
        local_default_visual_path = uri_to_local_path.get(DEFAULT_VISUAL_GCS_URI)
        if not local_audio_path: raise ValueError("Failed to download main audio.")
        if not local_default_visual_path: raise ValueError("Failed to download default visual.")

        # --- Load Audio ---
        main_audio_clip = await asyncio.to_thread(AudioFileClip, local_audio_path)
        total_audio_duration = main_audio_clip.duration
        if total_audio_duration <= 0: raise ValueError("Audio clip has zero duration.")
        logger.info(f"Loaded audio. Duration: {total_audio_duration:.2f} seconds.")

        # --- Process Timeline (Synchronous MoviePy parts in thread) ---
        logger.info(f"Processing {len(visual_timeline)} timeline segments...")

        def process_timeline_sync():
            # This function runs entirely in a separate thread
            sync_clips = []
            sync_opened_sources = []
            sync_last_end_time = 0.0

            for i, segment in enumerate(visual_timeline):
                start_time = segment.get('start_time_s', 0.0)
                end_time = segment.get('end_time_s', 0.0)
                visual_uri = segment.get('visual_uri')

                local_visual_path = uri_to_local_path.get(visual_uri)
                if not local_visual_path:
                    logger.warning(f"Segment {i+1}: Using default visual as asset download failed for {visual_uri}.")
                    local_visual_path = local_default_visual_path

                # Fill gaps
                if start_time > sync_last_end_time + 0.1:
                     gap_duration = start_time - sync_last_end_time
                     logger.warning(f"Filling gap ({gap_duration:.2f}s) with default visual.")
                     try:
                         gap_clip = (ImageClip(local_default_visual_path)
                                     .set_duration(gap_duration) # Use set_duration
                                     .set_start(sync_last_end_time)
                                     .resize(height=TARGET_HEIGHT))
                         sync_clips.append(gap_clip)
                     except Exception as gap_err: logger.error(f"Failed to create gap filler: {gap_err}")


                segment_duration = end_time - start_time
                if segment_duration <= 0.01:
                    logger.warning(f"Segment {i+1} duration too short. Skipping.")
                    sync_last_end_time = max(sync_last_end_time, end_time)
                    continue

                logger.info(f"  Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s -> {os.path.basename(local_visual_path)}")
                clip_to_add = None
                try:
                    is_video = local_visual_path.lower().endswith((".mp4", ".mov", ".avi"))
                    if is_video:
                        source_video_clip = VideoFileClip(local_visual_path)
                        sync_opened_sources.append(source_video_clip)
                        actual_duration = min(source_video_clip.duration, segment_duration)
                        if actual_duration <= 0.01: continue
                        clip_to_add = source_video_clip.subclip(0, actual_duration)
                    else: # Image
                        clip_to_add = ImageClip(local_visual_path).set_duration(segment_duration)
                        if local_visual_path.lower().endswith(".png"):
                             clip_to_add = clip_to_add.set_ismask(False) # Correct method

                    if clip_to_add:
                         clip_to_add = clip_to_add.resize(height=TARGET_HEIGHT).set_start(start_time)
                         # Explicitly set duration based on segment unless it's a shorter video
                         final_clip_duration = min(clip_to_add.duration, segment_duration)
                         clip_to_add = clip_to_add.set_duration(final_clip_duration)
                         sync_clips.append(clip_to_add)

                except Exception as clip_err:
                    logger.error(f"Error processing segment {i+1}: {clip_err}", exc_info=True)
                    # Add fallback default clip
                    try:
                          fallback_clip = (ImageClip(local_default_visual_path)
                                         .set_duration(segment_duration)
                                         .set_start(start_time)
                                         .resize(height=TARGET_HEIGHT))
                          sync_clips.append(fallback_clip)
                    except Exception as fb_err: logger.error(f"Failed fallback clip: {fb_err}")

                sync_last_end_time = max(sync_last_end_time, end_time)

            # Fill end gap
            if total_audio_duration > sync_last_end_time + 0.1:
                gap_duration = total_audio_duration - sync_last_end_time
                logger.warning(f"Filling final gap ({gap_duration:.2f}s).")
                try:
                    gap_clip = (ImageClip(local_default_visual_path)
                                .set_duration(gap_duration)
                                .set_start(sync_last_end_time)
                                .resize(height=TARGET_HEIGHT))
                    sync_clips.append(gap_clip)
                except Exception as gap_err: logger.error(f"Failed final gap filler: {gap_err}")

            if not sync_clips:
                raise ValueError("No visual clips prepared.")

            # --- Compose and Write Video ---
            logger.info(f"Composing {len(sync_clips)} clips in thread...")
            composed_video = CompositeVideoClip(sync_clips, size=(TARGET_WIDTH, TARGET_HEIGHT))
            final_video_with_audio = composed_video.set_audio(main_audio_clip)
            # Ensure duration matches audio precisely
            final_video_with_audio = final_video_with_audio.set_duration(total_audio_duration)

            timestamp_sync = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            local_output_filename_sync = os.path.join(temp_dir, f"final_game_{game_pk_str}_recap_sync_{timestamp_sync}.mp4")

            logger.info(f"Writing final video to {local_output_filename_sync} in thread...")
            final_video_with_audio.write_videofile(
                local_output_filename_sync, fps=TARGET_FPS, codec="libx264", audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, f'temp-audio-sync_{timestamp_sync}.m4a'),
                remove_temp=True, preset="medium", threads=4, logger='bar' # MoviePy logger
            )
            logger.info("Video writing complete in thread.")

            # Close clips used in this thread scope
            if main_audio_clip: main_audio_clip.close()
            if composed_video: composed_video.close()
            if final_video_with_audio: final_video_with_audio.close()
            for clip in sync_clips:
                if clip: clip.close()
            for source in sync_opened_sources:
                 if source: source.close()

            return local_output_filename_sync # Return the path of the generated file

        # Run the synchronous MoviePy processing in a thread
        local_output_filename = await asyncio.to_thread(process_timeline_sync)

        if not local_output_filename or not os.path.exists(local_output_filename):
             raise ValueError("Video assembly thread failed to produce output file.")

        local_files_to_clean.append(local_output_filename) # Add final video for cleanup

        # --- Upload Final Video ---
        timestamp_final = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        final_video_blob_name = f"{OUTPUT_VIDEO_PREFIX}game_{game_pk_str}/final_recap_synced_{timestamp_final}.mp4"
        final_video_gcs_uri_potential = f"gs://{OUTPUT_GCS_BUCKET}/{final_video_blob_name}"

        logger.info(f"Uploading final video to {final_video_gcs_uri_potential}...")
        uploaded_uri = await _upload_blob_async(local_output_filename, final_video_gcs_uri_potential)
        if uploaded_uri:
            final_video_gcs_uri = uploaded_uri
            logger.info("Upload successful.")
        else:
            logger.error(f"Upload failed. Video saved locally at {local_output_filename}")
            raise ValueError("Failed to upload final video.")


    except Exception as e:
        logger.error(f"ERROR during video assembly: {e}", exc_info=True)
        return json.dumps({"error": f"Failed during video assembly: {e}"})
    finally:
        # Cleanup (run synchronous part in thread)
        logger.info("Cleaning up temporary assembly files...")
        await asyncio.to_thread(_cleanup_temp_dir_sync, temp_dir)
        logger.info("Cleanup finished.")


    logger.info(f"ASSEMBLY_MCP: Video assembly complete. URI: {final_video_gcs_uri}")
    return json.dumps({"final_video_uri": final_video_gcs_uri})


if __name__ == "__main__":
    logger.info("Starting Video Assembly MCP Server (MoviePy)...")
    mcp.run(transport="stdio")