# mcp_server/audio_processing_server.py
import logging
import json
import os
import io
import time
import tempfile
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, UTC

from google.cloud import storage, texttospeech, speech
from google.api_core import exceptions as google_exceptions
from pydub import AudioSegment # Requires pydub and ffmpeg
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCS_BUCKET_GENERATED_AUDIO = os.getenv("GCS_BUCKET_GENERATED_AUDIO", "mlb_generated_audio")

# TTS Config
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "en-US-Chirp3-HD-Puck")
TTS_VOICE_NAME_ALT = os.getenv("TTS_VOICE_NAME_ALT", "en-US-Chirp3-HD-Aoede")
AUDIO_ENCODING = texttospeech.AudioEncoding.MP3
AUDIO_SILENCE_MS = int(os.getenv("AUDIO_SILENCE_MS", 350))

# STT Config
STT_LANGUAGE_CODE = "en-US"
STT_SAMPLE_RATE_HZ = int(os.getenv("STT_SAMPLE_RATE_HZ", 24000)) # IMPORTANT: Match TTS output rate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("audio_processing_mcp")

# --- Initialize Clients ---
storage_client = None
tts_client = None
stt_client = None

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech.SpeechClient()
    logger.info("Storage, TTS, and STT clients initialized.")
    # Check for pydub/ffmpeg early
    from pydub import AudioSegment
except ImportError:
     logger.critical("pydub library not found. Audio combination will fail. Install with `pip install pydub` and ensure ffmpeg is installed.")
     # Exit or allow server to run with failures? For now, allow run.
except Exception as e:
    logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
    # Allow server to start but tools will fail

# --- Helper Functions ---
async def _save_audio_to_gcs_async(audio_bytes: bytes, bucket_name: str, blob_name: str) -> Optional[str]:
    """Saves audio bytes to GCS asynchronously."""
    if not storage_client: return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(blob.upload_from_string, audio_bytes, content_type='audio/mpeg')
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved generated audio to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving audio to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

# --- MCP Tools ---

@mcp.tool()
async def synthesize_multi_speaker_speech(script: str, game_pk_str: str = "unknown_game") -> str:
    """
    Generates multi-speaker MP3 audio from a dialogue script.
    Returns a JSON string: {"audio_uri": "gs://bucket/object_name.mp3"} on success,
    or {"error": "error message"} on failure.
    """
    if not tts_client or not storage_client:
        logger.error("AUDIO_MCP: synthesize_multi_speaker_speech - TTS or Storage client not initialized.")
        return json.dumps({"error": "Server configuration error: TTS or Storage client not initialized."})
    if not script:
        logger.warning("AUDIO_MCP: synthesize_multi_speaker_speech - Script cannot be empty.")
        return json.dumps({"error": "Input error: Script cannot be empty."})

    logger.info(f"AUDIO_MCP: synthesize_multi_speaker_speech - Synthesizing audio for game {game_pk_str}...")

    dialogue_lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not dialogue_lines:
        logger.warning("AUDIO_MCP: synthesize_multi_speaker_speech - Script has no content after stripping.")
        return json.dumps({"error": "Input error: Script has no processable content."})

    temp_dir = None
    gcs_audio_uri = None # Renamed for clarity

    try:
        temp_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="audio_synth_")
        logger.info(f"Created temporary directory: {temp_dir}")

        temp_audio_files = []
        tasks = []
        for count, line in enumerate(dialogue_lines):
            voice_name = TTS_VOICE_NAME if count % 2 == 0 else TTS_VOICE_NAME_ALT
            temp_filename = os.path.join(temp_dir, f"part-{count}.mp3")
            tasks.append(asyncio.to_thread(
                _generate_tts_segment, line, voice_name, temp_filename, count+1
            ))
        segment_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(segment_results):
            if isinstance(result, str) and os.path.exists(result):
                temp_audio_files.append(result)
            elif isinstance(result, Exception):
                 logger.error(f"TTS generation failed for segment {i+1}: {result}")
            else:
                 logger.warning(f"Unexpected result for segment {i+1}: {result}")

        if not temp_audio_files:
            raise ValueError("No audio segments were successfully generated.")

        logger.info(f"Combining {len(temp_audio_files)} audio segments...")
        combined_audio_bytes = await asyncio.to_thread(
            _combine_audio_segments, temp_audio_files, AUDIO_SILENCE_MS
        )

        if not combined_audio_bytes:
            raise ValueError("Failed to combine audio segments or combined audio is empty.")

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        # Ensure GCS_BUCKET_GENERATED_AUDIO is correctly defined and accessible
        blob_name = f"generated/audio/game_{game_pk_str}/dialogue_audio_{timestamp}.mp3"
        gcs_audio_uri = await _save_audio_to_gcs_async(combined_audio_bytes, GCS_BUCKET_GENERATED_AUDIO, blob_name)

        if not gcs_audio_uri:
            raise ValueError("Failed to upload combined audio to GCS or GCS URI was not returned.")

        logger.info(f"AUDIO_MCP: Audio synthesis successful. URI: {gcs_audio_uri}")
        # THIS IS THE CRITICAL LINE FOR THE RETURN FORMAT
        return json.dumps({"audio_uri": gcs_audio_uri})

    except Exception as e:
        logger.error(f"AUDIO_MCP: Error during multi-speaker audio generation for game {game_pk_str}: {e}", exc_info=True)
        return json.dumps({"error": f"Failed during audio generation: {str(e)}"})
    finally:
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary audio files in {temp_dir}...")
            await asyncio.to_thread(_cleanup_temp_dir, temp_dir)


# Helper for TTS generation (to be run in thread)
def _generate_tts_segment(line: str, voice_name: str, output_path: str, segment_num: int) -> str:
    """Synchronous helper to generate a single TTS segment."""
    if not tts_client: raise RuntimeError("TTS client not available in helper thread.")
    logger.debug(f"  Generating segment {segment_num} ('{line[:30]}...')...")
    synthesis_input = texttospeech.SynthesisInput(text=line)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=AUDIO_ENCODING, sample_rate_hertz=STT_SAMPLE_RATE_HZ) # Ensure matching sample rate

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    return output_path # Return path on success

# Synchronous helper to combine audio segments using pydub
def _combine_audio_segments(file_paths: List[str], silence_ms: int) -> Optional[bytes]:
    try:
        # Start with silence if there are actual segments to process, otherwise start empty
        full_audio = AudioSegment.empty()
        if file_paths: # Only add initial silence if there's content
             full_audio = AudioSegment.silent(duration=silence_ms)

        for i, file_path in enumerate(file_paths):
            segment = AudioSegment.from_mp3(file_path)
            full_audio += segment
            if i < len(file_paths) - 1: # Add silence after each segment except the last one
                full_audio += AudioSegment.silent(duration=silence_ms)
        
        if not file_paths: # If no files, return None or empty bytes
            logger.warning("No audio file paths provided to combine.")
            return None

        buffer = io.BytesIO()
        full_audio.export(buffer, format="mp3")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error combining audio segments with pydub: {e}", exc_info=True)
        return None

# Helper for cleaning up temp dir (to be run in thread)
def _cleanup_temp_dir(dir_path: str):
    """Synchronous helper to clean up temporary directory and files."""
    import shutil
    try:
        shutil.rmtree(dir_path)
        logger.info(f"Removed temporary directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Could not remove temp directory {dir_path}: {e}")


@mcp.tool()
async def get_word_timestamps_from_audio(audio_gcs_uri: str) -> str:
    """
    Transcribes audio from GCS using STT to get word timestamps.
    Returns a JSON string list of timestamp dicts or an error object.
    """
    if not stt_client:
        return json.dumps({"error": "STT client not initialized."})
    if not audio_gcs_uri or not audio_gcs_uri.startswith("gs://"):
        return json.dumps({"error": "Invalid GCS URI for STT."})

    logger.info(f"AUDIO_MCP: get_word_timestamps - Transcribing {audio_gcs_uri}...")
    word_timestamps = []

    try:
        config = speech.RecognitionConfig(
            sample_rate_hertz=STT_SAMPLE_RATE_HZ, # Use configured sample rate
            language_code=STT_LANGUAGE_CODE,
            enable_word_time_offsets=True,
            # encoding=speech.RecognitionConfig.AudioEncoding.MP3, # Let STT auto-detect
            # model="long", # Consider specifying model for potentially better accuracy
        )
        audio = speech.RecognitionAudio(uri=audio_gcs_uri)

        # Use asyncio.to_thread for the blocking long_running_recognize call
        operation = await asyncio.to_thread(
            stt_client.long_running_recognize,
            config=config, audio=audio
        )
        # Also run operation.result() in a thread
        logger.info("Waiting for STT operation result...")
        response = await asyncio.to_thread(operation.result, timeout=300)
        logger.info("STT Transcription complete.")

        # Process results
        for result in response.results:
            if result.alternatives:
                alternative = result.alternatives[0]
                for word_info in alternative.words:
                    word_timestamps.append({
                        "word": word_info.word,
                        "start_time_s": word_info.start_time.total_seconds(),
                        "end_time_s": word_info.end_time.total_seconds(),
                    })

        logger.info(f"AUDIO_MCP: Extracted timestamps for {len(word_timestamps)} words.")

    except google_exceptions.NotFound:
        logger.error(f"STT Error: Audio file not found at {audio_gcs_uri}")
        return json.dumps({"error": f"Audio not found at GCS URI: {audio_gcs_uri}"})
    except Exception as e:
        logger.error(f"Error during STT transcription: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to transcribe audio: {e}"})

    return json.dumps(word_timestamps)

if __name__ == "__main__":
    logger.info("Starting Audio Processing MCP Server (TTS & STT)...")
    mcp.run(transport="stdio")