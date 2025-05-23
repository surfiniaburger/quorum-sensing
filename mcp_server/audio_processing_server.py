# mcp_server/audio_processing_server.py
import logging
import json
import os
import io
import time # Make sure time is imported if you use it for precise timing, though not strictly needed for these logs
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

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__) # Get logger for this specific module

mcp = FastMCP("audio_processing_mcp")

# --- Initialize Clients ---
storage_client = None
tts_client = None
stt_client = None

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech.SpeechClient()
    logger.info("Storage, TTS, and STT clients initialized successfully.")
    # Check for pydub/ffmpeg early
    from pydub import AudioSegment
    logger.info("Pydub AudioSegment imported successfully.")
except ImportError:
     logger.critical("pydub library not found. Audio combination will fail. Install with `pip install pydub` and ensure ffmpeg is installed on the system.")
     # Exit or allow server to run with failures? For now, allow run.
except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud clients or import pydub: {e}", exc_info=True)
    # Allow server to start but tools will likely fail

# --- Helper Functions ---
async def _save_audio_to_gcs_async(audio_bytes: bytes, bucket_name: str, blob_name: str) -> Optional[str]:
    """Saves audio bytes to GCS asynchronously."""
    logger.debug(f"Attempting to save audio to GCS: gs://{bucket_name}/{blob_name}")
    if not storage_client:
        logger.error("Storage client not initialized. Cannot save to GCS.")
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Using asyncio.to_thread for the blocking GCS call
        await asyncio.to_thread(blob.upload_from_string, audio_bytes, content_type='audio/mpeg')
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved {len(audio_bytes)} bytes to GCS: {gcs_uri}")
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
    tool_start_time = time.monotonic()
    logger.info(f"MCP TOOL: synthesize_multi_speaker_speech INVOKED. game_pk_str='{game_pk_str}', script (first 100 chars)='{script[:100]}...'")

    if not tts_client or not storage_client:
        logger.error("AUDIO_MCP: synthesize_multi_speaker_speech - TTS or Storage client not initialized.")
        return json.dumps({"error": "Server configuration error: TTS or Storage client not initialized."})
    if not script:
        logger.warning("AUDIO_MCP: synthesize_multi_speaker_speech - Script cannot be empty.")
        return json.dumps({"error": "Input error: Script cannot be empty."})

    dialogue_lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not dialogue_lines:
        logger.warning("AUDIO_MCP: synthesize_multi_speaker_speech - Script has no content after stripping.")
        return json.dumps({"error": "Input error: Script has no processable content."})

    logger.info(f"Processing {len(dialogue_lines)} dialogue lines for TTS.")

    temp_dir = None
    gcs_audio_uri = None

    try:
        # Using asyncio.to_thread for synchronous tempfile.mkdtemp
        temp_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="audio_synth_")
        logger.info(f"Created temporary directory for audio segments: {temp_dir}")

        temp_audio_files = []
        tasks = []
        for count, line in enumerate(dialogue_lines):
            voice_name = TTS_VOICE_NAME if count % 2 == 0 else TTS_VOICE_NAME_ALT
            temp_filename = os.path.join(temp_dir, f"part-{count}.mp3")
            # Pass clients and config to the threadable function if they are not global or easily accessible
            tasks.append(asyncio.to_thread(
                _generate_tts_segment, line, voice_name, temp_filename, count+1, tts_client, AUDIO_ENCODING, STT_SAMPLE_RATE_HZ
            ))

        segment_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(segment_results):
            if isinstance(result, str) and os.path.exists(result):
                temp_audio_files.append(result)
                logger.debug(f"TTS segment {i+1} generated successfully: {result}")
            elif isinstance(result, Exception):
                 logger.error(f"TTS generation failed for segment {i+1}: {result}", exc_info=True) # Log with exc_info
            else:
                 logger.warning(f"Unexpected result type for TTS segment {i+1}: {type(result)}, value: {result}")


        if not temp_audio_files:
            logger.error("No audio segments were successfully generated after TTS process.")
            raise ValueError("No audio segments were successfully generated.")

        logger.info(f"Combining {len(temp_audio_files)} audio segments using pydub...")
        combined_audio_bytes = await asyncio.to_thread(
            _combine_audio_segments, temp_audio_files, AUDIO_SILENCE_MS
        )

        if not combined_audio_bytes:
            logger.error("Failed to combine audio segments or combined audio is empty after pydub processing.")
            raise ValueError("Failed to combine audio segments or combined audio is empty.")
        logger.info(f"Combined audio bytes length: {len(combined_audio_bytes)}")

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        blob_name = f"generated/audio/game_{game_pk_str}/dialogue_audio_{timestamp}.mp3"
        logger.info(f"Attempting to save combined audio to GCS blob: {blob_name} in bucket {GCS_BUCKET_GENERATED_AUDIO}")
        gcs_audio_uri = await _save_audio_to_gcs_async(combined_audio_bytes, GCS_BUCKET_GENERATED_AUDIO, blob_name)

        if not gcs_audio_uri:
            logger.error("Failed to upload combined audio to GCS or GCS URI was not returned after save attempt.")
            raise ValueError("Failed to upload combined audio to GCS or GCS URI was not returned.")

        tool_duration = time.monotonic() - tool_start_time
        logger.info(f"MCP TOOL: synthesize_multi_speaker_speech COMPLETED SUCCESSFULLY. URI: {gcs_audio_uri}, Duration: {tool_duration:.2f}s")
        return json.dumps({"audio_uri": gcs_audio_uri})

    except Exception as e:
        tool_duration = time.monotonic() - tool_start_time
        logger.error(f"MCP TOOL: synthesize_multi_speaker_speech FAILED for game {game_pk_str}. Error: {e}, Duration: {tool_duration:.2f}s", exc_info=True)
        return json.dumps({"error": f"Failed during audio generation: {str(e)}"})
    finally:
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary audio files and directory {temp_dir}...")
            # Using asyncio.to_thread for synchronous _cleanup_temp_dir
            await asyncio.to_thread(_cleanup_temp_dir, temp_dir)


# Helper for TTS generation (to be run in thread)
# Added tts_client_local, audio_encoding_local, sample_rate_local as parameters
def _generate_tts_segment(line: str, voice_name: str, output_path: str, segment_num: int,
                          tts_client_local: texttospeech.TextToSpeechClient,
                          audio_encoding_local: texttospeech.AudioEncoding,
                          sample_rate_local: int) -> str:
    """Synchronous helper to generate a single TTS segment."""
    logger.debug(f"  TTS Segment Generation Thread: Starting segment {segment_num} ('{line[:30]}...'), Voice: {voice_name}")
    if not tts_client_local:
        logger.error("  TTS Segment Generation Thread: TTS client not available in helper thread.")
        raise RuntimeError("TTS client not available in helper thread.")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=line)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding_local, sample_rate_hertz=sample_rate_local)

        response = tts_client_local.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        logger.debug(f"  TTS Segment Generation Thread: Finished segment {segment_num}, saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"  TTS Segment Generation Thread: Error in segment {segment_num} for voice {voice_name}: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by asyncio.gather

# Synchronous helper to combine audio segments using pydub
def _combine_audio_segments(file_paths: List[str], silence_ms: int) -> Optional[bytes]:
    logger.debug(f"Pydub Combine Thread: Starting combination of {len(file_paths)} segments with {silence_ms}ms silence.")
    try:
        full_audio = AudioSegment.empty()
        if file_paths:
             full_audio = AudioSegment.silent(duration=silence_ms) # Start with silence if segments exist

        for i, file_path in enumerate(file_paths):
            logger.debug(f"  Pydub Combine Thread: Loading segment {i+1} from {file_path}")
            segment = AudioSegment.from_mp3(file_path)
            full_audio += segment
            if i < len(file_paths) - 1: # Add silence after each segment except the last one
                full_audio += AudioSegment.silent(duration=silence_ms)
        
        if not file_paths:
            logger.warning("Pydub Combine Thread: No audio file paths provided to combine. Returning None.")
            return None

        buffer = io.BytesIO()
        full_audio.export(buffer, format="mp3")
        logger.debug(f"Pydub Combine Thread: Exported combined audio, length {len(buffer.getvalue())} bytes.")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Pydub Combine Thread: Error combining audio segments: {e}", exc_info=True)
        return None # Return None on failure

# Helper for cleaning up temp dir (to be run in thread)
def _cleanup_temp_dir(dir_path: str):
    """Synchronous helper to clean up temporary directory and files."""
    import shutil # Import here if not already global in this file
    logger.debug(f"Cleanup Thread: Attempting to remove directory tree: {dir_path}")
    try:
        shutil.rmtree(dir_path)
        logger.info(f"Cleanup Thread: Successfully removed temporary directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Cleanup Thread: Could not remove temp directory {dir_path}: {e}", exc_info=True)


@mcp.tool()
async def get_word_timestamps_from_audio(audio_gcs_uri: str) -> str:
    """
    Transcribes audio from GCS using STT to get word timestamps.
    Returns a JSON string list of timestamp dicts or an error object.
    """
    tool_start_time = time.monotonic()
    logger.info(f"MCP TOOL: get_word_timestamps_from_audio INVOKED. audio_gcs_uri='{audio_gcs_uri}'")

    if not stt_client:
        logger.error("AUDIO_MCP: get_word_timestamps - STT client not initialized.")
        return json.dumps({"error": "Server configuration error: STT client not initialized."})
    if not audio_gcs_uri or not audio_gcs_uri.startswith("gs://"):
        logger.warning(f"AUDIO_MCP: get_word_timestamps - Invalid GCS URI received: '{audio_gcs_uri}'. Expected 'gs://' prefix.")
        return json.dumps({"error": f"Invalid GCS URI for STT: '{audio_gcs_uri}'. Must start with 'gs://'."})

    word_timestamps = []
    try:
        config = speech.RecognitionConfig(
            sample_rate_hertz=STT_SAMPLE_RATE_HZ,
            language_code=STT_LANGUAGE_CODE,
            enable_word_time_offsets=True,
            # model="long", # Consider "medical_dictation" or "telephony" if more specific and applicable
        )
        audio = speech.RecognitionAudio(uri=audio_gcs_uri)
        logger.info(f"STT config: SampleRate={STT_SAMPLE_RATE_HZ}, Lang={STT_LANGUAGE_CODE}, WordTimeOffsets=True")

        logger.info(f"Calling STT long_running_recognize for {audio_gcs_uri}...")
        operation = await asyncio.to_thread(
            stt_client.long_running_recognize,
            config=config, audio=audio
        )
        operation_name = operation.operation.name if hasattr(operation, 'operation') else "N/A"
        logger.info(f"STT operation started (Name: {operation_name}). Waiting for result (timeout 300s)...")

        response = await asyncio.to_thread(operation.result, timeout=300)
        logger.info(f"STT operation {operation_name} completed.")

        processed_results = False
        for result_idx, result in enumerate(response.results):
            if result.alternatives:
                alternative = result.alternatives[0]
                logger.debug(f"Processing STT result alternative {result_idx+1} with {len(alternative.words)} words. Transcript snippet: '{alternative.transcript[:100]}...'")
                for word_info in alternative.words:
                    word_timestamps.append({
                        "word": word_info.word,
                        "start_time_s": word_info.start_time.total_seconds(),
                        "end_time_s": word_info.end_time.total_seconds(),
                    })
                processed_results = True
        if not processed_results:
            logger.warning(f"STT response for {audio_gcs_uri} contained no processable alternatives or words.")


        tool_duration = time.monotonic() - tool_start_time
        logger.info(f"MCP TOOL: get_word_timestamps_from_audio COMPLETED. Extracted timestamps for {len(word_timestamps)} words. Duration: {tool_duration:.2f}s")
        return json.dumps(word_timestamps)

    except google_exceptions.NotFound:
        tool_duration = time.monotonic() - tool_start_time
        logger.error(f"MCP TOOL: get_word_timestamps_from_audio FAILED. STT Error: Audio file not found at {audio_gcs_uri}. Duration: {tool_duration:.2f}s", exc_info=True)
        return json.dumps({"error": f"Audio file not found by STT service at GCS URI: {audio_gcs_uri}"})
    except google_exceptions.InvalidArgument as e:
        tool_duration = time.monotonic() - tool_start_time
        logger.error(f"MCP TOOL: get_word_timestamps_from_audio FAILED. STT Error: Invalid argument for {audio_gcs_uri}. Check config (e.g., sample rate {STT_SAMPLE_RATE_HZ}Hz, audio format). Error: {e}. Duration: {tool_duration:.2f}s", exc_info=True)
        return json.dumps({"error": f"STT Invalid argument: {e}. Ensure audio format and sample rate match STT expectations."})
    except asyncio.TimeoutError: # Catch timeout from operation.result()
        tool_duration = time.monotonic() - tool_start_time
        logger.error(f"MCP TOOL: get_word_timestamps_from_audio FAILED. STT operation timed out for {audio_gcs_uri} after 300s. Duration: {tool_duration:.2f}s", exc_info=True)
        return json.dumps({"error": f"STT operation timed out for {audio_gcs_uri}"})
    except Exception as e:
        tool_duration = time.monotonic() - tool_start_time
        logger.error(f"MCP TOOL: get_word_timestamps_from_audio FAILED. Unexpected error during STT transcription for {audio_gcs_uri}: {e}. Duration: {tool_duration:.2f}s", exc_info=True)
        return json.dumps({"error": f"Failed to transcribe audio: {str(e)}"})


if __name__ == "__main__":
    logger.info("Starting Audio Processing MCP Server (TTS & STT) with enhanced logging...")
    # Example: you can further increase log level for specific libraries if needed for deep debugging
    # logging.getLogger('google.api_core.bidi').setLevel(logging.DEBUG)
    # logging.getLogger('google.auth.transport.requests').setLevel(logging.DEBUG)
    mcp.run(transport="stdio")