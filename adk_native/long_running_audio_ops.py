# adk_native/long_running_audio_ops.py
import logging
import json
import os
import io
import time
import tempfile
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, UTC
import hashlib
import shutil

from google.cloud import storage, texttospeech, speech # type: ignore
from google.api_core import exceptions as google_exceptions # type: ignore
from pydub import AudioSegment # type: ignore
from google.adk.tools.tool_context import ToolContext # type: ignore

# --- Configuration ---
GCP_PROJECT_ID_AUDIO = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCS_BUCKET_GENERATED_AUDIO = os.getenv("GCS_BUCKET_GENERATED_AUDIO", "mlb_generated_audio")
GCS_AUDIO_OUTPUT_PREFIX = "adk_native/audio/" # ADK native prefix

# TTS Config
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "en-US-Chirp3-HD-Puck")
TTS_VOICE_NAME_ALT = os.getenv("TTS_VOICE_NAME_ALT", "en-US-Chirp3-HD-Aoede")
AUDIO_ENCODING_TTS = texttospeech.AudioEncoding.MP3 # Explicitly for TTS
AUDIO_SILENCE_MS = int(os.getenv("AUDIO_SILENCE_MS", 350))
TTS_SAMPLE_RATE_HZ = int(os.getenv("STT_SAMPLE_RATE_HZ", 24000)) # Match STT expectation

# STT Config
STT_LANGUAGE_CODE = "en-US"
# STT_SAMPLE_RATE_HZ is already defined for TTS, ensure they match.

logger = logging.getLogger(__name__)

# --- Global Task Tracking ---
TTS_GENERATION_TASKS: Dict[str, Dict[str, Any]] = {}
STT_TRANSCRIPTION_TASKS: Dict[str, Dict[str, Any]] = {}

# --- Client Initialization ---
storage_client_audio_instance = None
tts_client_instance = None
stt_client_instance = None

def ensure_audio_clients_initialized():
    global storage_client_audio_instance, tts_client_instance, stt_client_instance
    if not storage_client_audio_instance:
        storage_client_audio_instance = storage.Client(project=GCP_PROJECT_ID_AUDIO)
    if not tts_client_instance:
        tts_client_instance = texttospeech.TextToSpeechClient()
    if not stt_client_instance:
        stt_client_instance = speech.SpeechClient()
    try:
        from pydub import AudioSegment # Check again in case it wasn't checked at module import
        logger.debug("Pydub AudioSegment confirmed available for audio ops.")
    except ImportError:
        logger.critical("CRITICAL: pydub library not found. Audio combination will fail.")


# --- Helper Functions (Adapted from MCP server) ---
async def _save_audio_to_gcs_native(audio_bytes: bytes, bucket_name: str, blob_name: str, timeout: int = 180) -> Optional[str]:
    if not storage_client_audio_instance:
        logger.error("Storage client not initialized for _save_audio_to_gcs_native.")
        return None
    try:
        bucket = storage_client_audio_instance.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(
            blob.upload_from_string,
            audio_bytes,
            content_type='audio/mpeg', # Assuming MP3
            timeout=timeout
        )
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved audio to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving audio to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

def _sync_generate_tts_segment(line: str, voice_name: str, output_path: str, segment_num: int) -> str:
    logger.debug(f"  TTS Segment Sync: Starting segment {segment_num} ('{line[:30]}...'), Voice: {voice_name}")
    if not tts_client_instance:
        raise RuntimeError("TTS client not available in _sync_generate_tts_segment.")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=line)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name) # Assuming en-US
        audio_config = texttospeech.AudioConfig(
            audio_encoding=AUDIO_ENCODING_TTS,
            sample_rate_hertz=TTS_SAMPLE_RATE_HZ
        )
        response = tts_client_instance.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        logger.debug(f"  TTS Segment Sync: Finished segment {segment_num}, saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"  TTS Segment Sync: Error in segment {segment_num} for voice {voice_name}: {e}", exc_info=True)
        raise

def _sync_combine_audio_segments(file_paths: List[str], silence_ms: int) -> Optional[bytes]:
    logger.debug(f"Pydub Combine Sync: Starting combination of {len(file_paths)} segments with {silence_ms}ms silence.")
    if not file_paths:
        logger.warning("Pydub Combine Sync: No audio file paths provided. Returning empty audio.")
        return AudioSegment.empty().export(format="mp3").read() # Return empty mp3 bytes

    try:
        # Start with silence if there are segments to process, otherwise AudioSegment.empty()
        full_audio = AudioSegment.silent(duration=silence_ms) if silence_ms > 0 else AudioSegment.empty()

        for i, file_path in enumerate(file_paths):
            segment = AudioSegment.from_mp3(file_path)
            full_audio += segment
            if i < len(file_paths) - 1:
                full_audio += AudioSegment.silent(duration=silence_ms)
        
        buffer = io.BytesIO()
        full_audio.export(buffer, format="mp3")
        logger.debug(f"Pydub Combine Sync: Exported combined audio, length {len(buffer.getvalue())} bytes.")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Pydub Combine Sync: Error combining audio segments: {e}", exc_info=True)
        return None

def _sync_cleanup_temp_dir(dir_path: str):
    logger.debug(f"Cleanup Sync: Attempting to remove directory tree: {dir_path}")
    try:
        shutil.rmtree(dir_path)
        logger.info(f"Cleanup Sync: Successfully removed temporary directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Cleanup Sync: Could not remove temp directory {dir_path}: {e}", exc_info=True)

# --- TTS Background Work ---
async def _perform_tts_generation_work(task_id: str, script: str, game_pk_str: str):
    ensure_audio_clients_initialized()
    TTS_GENERATION_TASKS[task_id]["status"] = "processing"
    gcs_audio_uri = None
    error_message = None
    temp_dir = None

    try:
        if not script:
            raise ValueError("Input script for TTS cannot be empty.")
        dialogue_lines = [line.strip() for line in script.splitlines() if line.strip()]
        if not dialogue_lines:
            raise ValueError("Script has no processable content after stripping.")

        temp_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="adk_tts_")
        logger.info(f"[Task {task_id}] TTS: Created temp dir {temp_dir}")

        segment_tasks = []
        for count, line in enumerate(dialogue_lines):
            voice_name = TTS_VOICE_NAME if count % 2 == 0 else TTS_VOICE_NAME_ALT
            temp_filename = os.path.join(temp_dir, f"part-{count}.mp3")
            segment_tasks.append(asyncio.to_thread(_sync_generate_tts_segment, line, voice_name, temp_filename, count + 1))
        
        segment_paths_results = await asyncio.gather(*segment_tasks, return_exceptions=True)
        
        temp_audio_files = []
        for i, result in enumerate(segment_paths_results):
            if isinstance(result, str) and os.path.exists(result):
                temp_audio_files.append(result)
            else:
                # Log the specific error for the segment if it's an exception
                logger.error(f"[Task {task_id}] TTS: Segment {i+1} generation failed. Error: {result if isinstance(result, Exception) else 'Unknown error'}")
                # Decide if one failed segment should fail the whole task or continue with available ones
        
        if not temp_audio_files:
            raise ValueError("No audio segments were successfully generated by TTS.")

        combined_audio_bytes = await asyncio.to_thread(_sync_combine_audio_segments, temp_audio_files, AUDIO_SILENCE_MS)
        if not combined_audio_bytes:
            raise ValueError("Failed to combine audio segments or combined audio is empty.")

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        blob_name = f"{GCS_AUDIO_OUTPUT_PREFIX}game_{game_pk_str}/dialogue_audio_{timestamp}.mp3"
        gcs_audio_uri = await _save_audio_to_gcs_native(combined_audio_bytes, GCS_BUCKET_GENERATED_AUDIO, blob_name)
        if not gcs_audio_uri:
            raise ValueError("Failed to upload combined audio to GCS.")

        TTS_GENERATION_TASKS[task_id].update({
            "status": "completed",
            "result": json.dumps({"audio_uri": gcs_audio_uri}) # JSON string as result
        })
    except Exception as e:
        logger.error(f"[Task {task_id}] TTS generation failed: {e}", exc_info=True)
        error_message = str(e)
        TTS_GENERATION_TASKS[task_id].update({
            "status": "failed",
            "error": error_message
        })
    finally:
        if temp_dir:
            await asyncio.to_thread(_sync_cleanup_temp_dir, temp_dir)
    logger.info(f"[Task {task_id}] TTS generation work finished. Status: {TTS_GENERATION_TASKS[task_id]['status']}")


# --- STT Background Work ---
async def _perform_stt_transcription_work(task_id: str, audio_gcs_uri: str):
    ensure_audio_clients_initialized()
    STT_TRANSCRIPTION_TASKS[task_id]["status"] = "processing"
    word_timestamps: List[Dict[str, Any]] = []
    error_message = None

    try:
        if not audio_gcs_uri or not audio_gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI for STT: '{audio_gcs_uri}'. Must start with 'gs://'.")

        config = speech.RecognitionConfig(
            sample_rate_hertz=TTS_SAMPLE_RATE_HZ, # Use the TTS sample rate
            language_code=STT_LANGUAGE_CODE,
            enable_word_time_offsets=True,
        )
        audio = speech.RecognitionAudio(uri=audio_gcs_uri)
        logger.info(f"[Task {task_id}] STT: Calling long_running_recognize for {audio_gcs_uri}")

        operation = await asyncio.to_thread(stt_client_instance.long_running_recognize, config=config, audio=audio) # type: ignore
        logger.info(f"[Task {task_id}] STT: Operation started. Waiting for result (timeout 300s)...")
        response = await asyncio.to_thread(operation.result, timeout=300)

        for result_idx, result in enumerate(response.results):
            if result.alternatives:
                alternative = result.alternatives[0]
                for word_info in alternative.words:
                    word_timestamps.append({
                        "word": word_info.word,
                        "start_time_s": word_info.start_time.total_seconds(),
                        "end_time_s": word_info.end_time.total_seconds(),
                    })
        if not word_timestamps:
             logger.warning(f"[Task {task_id}] STT: No words returned from transcription of {audio_gcs_uri}")


        STT_TRANSCRIPTION_TASKS[task_id].update({
            "status": "completed",
            "result": json.dumps(word_timestamps) # JSON string as result
        })

    except Exception as e:
        logger.error(f"[Task {task_id}] STT transcription failed for {audio_gcs_uri}: {e}", exc_info=True)
        error_message = str(e)
        STT_TRANSCRIPTION_TASKS[task_id].update({
            "status": "failed",
            "error": error_message
        })
    logger.info(f"[Task {task_id}] STT transcription work finished. Status: {STT_TRANSCRIPTION_TASKS[task_id]['status']}")


# --- Initiator Functions for LongRunningFunctionTool ---
def initiate_tts_generation(script: str, game_pk_str: str = "unknown_game", tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    ensure_audio_clients_initialized()
    if not script:
        return {"status": "error", "message": "TTS script cannot be empty."}

    task_id = f"tts_task_{hashlib.md5(script[:50].encode() + str(time.time()).encode()).hexdigest()[:12]}"
    agent_name = tool_context.agent_name if tool_context and hasattr(tool_context, 'agent_name') else "UnknownAgent"
    logger.info(f"LRFT:init_tts (Agent: {agent_name}): Initiating task {task_id} for game {game_pk_str}")

    TTS_GENERATION_TASKS[task_id] = {"status": "submitted", "start_time": time.time()}
    asyncio.create_task(_perform_tts_generation_work(task_id, script, game_pk_str))

    return {
        "status": "pending_agent_client_action",
        "task_id": task_id,
        "tool_name": "initiate_tts_generation",
        "message": f"TTS generation task {task_id} initiated. Awaiting client polling."
    }

def initiate_stt_transcription(audio_gcs_uri: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    ensure_audio_clients_initialized()
    if not audio_gcs_uri or not audio_gcs_uri.startswith("gs://"):
         return {"status": "error", "message": f"Invalid GCS URI for STT: '{audio_gcs_uri}'. Must start with 'gs://'."}

    task_id = f"stt_task_{hashlib.md5(audio_gcs_uri.encode() + str(time.time()).encode()).hexdigest()[:12]}"
    agent_name = tool_context.agent_name if tool_context and hasattr(tool_context, 'agent_name') else "UnknownAgent"
    logger.info(f"LRFT:init_stt (Agent: {agent_name}): Initiating task {task_id} for URI {audio_gcs_uri}")

    STT_TRANSCRIPTION_TASKS[task_id] = {"status": "submitted", "start_time": time.time()}
    asyncio.create_task(_perform_stt_transcription_work(task_id, audio_gcs_uri))

    return {
        "status": "pending_agent_client_action",
        "task_id": task_id,
        "tool_name": "initiate_stt_transcription",
        "message": f"STT transcription task {task_id} initiated. Awaiting client polling."
    }