# test_voice_client.py
import asyncio
import websockets
import json
import logging
import sounddevice as sd
import numpy as np
import queue # For thread-safe audio buffering for playback

# Configure basic logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceClient")

WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/voice/python-client-session-002" # Use a new session ID for testing
AUDIO_FILE_PATH = "london_weather_16k.pcm"  # Ensure this path is correct and file has data!

# Audio playback parameters (Gemini Live API sends 24kHz, 16-bit mono)
PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_CHANNELS = 1
PLAYBACK_DTYPE = 'int16' # Corresponds to 16-bit PCM

# Thread-safe queue for audio chunks to be played
audio_playback_queue = queue.Queue()
playback_finished_event = asyncio.Event() # To signal when playback might be done

def audio_callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
    """This is called (from a separate thread) by sounddevice whenever it needs more audio data."""
    if status:
        logger.warning(f"Sounddevice playback status: {status}")
    try:
        # Try to get a chunk from the queue without blocking indefinitely
        data_chunk = audio_playback_queue.get_nowait()
        chunk_len = len(data_chunk)
        if chunk_len < len(outdata):
            outdata[:chunk_len] = data_chunk
            outdata[chunk_len:] = 0 # Fill rest with silence
            logger.debug(f"Playback: Partial frame, silence padded. Got {chunk_len} samples.")
            # If this was the last chunk, we might want to signal completion, but
            # it's hard to know if it's truly the "last" from here.
        else:
            outdata[:] = data_chunk
            logger.debug(f"Playback: Full frame. Got {chunk_len} samples.")

    except queue.Empty:
        logger.debug("Playback: Queue empty, outputting silence.")
        outdata[:] = 0 # Fill with silence
        # Consider signaling that the queue is empty and playback might be done.
        # For continuous streaming, this might just mean a pause in incoming audio.
        # For now, we'll let it continue to output silence.
        # If we knew it was the absolute end, we could: playback_finished_event.set()

async def play_audio_from_queue():
    """Opens an output stream and plays audio from the queue."""
    loop = asyncio.get_event_loop()
    
    def run_playback():
        try:
            with sd.OutputStream(
                samplerate=PLAYBACK_SAMPLE_RATE,
                channels=PLAYBACK_CHANNELS,
                dtype=PLAYBACK_DTYPE,
                callback=audio_callback # Our callback function
            ):
                logger.info("Audio playback stream started. Playing silence until data arrives...")
                # Keep the stream alive. The callback will be continuously called.
                # We need a way to stop this when the conversation is over or WebSocket closes.
                # For this test client, it might run until manually stopped or WebSocket listener ends.
                while not playback_finished_event.is_set(): # Loop until signaled
                    sd.sleep(100) # Check event every 100ms
            logger.info("Audio playback stream finished.")
        except Exception as e:
            logger.error(f"Error in audio playback thread: {e}", exc_info=True)

    await loop.run_in_executor(None, run_playback) # Run synchronous sounddevice stream in a thread


async def send_audio_and_listen():
    logger.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URL}")
    
    # Start playback task in the background
    # playback_task = asyncio.create_task(play_audio_from_queue()) # Start playback immediately

    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            logger.info("Successfully connected to WebSocket.")
            
            # Start playback task only after successful connection
            playback_task = asyncio.create_task(play_audio_from_queue())


            # 1. Read the raw PCM audio data
            try:
                with open(AUDIO_FILE_PATH, "rb") as f:
                    audio_data_to_send = f.read()
                if not audio_data_to_send:
                    logger.error(f"Audio file '{AUDIO_FILE_PATH}' is empty. Aborting.")
                    playback_finished_event.set() # Signal playback to stop
                    if playback_task: await playback_task # Wait for playback task to finish
                    return
                logger.info(f"Successfully read {len(audio_data_to_send)} bytes from {AUDIO_FILE_PATH}.")
            except FileNotFoundError:
                logger.error(f"Audio file not found: {AUDIO_FILE_PATH}")
                playback_finished_event.set()
                if playback_task: await playback_task
                return
            except Exception as e:
                logger.error(f"Error reading audio file: {e}")
                playback_finished_event.set()
                if playback_task: await playback_task
                return

            # 2. Send the audio data
            try:
                logger.info("Sending audio data...")
                await websocket.send(audio_data_to_send)
                logger.info("Audio data sent successfully.")
            except Exception as e:
                logger.error(f"Error sending audio data: {e}")
                playback_finished_event.set()
                if playback_task: await playback_task
                return

            # 3. Listen for responses (both text and binary)
            logger.info("Listening for responses from server...")
            try:
                while True: # Keep listening until connection closes
                    message = await websocket.recv()
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            text_content = data.get("text")
                            if msg_type == "gemini_transcript":
                                logger.info(f"GEMINI TRANSCRIPT: {text_content}")
                            elif msg_type == "user_transcript":
                                logger.info(f"USER AUDIO TRANSCRIPT: {text_content}")
                            elif msg_type == "error":
                                logger.error(f"SERVER ERROR MESSAGE: {data.get('message')}")
                            else:
                                logger.info(f"Received TEXT: {message}")
                        except json.JSONDecodeError:
                            logger.info(f"Received non-JSON TEXT: {message}")
                    elif isinstance(message, bytes):
                        logger.info(f"Received BINARY (audio data): {len(message)} bytes.")
                        # Convert received bytes to NumPy array for sounddevice
                        # The audio from Gemini is 16-bit PCM, so 2 bytes per sample.
                        try:
                            audio_chunk_np = np.frombuffer(message, dtype=np.int16)
                            audio_playback_queue.put(audio_chunk_np)
                            logger.debug(f"Added {len(audio_chunk_np)} audio samples to playback queue.")
                        except Exception as e_audio_proc:
                            logger.error(f"Error processing received audio bytes: {e_audio_proc}")
                    else:
                        logger.info(f"Received other message type: {type(message)}")
            
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Server closed the WebSocket connection normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"WebSocket connection closed with error: {e}")
            except Exception as e: # Catch other errors during receive loop
                logger.error(f"Error while listening for messages: {e}", exc_info=True)

    except ConnectionRefusedError:
        logger.error(f"Connection refused. Ensure server is running at {WEBSOCKET_URL.split('/ws/')[0]}.")
    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {WEBSOCKET_URL}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Client shutting down. Signaling playback to stop.")
        playback_finished_event.set() # Signal playback thread to terminate
        if 'playback_task' in locals() and playback_task: # Check if task was created
            await playback_task # Wait for the playback task to finish
        logger.info("Client fully stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_listen())
    except KeyboardInterrupt:
        logger.info("Client interrupted by user (Ctrl+C).")
    finally:
        # Ensure the playback_finished_event is set if loop was broken by Ctrl+C before finally block in send_audio_and_listen
        if not playback_finished_event.is_set():
            playback_finished_event.set()
        logger.info("Exiting test client script.")