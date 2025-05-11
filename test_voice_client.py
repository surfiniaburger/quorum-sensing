# test_voice_client.py
import asyncio
import websockets
import json
import logging

# Configure basic logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/voice/python-client-session-001" # Use a unique session ID
AUDIO_FILE_PATH = "london_weather_16k.pcm" # Ensure this path is correct

async def send_audio_and_listen():
    logging.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URL}")
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            logging.info("Successfully connected to WebSocket.")

            # 1. Read the raw PCM audio data
            try:
                with open(AUDIO_FILE_PATH, "rb") as f:
                    audio_data = f.read()
                logging.info(f"Successfully read {len(audio_data)} bytes from {AUDIO_FILE_PATH}.")
            except FileNotFoundError:
                logging.error(f"Audio file not found: {AUDIO_FILE_PATH}")
                return
            except Exception as e:
                logging.error(f"Error reading audio file: {e}")
                return

            # 2. Send the audio data
            try:
                logging.info("Sending audio data...")
                await websocket.send(audio_data)
                logging.info("Audio data sent successfully.")
            except Exception as e:
                logging.error(f"Error sending audio data: {e}")
                return

            # 3. Listen for responses (both text and binary)
            logging.info("Listening for responses from server...")
            try:
                while True:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            if data.get("type") == "gemini_transcript":
                                logging.info(f"Received GEMINI TRANSCRIPT: {data.get('text')}")
                            elif data.get("type") == "user_transcript": # If you enabled input transcription
                                logging.info(f"Received USER AUDIO TRANSCRIPT: {data.get('text')}")
                            elif data.get("type") == "error":
                                logging.error(f"Received SERVER ERROR: {data.get('message')}")
                            else:
                                logging.info(f"Received TEXT message: {message}")
                        except json.JSONDecodeError:
                            logging.info(f"Received non-JSON TEXT message: {message}")
                    elif isinstance(message, bytes):
                        logging.info(f"Received BINARY message (audio data): {len(message)} bytes.")
                        # Here, you would typically buffer and play this audio.
                        # For this test, we're just logging the receipt.
                    else:
                        logging.info(f"Received other message type: {type(message)}")

            except websockets.exceptions.ConnectionClosedOK:
                logging.info("Server closed the connection normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logging.error(f"Server closed the connection with error: {e}")
            except Exception as e:
                logging.error(f"Error while listening for messages: {e}")

    except ConnectionRefusedError:
        logging.error(f"Connection refused. Ensure the FastAPI server is running at {WEBSOCKET_URL.split('/ws/')[0]} and the endpoint is correct.")
    except websockets.exceptions.InvalidURI:
        logging.error(f"Invalid WebSocket URI: {WEBSOCKET_URL}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(send_audio_and_listen())