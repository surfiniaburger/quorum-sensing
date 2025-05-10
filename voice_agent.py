# voice_agent.py

import asyncio
import json
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from google import genai
from google.genai.types import (
    Content, Part, Tool, FunctionDeclaration, LiveConnectConfig,
    SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, FunctionResponse,
    RealtimeInputConfig, AutomaticActivityDetection
)


try:
    import main
except ImportError:
    logging.error("Failed to import 'process_voice_query_with_adk' from main. This is critical for voice agent.")
    # Define a fallback or raise an error to prevent app from running in a broken state
    async def process_voice_query_with_adk(session_id: str, user_query: str) -> str:
        return "Error: ADK processing function is not available."


# --- Configuration for Live API ---
LIVE_API_MODEL_ID = "gemini-2.0-flash-live-preview-04-09"
LIVE_API_VOICE_NAME = "Aoede"  # Choose your preferred voice
LIVE_API_LANGUAGE_CODE = "en-US"

# Create an APIRouter instance for this module
router = APIRouter()

# --- 1. Live API Tool Declaration (Root Agent as a Tool) ---
def declare_tools_for_live_api() -> list[Tool]:
    return [
        Tool(function_declarations=[
            FunctionDeclaration(
                name="invoke_adk_agent_system",
                description="Processes a user's query using the main ADK multi-agent system to get an answer or perform an action. Use this for any complex queries about cocktails, weather, MLB stats, or bookings.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "user_query": {"type_": "STRING", "description": "The user's full transcribed query."},
                        "session_id": {"type_": "STRING", "description": "The session ID for the ADK agent interaction, derived from the voice connection."}
                    },
                    "required": ["user_query", "session_id"]
                })
        ])
    ]

# --- 2. Live API Tool Dispatcher ---
async def dispatch_live_api_tool_call(function_name: str, args: dict, original_voice_session_id: str) -> dict:
    """
    Dispatches tool calls from the Live API.
    Now primarily calls the ADK agent system.
    """
    logging.info(f"Live API tool call: {function_name} with args: {args} for voice session: {original_voice_session_id}")
    if function_name == "invoke_adk_agent_system":
        user_query = args.get("user_query")
        # IMPORTANT: Use the original_voice_session_id as the session_id for the ADK system
        # to maintain context if the user switches between voice and text, or for consistent ADK logging.
        adk_session_id = original_voice_session_id # Use the voice session ID for the ADK call

        if not user_query: # session_id is now passed directly
            logging.error("User query missing in invoke_adk_agent_system call.")
            return {"error_message": "User query was not provided."}

        try:
            logging.info(f"Invoking ADK system for session '{adk_session_id}' with query: '{user_query}'")
            adk_response_text = await main.process_voice_query_with_adk(
                session_id=adk_session_id,
                user_query=user_query
            )
            logging.info(f"ADK system responded for session '{adk_session_id}': {adk_response_text}")
            # The Live API expects a JSON serializable object as the 'response'
            return {"processed_text_response": adk_response_text}
        except Exception as e:
            logging.error(f"Error invoking ADK agent system via process_voice_query_with_adk: {e}", exc_info=True)
            return {"error_message": f"Failed to process your query with the main assistant: {str(e)}"}
    else:
        logging.warning(f"Unknown function call by Live API: {function_name}")
        return {"error_message": f"The requested action '{function_name}' is not available through voice."}


# --- 3. Voice WebSocket Endpoint (largely the same, but calls new dispatcher) ---
@router.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str): # session_id here is from the URL
    await websocket.accept()
    logging.info(f"Voice client {session_id} connected to voice WebSocket endpoint.")

    genai_client = None
    live_session = None
    is_live_session_active = False

    try:
        genai_client = genai.Client()
        live_api_config = LiveConnectConfig(
            response_modalities=["AUDIO"], # We want audio output
            # Optional: Add output_audio_transcription={"min_utterance_split_duration_ms": 500} for Gemini text
            # Optional: Add input_audio_transcription={} for user text
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=LIVE_API_VOICE_NAME)
                ),
                language_code=LIVE_API_LANGUAGE_CODE
            ),
            realtime_input_config=RealtimeInputConfig(
                automatic_activity_detection=AutomaticActivityDetection()
            )
        )
        declared_tools = declare_tools_for_live_api()

        async with genai_client.aio.live.connect(
            model=LIVE_API_MODEL_ID,
            config=live_api_config,
            tools=declared_tools,
            # System instruction for the voice interaction itself (Gemini in Live API)
            system_instruction=Content(parts=[Part(text="You are a voice assistant. When asked to process a query, use the 'invoke_adk_agent_system' tool to get a detailed answer from the main assistant. Then, present that answer back to the user naturally in voice. If the main assistant provides an error, relay that information calmly.")])
        ) as session:
            live_session = session
            is_live_session_active = True
            logging.info(f"Voice client {session_id}: Live API session established with Gemini.")

            async def forward_from_live_api_to_client():
                nonlocal is_live_session_active
                try:
                    async for server_message in live_session.receive():
                        if not is_live_session_active: break

                        if server_message.server_content:
                            model_turn = server_message.server_content.model_turn
                            if model_turn and model_turn.parts:
                                for part in model_turn.parts:
                                    # Forward audio data
                                    if part.inline_data and part.inline_data.mime_type == f"audio/pcm;rate=24000":
                                        if websocket.client_state == WebSocketState.CONNECTED:
                                            await websocket.send_bytes(part.inline_data.data)
                                    # If you enabled output_audio_transcription in LiveConnectConfig:
                                    elif part.text: # This is the transcribed Gemini audio response
                                         if websocket.client_state == WebSocketState.CONNECTED:
                                            logging.info(f"Voice client {session_id}: Gemini audio transcript: {part.text}")
                                            await websocket.send_text(json.dumps({"type": "gemini_transcript", "text": part.text}))

                        elif server_message.tool_call:
                            logging.info(f"Voice client {session_id}: Live API tool_call: {server_message.tool_call.function_calls}")
                            responses_for_gemini = []
                            for func_call in server_message.tool_call.function_calls:
                                # Pass the original session_id from the voice websocket URL
                                tool_output_dict = await dispatch_live_api_tool_call(
                                    func_call.name, func_call.args, original_voice_session_id=session_id
                                )
                                responses_for_gemini.append(Part(
                                    function_response=FunctionResponse(name=func_call.name, response=tool_output_dict)
                                ))
                            if responses_for_gemini and is_live_session_active:
                                await live_session.send_client_content(
                                    turns=[Content(role="tool", parts=responses_for_gemini)]
                                )
                        # ... (go_away handling, etc. as before) ...
                        elif server_message.go_away:
                            logging.warning(f"Voice client {session_id}: Live API session ending (go_away received).")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.close(code=1000, reason="Live API session terminated by Google.")
                            return
                except WebSocketDisconnect: logging.info(f"Voice client {session_id} (Live API -> Client task) disconnected.")
                except Exception as e:
                    logging.error(f"Voice client {session_id}: Error in Live API -> Client task: {e}", exc_info=True)
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
                finally: is_live_session_active = False

            async def forward_from_client_to_live_api():
                nonlocal is_live_session_active
                try:
                    while is_live_session_active:
                        if websocket.client_state != WebSocketState.CONNECTED: break
                        audio_bytes = await websocket.receive_bytes()
                        if audio_bytes and is_live_session_active:
                            # Client MUST send audio in 16-bit PCM, 16kHz, little-endian format.
                            audio_part = Part(inline_data={"mime_type": "audio/pcm;rate=16000", "data": audio_bytes})
                            await live_session.send_client_content(turns=[Content(role="user", parts=[audio_part])])
                except WebSocketDisconnect: logging.info(f"Voice client {session_id} (Client -> Live API task) disconnected.")
                except Exception as e:
                    if isinstance(e, asyncio.CancelledError): logging.info(f"Voice client {session_id}: Client -> Live API task cancelled."); raise
                    logging.error(f"Voice client {session_id}: Error in Client -> Live API task: {e}", exc_info=True)
                finally: is_live_session_active = False
            
            # Run tasks (same as before)
            receive_task = asyncio.create_task(forward_from_live_api_to_client())
            send_task = asyncio.create_task(forward_from_client_to_live_api())
            done, pending = await asyncio.wait([receive_task, send_task], return_when=asyncio.FIRST_COMPLETED)
            is_live_session_active = False
            for task in pending: task.cancel()
            # ... (exception propagation from done tasks as before) ...
            for task in done:
                if task.exception() and not isinstance(task.exception(), asyncio.CancelledError):
                    logging.error(f"Voice client {session_id}: Task {task.get_name()} raised: {task.exception()}", exc_info=task.exception())
                    raise task.exception()

    # ... (Outer try-except-finally for WebSocket connection as before) ...
    except WebSocketDisconnect: logging.info(f"Voice client {session_id} main WebSocket handler disconnected.")
    except Exception as e: # ...
        logging.error(f"Voice client {session_id}: Unhandled error: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_text(json.dumps({"type": "error", "message": "Voice service error."}))
    finally: # ...
        is_live_session_active = False
        logging.info(f"Voice client {session_id} closing.")
        if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1000)
        logging.info(f"Voice client {session_id} fully closed.")
