# voice_agent.py
from typing import AsyncGenerator, Literal
import websockets
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from google import genai
from google.genai.types import (
    Content, Part, Tool, FunctionDeclaration, LiveConnectConfig,
    SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, FunctionResponse,
    RealtimeInputConfig, AutomaticActivityDetection, AudioTranscriptionConfig
)
#from google.genai.types.generation_types import BlockedPromptException # For specific error handling

# Attempt to import the ADK processing function
try:
    import main
except ImportError:
    logging.error("CRITICAL: Failed to import 'process_voice_query_with_adk' from main module.")
    # Define a fallback function if import fails, to prevent app crash on load
    async def process_voice_query_with_adk(session_id: str, user_query: str) -> str:
        logging.error("Fallback process_voice_query_with_adk called due to import error.")
        return "Error: Core ADK processing function (process_voice_query_with_adk) is unavailable."

# --- Configuration for Live API ---
LIVE_API_MODEL_ID = "gemini-2.0-flash-live-preview-04-09"
LIVE_API_VOICE_NAME = "Aoede"  # Or your preferred voice from the Gradio example list
# LIVE_API_LANGUAGE_CODE is not used directly in SpeechConfig based on Gradio example

router = APIRouter()

# --- Tool Declaration for Live API ---
# voice_agent.py

def declare_tools_for_live_api() -> list[Tool]:
    return [
        Tool(function_declarations=[
            FunctionDeclaration(
                name="invoke_adk_agent_system",
                description="Processes a user's query using the main ADK multi-agent system to get an answer or perform an action. Use this for any complex queries about cocktails, weather, MLB stats, or bookings.",
                parameters={
                    # --- MODIFIED HERE ---
                    "type": "OBJECT",  # Changed from "type_"
                    # --- END MODIFICATION ---
                    "properties": {
                        "user_query": {
                            # --- MODIFIED HERE ---
                            "type": "STRING", # Changed from "type_"
                            # --- END MODIFICATION ---
                            "description": "The user's full transcribed query from their speech."
                        }
                        # If you had other parameters, you'd change "type_" to "type" for them too.
                    },
                    "required": ["user_query"]
                })
        ])
    ]


# --- Tool Dispatcher ---
async def dispatch_live_api_tool_call(function_name: str, args: dict, original_voice_session_id: str) -> dict:
    logging.info(f"Live API tool call dispatch: '{function_name}' with args: {args} for voice session: '{original_voice_session_id}'")
    if function_name == "invoke_adk_agent_system":
        user_query = args.get("user_query")
        adk_session_id = original_voice_session_id # Use the voice WebSocket session_id for ADK

        if not user_query:
            logging.error("User query missing in invoke_adk_agent_system call.")
            return {"error_message": "User query was not provided to ADK system."}
        try:
            logging.info(f"Invoking ADK system (process_voice_query_with_adk) for session '{adk_session_id}' with query: '{user_query}'")
            adk_response_text = await main.process_voice_query_with_adk(
                session_id=adk_session_id, user_query=user_query
            )
            logging.info(f"ADK system responded for session '{adk_session_id}': '{adk_response_text}'")
            # This dictionary will become the 'response' field in the FunctionResponse
            return {"processed_text_response": adk_response_text}
        except Exception as e:
            logging.error(f"Error invoking ADK system via process_voice_query_with_adk: {e}", exc_info=True)
            return {"error_message": f"Failed to process your query with the main assistant: {str(e)}"}
    else:
        logging.warning(f"Unknown function call by Live API: {function_name}")
        return {"error_message": f"The requested action '{function_name}' is not available through voice."}


# --- Voice WebSocket Endpoint ---
@router.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logging.info(f"Voice client '{session_id}' connected to WebSocket endpoint.")

    genai_client = None
    live_session_obj = None
    is_live_session_active = True

    try:
        genai_client = genai.Client()

        declared_tools = declare_tools_for_live_api()

        # Construct config objects based on actual SDK signatures from your environment
        speech_config_instance = SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=LIVE_API_VOICE_NAME)
            )
            # No language_code here as per your SDK's SpeechConfig signature
        )

        # AudioTranscriptionConfig takes no arguments for default enablement per your SDK's signature
        output_transcription_config_instance = AudioTranscriptionConfig()
        # input_transcription_config_instance = AudioTranscriptionConfig() # Optional for user speech

        live_api_config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config_instance,
            tools=declared_tools, # Tools are part of LiveConnectConfig
            output_audio_transcription=output_transcription_config_instance,
            # input_audio_transcription=input_transcription_config_instance, # Optional
            # OMITTING realtime_input_config as it's not a direct field of LiveConnectConfig in your SDK.
            # We will rely on API defaults for input audio handling (like VAD) first.
            # If VAD is an issue, we'd need to find where RealtimeInputConfig is applied in your SDK version
            # (e.g., on the session object after connect, or on individual send_client_content calls if supported).
            system_instruction=Content(parts=[Part.from_text(
                text="You are a voice assistant. When a user speaks, transcribe their query. Then, use the 'invoke_adk_agent_system' tool to get a detailed answer from the main assistant using the transcribed query. Finally, present that answer back to the user naturally in voice. If the main assistant provides an error, relay that information calmly."
            )])
        )
        logging.info(f"Voice client '{session_id}': Constructed LiveConnectConfig: {live_api_config.to_dict() if hasattr(live_api_config, 'to_dict') else 'Cannot print config dict'}")

        logging.info(f"Voice client '{session_id}': Attempting genai_client.aio.live.connect...")
        async with genai_client.aio.live.connect(
            model=LIVE_API_MODEL_ID,
            config=live_api_config # Config now contains tools, speech_config, etc.
            # No 'tools=' argument here directly
        ) as session:
            live_session_obj = session
            logging.info(f"Voice client '{session_id}': Live API session established successfully with Gemini!")

            async def client_audio_stream_generator() -> AsyncGenerator[bytes, None]:
                nonlocal is_live_session_active
                try:
                    while is_live_session_active:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            logging.info(f"Voice client '{session_id}': WebSocket disconnected in audio generator.")
                            is_live_session_active = False; break
                        audio_bytes = await websocket.receive_bytes()
                        if not is_live_session_active: break
                        if audio_bytes:
                            logging.debug(f"Voice client '{session_id}': Yielding {len(audio_bytes)} audio bytes to Live API stream.")
                            yield audio_bytes
                        else:
                            logging.debug(f"Voice client '{session_id}': Received empty audio message from client.")
                except WebSocketDisconnect:
                    logging.info(f"Voice client '{session_id}': WebSocket disconnected during client audio streaming generator.")
                except Exception as e:
                    logging.error(f"Voice client '{session_id}': Error in client_audio_stream_generator: {e}", exc_info=True)
                finally:
                    is_live_session_active = False
                    logging.info(f"Voice client '{session_id}': Client audio stream generator finished.")

            logging.info(f"Voice client '{session_id}': Starting Live API stream processing using session.start_stream().")
            async for server_message in session.start_stream(
                stream=client_audio_stream_generator(),
                mime_type="audio/pcm;rate=16000"
            ):
                if not is_live_session_active: break

                if server_message.server_content:
                    model_turn = server_message.server_content.model_turn
                    if model_turn and model_turn.parts:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.mime_type == f"audio/pcm;rate=24000":
                                if websocket.client_state == WebSocketState.CONNECTED:
                                    await websocket.send_bytes(part.inline_data.data)
                            elif part.text:
                                 if websocket.client_state == WebSocketState.CONNECTED:
                                    logging.info(f"Voice client '{session_id}': Live API Gemini audio transcript: '{part.text}'")
                                    await websocket.send_text(json.dumps({"type": "gemini_transcript", "text": part.text}))
                
                elif server_message.tool_call:
                    logging.info(f"Voice client '{session_id}': Live API requesting tool_call: {server_message.tool_call.function_calls}")
                    responses_for_gemini = []
                    for func_call in server_message.tool_call.function_calls:
                        if func_call.name == "invoke_adk_agent_system":
                            tool_output_dict = await dispatch_live_api_tool_call(
                                func_call.name, func_call.args, original_voice_session_id=session_id
                            )
                            responses_for_gemini.append(Part(
                                function_response=FunctionResponse(name=func_call.name, response=tool_output_dict)
                            ))
                        else:
                            logging.warning(f"Voice client '{session_id}': Live API called unknown tool '{func_call.name}'")
                            responses_for_gemini.append(Part(
                                function_response=FunctionResponse(name=func_call.name, response={"error_message": "Unknown tool called by Live API model."})
                            ))
                    
                    if responses_for_gemini and is_live_session_active and live_session_obj:
                        try:
                            # Use the stored live_session_obj to send tool responses
                            await live_session_obj.send_client_content(turns=[Content(role="tool", parts=responses_for_gemini)])
                            logging.info(f"Voice client '{session_id}': Sent tool response(s) to Live API via send_client_content.")
                        except Exception as e_send_tool_resp:
                            logging.error(f"Voice client '{session_id}': Error sending tool response to Live API: {e_send_tool_resp}", exc_info=True)
                
                elif server_message.go_away:
                    logging.warning(f"Voice client '{session_id}': Live API session ending (go_away received).")
                    is_live_session_active = False
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(code=1000, reason="Live API session ended by Google.")
                    break 

            logging.info(f"Voice client '{session_id}': Live API stream processing loop finished.")

    except websockets.exceptions.ConnectionClosedError as cce:
        logging.info(f"Voice client '{session_id}': WebSocket connection closed by client/network: {cce}")
    except WebSocketDisconnect:
        logging.info(f"Voice client '{session_id}': FastAPI WebSocket disconnected.")
    except Exception as e:
        logging.error(f"Voice client '{session_id}': Unhandled error in voice WebSocket endpoint: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "message": "Voice service encountered an internal error."}))
    finally:
        is_live_session_active = False
        logging.info(f"Voice client '{session_id}': Initiating final cleanup procedures.")
        if live_session_obj: # Check if it was assigned
             logging.info(f"Voice client '{session_id}': Live API session for model '{LIVE_API_MODEL_ID}' will be closed by context manager.")
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
            except RuntimeError as e_ws_close:
                logging.warning(f"Voice client '{session_id}': Minor error during WebSocket close: {e_ws_close}")
        logging.info(f"Voice client '{session_id}': WebSocket fully closed.")