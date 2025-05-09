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
    RealtimeInputConfig, AutomaticActivityDetectionConfig
)
from google.genai.types.generation_types import BlockedPromptException

# --- Configuration for Live API ---
LIVE_API_MODEL_ID = "gemini-2.0-flash-live-preview-04-09"
LIVE_API_VOICE_NAME = "Aoede"  # Choose your preferred voice
LIVE_API_LANGUAGE_CODE = "en-US"

# Create an APIRouter instance for this module
router = APIRouter()

# --- Placeholder imports for refactored MCP tool logic ---
# You will need to implement these refactored functions in your MCP server files
# and ensure these imports work correctly.

# from mcp_server.cocktail import (
#     search_cocktail_by_name_logic,
#     list_cocktails_by_first_letter_logic,
#     search_ingredient_by_name_logic,
#     list_random_cocktails_logic,
#     lookup_cocktail_details_by_id_logic
# )
# from mcp_server.weather_server import (
#     get_forecast_by_city_logic,
#     get_forecast_logic, # For coordinates
#     get_alerts_logic
# )
# from mcp_server.mlb_stats_server import (
#     get_live_game_score_logic,
#     get_game_play_by_play_summary_logic,
#     get_player_stats_for_game_logic,
#     get_team_schedule_logic,
#     get_league_standings_logic,
#     get_team_roster_logic
# )

# --- 1. MCP Tool Declaration for Live API Function Calling ---
def declare_mcp_tools_for_live_api() -> list[Tool]:
    """
    Defines the structure of your MCP tools for the Live API's function calling.
    """
    declared_tools = [
        # === Cocktail Tools ===
        Tool(function_declarations=[
            FunctionDeclaration(
                name="search_cocktail_by_name",
                description="Searches for cocktails by name.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"name": {"type_": "STRING", "description": "The name of the cocktail (e.g., margarita)"}},
                    "required": ["name"]
                }),
            FunctionDeclaration(
                name="list_cocktails_by_first_letter",
                description="Lists all cocktails starting with a specific letter.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"letter": {"type_": "STRING", "description": "The first letter (single character)"}},
                    "required": ["letter"]
                }),
            FunctionDeclaration(
                name="search_ingredient_by_name",
                description="Searches for an ingredient by its name.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"name": {"type_": "STRING", "description": "The name of the ingredient (e.g., vodka)"}},
                    "required": ["name"]
                }),
            FunctionDeclaration(
                name="list_random_cocktails",
                description="Looks up a single random cocktail. Takes no arguments.",
                parameters={"type_": "OBJECT", "properties": {}}
            ),
            FunctionDeclaration(
                name="lookup_cocktail_details_by_id",
                description="Looks up the full details of a specific cocktail by its ID.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"cocktail_id": {"type_": "STRING", "description": "The unique ID of the cocktail"}},
                    "required": ["cocktail_id"]
                }),
        ]),
        # === Weather Tools ===
        Tool(function_declarations=[
            FunctionDeclaration(
                name="get_forecast_by_city",
                description="Get the weather forecast for a specific US city and state.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "city": {"type_": "STRING", "description": "The name of the city"},
                        "state": {"type_": "STRING", "description": "The two-letter US state code (e.g., CA, NY)"}
                    },
                    "required": ["city", "state"]
                }),
            FunctionDeclaration(
                name="get_forecast",
                description="Get the weather forecast for a specific location using latitude and longitude.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "latitude": {"type_": "NUMBER", "description": "The latitude of the location"},
                        "longitude": {"type_": "NUMBER", "description": "The longitude of the location"}
                    },
                    "required": ["latitude", "longitude"]
                }),
            FunctionDeclaration(
                name="get_alerts",
                description="Get active weather alerts for a specific US state.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"state": {"type_": "STRING", "description": "The two-letter US state code (e.g., CA, NY, TX)"}},
                    "required": ["state"]
                }),
        ]),
        # === MLB Stats Tools ===
        Tool(function_declarations=[
            FunctionDeclaration(
                name="get_live_game_score",
                description="Retrieves the live score and basic status for a given MLB game ID (gamePk).",
                parameters={
                    "type_": "OBJECT",
                    "properties": {"game_pk": {"type_": "INTEGER", "description": "The unique integer identifier for the MLB game."}},
                    "required": ["game_pk"]
                }),
            FunctionDeclaration(
                name="get_game_play_by_play_summary",
                description="Retrieves a summary of the last few plays for a given MLB game ID (gamePk).",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "game_pk": {"type_": "INTEGER", "description": "The unique integer identifier for the MLB game."},
                        "count": {"type_": "INTEGER", "description": "Number of recent plays to summarize (e.g., 3). Optional, defaults to 3."}
                    },
                    "required": ["game_pk"]
                }),
            FunctionDeclaration(
                name="get_player_stats_for_game",
                description="Retrieves a specific player's stats for a given game ID (gamePk) and player ID.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "game_pk": {"type_": "INTEGER", "description": "The unique integer identifier for the MLB game."},
                        "player_id": {"type_": "INTEGER", "description": "The MLB official integer ID for the player."}
                    },
                    "required": ["game_pk", "player_id"]
                }),
            FunctionDeclaration(
                name="get_team_schedule",
                description="Retrieves a team's game schedule. Provide team name or team ID.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "team_identifier": {"type_": "STRING", "description": "Team name (e.g., 'Dodgers') or team ID as a string (e.g., '119')."},
                        "days_range": {"type_": "INTEGER", "description": "Number of days from today (e.g., 7 for next 7 days, -3 for last 3 days). Optional, defaults to 7."}
                    },
                    "required": ["team_identifier"]
                }),
            FunctionDeclaration(
                name="get_league_standings",
                description="Retrieves standings for a given MLB league (AL or NL) and season year.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "league_id": {"type_": "INTEGER", "description": "League ID (103 for AL, 104 for NL)."},
                        "season": {"type_": "INTEGER", "description": "The season year (e.g., 2024)."}
                    },
                    "required": ["league_id", "season"]
                }),
            FunctionDeclaration(
                name="get_team_roster",
                description="Retrieves a team's roster. Provide team name or team ID, optionally season and roster type.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "team_identifier": {"type_": "STRING", "description": "Team name (e.g., 'Dodgers') or team ID as a string (e.g., '119')."},
                        "season": {"type_": "INTEGER", "description": "Season year (e.g., 2024). Optional, defaults to current year."},
                        "roster_type": {"type_": "STRING", "description": "Roster type (e.g., '40Man', 'active'). Optional, defaults to API default."}
                    },
                    "required": ["team_identifier"]
                }),
        ]),
        # === Airbnb Tools (If you decide to proxy them or Gemini can call external URLs) ===
        # Tool(function_declarations=[
        #     FunctionDeclaration(name="search_airbnb_listings", ...),
        #     FunctionDeclaration(name="get_airbnb_listing_details", ...),
        # ]),
    ]
    return declared_tools

# --- 2. MCP Tool Execution Logic ---
async def dispatch_mcp_tool_call(function_name: str, args: dict) -> dict:
    """
    Executes the appropriate MCP tool based on the function_name and arguments.
    Returns a dictionary suitable for the FunctionResponse.
    """
    logging.info(f"Attempting to dispatch tool call: {function_name} with args: {args}")
    tool_result_text = "" # Your *_logic functions should return strings
    try:
        # --- Cocktail Tools ---
        if function_name == "search_cocktail_by_name":
            # tool_result_text = await search_cocktail_by_name_logic(args.get("name"))
            tool_result_text = f"Mock Cocktail: Searched for '{args.get('name')}'."
        elif function_name == "list_cocktails_by_first_letter":
            # tool_result_text = await list_cocktails_by_first_letter_logic(args.get("letter"))
            tool_result_text = f"Mock Cocktail: Listed for letter '{args.get('letter')}'."
        elif function_name == "search_ingredient_by_name":
            # tool_result_text = await search_ingredient_by_name_logic(args.get("name"))
            tool_result_text = f"Mock Cocktail: Searched ingredient '{args.get('name')}'."
        elif function_name == "list_random_cocktails":
            # tool_result_text = await list_random_cocktails_logic()
            tool_result_text = "Mock Cocktail: Here is The Mockarita."
        elif function_name == "lookup_cocktail_details_by_id":
            # tool_result_text = await lookup_cocktail_details_by_id_logic(args.get("cocktail_id"))
            tool_result_text = f"Mock Cocktail: Details for ID '{args.get('cocktail_id')}'."

        # --- Weather Tools ---
        elif function_name == "get_forecast_by_city":
            # tool_result_text = await get_forecast_by_city_logic(city=args.get("city"), state=args.get("state"))
            tool_result_text = f"Mock Weather: Forecast for {args.get('city')}, {args.get('state')} is sunny."
        elif function_name == "get_forecast": # by coordinates
            # tool_result_text = await get_forecast_logic(latitude=args.get("latitude"), longitude=args.get("longitude"))
            tool_result_text = f"Mock Weather: Forecast for lat {args.get('latitude')}, lon {args.get('longitude')} is cloudy."
        elif function_name == "get_alerts":
            # tool_result_text = await get_alerts_logic(args.get("state"))
            tool_result_text = f"Mock Weather: Alerts for state '{args.get('state')}' are none."

        # --- MLB Stats Tools ---
        elif function_name == "get_live_game_score":
            # tool_result_text = await get_live_game_score_logic(game_pk=args.get("game_pk"))
            tool_result_text = f"Mock MLB: Live score for game {args.get('game_pk')} is 3-2."
        elif function_name == "get_game_play_by_play_summary":
            # tool_result_text = await get_game_play_by_play_summary_logic(game_pk=args.get("game_pk"), count=args.get("count", 3))
            tool_result_text = f"Mock MLB: Plays for game {args.get('game_pk')} are exciting."
        elif function_name == "get_player_stats_for_game":
            # tool_result_text = await get_player_stats_for_game_logic(game_pk=args.get("game_pk"), player_id=args.get("player_id"))
            tool_result_text = f"Mock MLB: Stats for player {args.get('player_id')} in game {args.get('game_pk')} are great."
        elif function_name == "get_team_schedule":
            # tool_result_text = await get_team_schedule_logic(team_identifier=args.get("team_identifier"), days_range=args.get("days_range", 7))
            tool_result_text = f"Mock MLB: Schedule for team '{args.get('team_identifier')}' is busy."
        elif function_name == "get_league_standings":
            # tool_result_text = await get_league_standings_logic(league_id=args.get("league_id"), season=args.get("season"))
            tool_result_text = f"Mock MLB: Standings for league {args.get('league_id')}, season {args.get('season')} are tight."
        elif function_name == "get_team_roster":
            # tool_result_text = await get_team_roster_logic(team_identifier=args.get("team_identifier"), season=args.get("season", 0), roster_type=args.get("roster_type", ""))
            tool_result_text = f"Mock MLB: Roster for team '{args.get('team_identifier')}' is strong."

        else:
            logging.warning(f"Unknown function call requested by Live API: {function_name}")
            return {"error": f"Tool '{function_name}' is not implemented or recognized by the voice agent dispatch."}

        # Gemini expects the 'response' in FunctionResponse to be a JSON serializable object.
        # If your tools return simple strings as designed, wrap them.
        return {"result_text": tool_result_text} # The model will then use this text in its response.

    except Exception as e:
        logging.error(f"Error executing tool '{function_name}': {e}", exc_info=True)
        # Provide a structured error back to Gemini
        return {"error": f"Execution of tool '{function_name}' failed: {str(e)}"}

# --- 3. New Voice WebSocket Endpoint ---
@router.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logging.info(f"Voice client {session_id} connected to voice WebSocket endpoint.")

    genai_client = None
    live_session = None
    is_live_session_active = False # Local flag to control task loops

    try:
        genai_client = genai.Client() # Will use GOOGLE_GENAI_USE_VERTEXAI, etc. from env

        live_api_config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=LIVE_API_VOICE_NAME)
                ),
                language_code=LIVE_API_LANGUAGE_CODE
            ),
            realtime_input_config=RealtimeInputConfig(
                automatic_activity_detection=AutomaticActivityDetectionConfig()
            )
            # Optional: Add output_audio_transcription={} or input_audio_transcription={} here
        )

        declared_mcp_tools = declare_mcp_tools_for_live_api()

        async with genai_client.aio.live.connect(
            model=LIVE_API_MODEL_ID,
            config=live_api_config,
            tools=declared_mcp_tools if declared_mcp_tools else None,
            # system_instruction=Content(parts=[Part(text="You are a helpful voice assistant. Keep your responses concise.")]) # Optional
        ) as session:
            live_session = session
            is_live_session_active = True
            logging.info(f"Voice client {session_id}: Live API session established with Gemini.")

            async def forward_audio_from_live_api_to_client():
                nonlocal is_live_session_active
                try:
                    async for server_message in live_session.receive():
                        if not is_live_session_active: break

                        if server_message.server_content:
                            model_turn = server_message.server_content.model_turn
                            if model_turn and model_turn.parts:
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.mime_type == f"audio/pcm;rate=24000":
                                        if websocket.client_state == WebSocketState.CONNECTED:
                                            await websocket.send_bytes(part.inline_data.data)
                                    # Handle text transcriptions if enabled
                                    # elif part.text and live_api_config.output_audio_transcription:
                                    #     if websocket.client_state == WebSocketState.CONNECTED:
                                    #         await websocket.send_text(json.dumps({"type": "transcript_gemini", "text": part.text}))

                        elif server_message.tool_call:
                            logging.info(f"Voice client {session_id}: Live API requested tool_call: {server_message.tool_call.function_calls}")
                            responses_for_gemini = []
                            for func_call in server_message.tool_call.function_calls:
                                tool_output_dict = await dispatch_mcp_tool_call(func_call.name, func_call.args)
                                responses_for_gemini.append(Part(
                                    function_response=FunctionResponse(
                                        name=func_call.name,
                                        response=tool_output_dict # Must be JSON-serializable
                                    )
                                ))
                            if responses_for_gemini and is_live_session_active:
                                await live_session.send_client_content(
                                    turns=[Content(role="tool", parts=responses_for_gemini)]
                                )

                        elif server_message.go_away:
                            logging.warning(f"Voice client {session_id}: Live API session ending (go_away received).")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.close(code=1000, reason="Live API session terminated by Google.")
                            return
                except WebSocketDisconnect:
                    logging.info(f"Voice client {session_id} (Live API -> Client task) disconnected.")
                except Exception as e:
                    logging.error(f"Voice client {session_id}: Error in Live API -> Client task: {e}", exc_info=True)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(code=1011, reason="Server error processing Live API response")
                finally:
                    is_live_session_active = False

            async def forward_audio_from_client_to_live_api():
                nonlocal is_live_session_active
                try:
                    while is_live_session_active:
                        if websocket.client_state != WebSocketState.CONNECTED: break
                        audio_bytes_from_client = await websocket.receive_bytes()
                        if audio_bytes_from_client and is_live_session_active:
                            audio_part_for_live_api = Part(inline_data={
                                "mime_type": "audio/pcm;rate=16000", # Client must send in this format
                                "data": audio_bytes_from_client
                            })
                            await live_session.send_client_content(
                                turns=[Content(role="user", parts=[audio_part_for_live_api])]
                            )
                except WebSocketDisconnect:
                    logging.info(f"Voice client {session_id} (Client -> Live API task) disconnected.")
                except Exception as e:
                    if isinstance(e, asyncio.CancelledError): # Don't log cancellation as an error
                        logging.info(f"Voice client {session_id}: Client -> Live API task cancelled.")
                        raise
                    logging.error(f"Voice client {session_id}: Error in Client -> Live API task: {e}", exc_info=True)
                finally:
                    is_live_session_active = False
                    # If not using VAD, might send audio_stream_end here if session is still active
                    # if live_session and is_live_session_active and \
                    #    not live_api_config.realtime_input_config.automatic_activity_detection:
                    #     await live_session.send_client_content(realtime_input_config={"audio_stream_end": True})


            receive_task = asyncio.create_task(forward_audio_from_live_api_to_client())
            send_task = asyncio.create_task(forward_audio_from_client_to_live_api())

            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            is_live_session_active = False # Ensure flag is set to stop other task
            for task in pending:
                task.cancel()
                try:
                    await task # Wait for cancellation to complete
                except asyncio.CancelledError:
                    logging.info(f"Voice client {session_id}: Task {task.get_name()} successfully cancelled.")
            for task in done:
                if task.exception() and not isinstance(task.exception(), asyncio.CancelledError):
                    logging.error(f"Voice client {session_id}: Task {task.get_name()} raised an exception: {task.exception()}", exc_info=task.exception())
                    # Propagate the exception to the main handler if it's not a CancelledError
                    raise task.exception() 


    except WebSocketDisconnect:
        logging.info(f"Voice client {session_id} main WebSocket handler disconnected.")
    except BlockedPromptException as bpe:
        logging.error(f"Voice client {session_id}: Live API prompt blocked: {bpe}", exc_info=False) # Less verbose for BPE
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "message": "Your request was blocked due to safety settings."}))
    except Exception as e:
        logging.error(f"Voice client {session_id}: Unhandled error in voice WebSocket endpoint: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "message": "An unexpected server error occurred with the voice service."}))
    finally:
        is_live_session_active = False # Double ensure
        logging.info(f"Voice client {session_id} connection closing procedures.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
        logging.info(f"Voice client {session_id} WebSocket fully closed.")