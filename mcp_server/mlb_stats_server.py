# mcp_server/mlb_stats_server.py
import json
import logging
from typing import Any, Dict, Optional

import httpx # Using httpx for async requests, similar to your weather_server
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with a unique name
mcp = FastMCP("mlb_stats")

# --- Configuration & Constants ---
MLB_API_BASE_URL = "https://statsapi.mlb.com/api/"
# If you have your LangGraph's rate-limited call_mlb_api, you might adapt it here.
# For simplicity, we'll make direct calls with basic error handling.
# A production system would need robust rate limiting and error handling.
REQUEST_TIMEOUT = 20.0
USER_AGENT = "mlb-stats-mcp-agent/1.0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
async def make_mlb_api_request(
    endpoint_version: str, # e.g., "v1" or "v1.1"
    path: str,             # e.g., "game/{game_pk}/feed/live"
    params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Makes a request to the MLB Stats API and returns the JSON response.
    """
    url = f"{MLB_API_BASE_URL}{endpoint_version}/{path}"
    headers = {"User-Agent": USER_AGENT}
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Requesting MLB API: {url} with params: {params}")
            response = await client.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            if not data: # Handle empty responses
                logger.warning(f"Empty response from MLB API: {url}")
                return None
            return data
    except httpx.HTTPStatusError as e:
        logger.error(f"MLB API HTTP error: {e.response.status_code} for {e.request.url!r}")
        return None
    except httpx.RequestError as e:
        logger.error(f"MLB API Request error for {e.request.url!r}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"MLB API JSON Decode error for {url}: {e}")
        return None

def format_game_score_summary(game_data: Dict[str, Any], game_pk: Any) -> str:
    """Formats live game data into a score summary string."""
    try:
        live_data = game_data.get("liveData", {})
        linescore = live_data.get("linescore", {})
        teams = linescore.get("teams", {})
        home_team_runs = teams.get("home", {}).get("runs", "N/A")
        away_team_runs = teams.get("away", {}).get("runs", "N/A")

        game_info = game_data.get("gameData", {})
        game_status = game_info.get("status", {}).get("detailedState", "Unknown Status")
        home_team_name = game_info.get("teams", {}).get("home", {}).get("name", "Home")
        away_team_name = game_info.get("teams", {}).get("away", {}).get("name", "Away")

        current_inning = linescore.get("currentInning", "N/A")
        inning_half = linescore.get("halfInning", "")
        inning_state = linescore.get("inningState", "") # e.g., Top, Bottom, Middle, End

        summary = (
            f"Game PK: {game_pk}\n"
            f"Status: {game_status}\n"
            f"Score: {away_team_name} {away_team_runs} - {home_team_name} {home_team_runs}\n"
            f"Current Inning: {inning_half} {current_inning} ({inning_state})"
        )
        if "outs" in linescore:
            summary += f"\nOuts: {linescore.get('outs')}"

        return summary
    except Exception as e:
        logger.error(f"Error formatting game score summary for PK {game_pk}: {e}")
        return f"Error formatting score for Game PK {game_pk}."

# --- MCP Tools for MLB Stats API ---

@mcp.tool()
async def get_live_game_score(game_pk: int) -> str:
    """
    Retrieves the live score and basic status for a given MLB game_pk.

    Args:
        game_pk: The unique identifier for the MLB game.
    """
    if not isinstance(game_pk, int):
        try:
            game_pk = int(game_pk)
        except ValueError:
            return "Invalid game_pk provided. It must be an integer."

    logger.info(f"MCP Tool: get_live_game_score called for game_pk={game_pk}")
    data = await make_mlb_api_request(
        endpoint_version="v1.1",
        path=f"game/{game_pk}/feed/live"
        # path=f"game/{game_pk}/linescore" # Alternative, more focused endpoint
    )
    if data:
        return format_game_score_summary(data, game_pk)
    return f"Could not retrieve live score for game_pk {game_pk}."

@mcp.tool()
async def get_game_play_by_play_summary(game_pk: int, count: int = 3) -> str:
    """
    Retrieves a summary of the last few plays for a given MLB game_pk.

    Args:
        game_pk: The unique identifier for the MLB game.
        count: The number of recent plays to summarize (default is 3).
    """
    if not isinstance(game_pk, int):
        try:
            game_pk = int(game_pk)
        except ValueError:
            return "Invalid game_pk for play-by-play. It must be an integer."
    if not isinstance(count, int) or count <= 0:
        count = 3

    logger.info(f"MCP Tool: get_game_play_by_play_summary called for game_pk={game_pk}, count={count}")
    data = await make_mlb_api_request(
        endpoint_version="v1.1",
        path=f"game/{game_pk}/feed/live" # Full feed contains plays
    )
    if data and "liveData" in data and "plays" in data["liveData"] and "allPlays" in data["liveData"]["plays"]:
        all_plays = data["liveData"]["plays"]["allPlays"]
        if not all_plays:
            return f"No plays found for game_pk {game_pk}."

        # Get the last 'count' plays
        recent_plays = all_plays[-count:]
        summaries = []
        for play in reversed(recent_plays): # Show most recent first
            about = play.get("about", {})
            result = play.get("result", {})
            description = result.get("description", "No description.")
            inning = about.get("inning", "?")
            half_inning = about.get("halfInning", "")
            summaries.append(f"Inning {inning} ({half_inning}): {description}")

        if not summaries:
            return f"Could not summarize recent plays for game_pk {game_pk}."
        return f"Recent Plays for Game PK {game_pk} (most recent first):\n" + "\n".join(summaries)
    return f"Could not retrieve play-by-play data for game_pk {game_pk}."


# You can add more tools here:
# - get_player_stats_for_game(game_pk: int, player_id: int)
# - get_team_schedule(team_id: int, days_ahead: int = 7)
# - get_league_standings()

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting MLB Stats MCP Server...")
    mcp.run(transport="stdio")