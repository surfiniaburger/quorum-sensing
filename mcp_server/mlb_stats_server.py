# mcp_server/mlb_stats_server.py
import json
import logging
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, UTC, timedelta

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


# --- Team ID Mapping ---
TEAMS = {
    'rangers': 140, 'texas rangers': 140,
    'angels': 108, 'los angeles angels': 108, 'anaheim angels': 108,
    'astros': 117, 'houston astros': 117,
    'rays': 139, 'tampa bay rays': 139,
    'blue jays': 141, 'toronto blue jays': 141,
    'jays': 141,
    'yankees': 147, 'new york yankees': 147,
    'orioles': 110, 'baltimore orioles': 110,
    'red sox': 111, 'boston red sox': 111,
    'sox': 111, # Could be ambiguous with White Sox, but often refers to Red Sox in general context
    'twins': 142, 'minnesota twins': 142,
    'white sox': 145, 'chicago white sox': 145,
    'guardians': 114, 'cleveland guardians': 114,
    'tigers': 116, 'detroit tigers': 116,
    'royals': 118, 'kansas city royals': 118,
    'padres': 135, 'san diego padres': 135,
    'giants': 137, 'san francisco giants': 137,
    'diamondbacks': 109, 'arizona diamondbacks': 109, 'd-backs': 109,
    'rockies': 115, 'colorado rockies': 115,
    'phillies': 143, 'philadelphia phillies': 143,
    'braves': 144, 'atlanta braves': 144,
    'marlins': 146, 'miami marlins': 146,
    'nationals': 120, 'washington nationals': 120, 'nats': 120,
    'mets': 121, 'new york mets': 121,
    'pirates': 134, 'pittsburgh pirates': 134,
    'cardinals': 138, 'st. louis cardinals': 138, 'cards': 138,
    'brewers': 158, 'milwaukee brewers': 158,
    'cubs': 112, 'chicago cubs': 112,
    'reds': 113, 'cincinnati reds': 113,
    'athletics': 133, 'oakland athletics': 133, "a's": 133,
    'mariners': 136, 'seattle mariners': 136,
    'dodgers': 119, 'los angeles dodgers': 119,
}

CURRENT_YEAR = datetime.now(UTC).year

# --- Helper Functions ---

def internal_resolve_team_id(identifier: str) -> Optional[int]:
    """
    Tries to convert identifier to int if it's numeric, otherwise looks up in TEAMS.
    This is an INTERNAL helper.
    """
    try:
        # Check if the identifier string is purely numeric
        if identifier.isdigit():
            team_id_int = int(identifier)
            # You might add a sanity check for typical ID ranges, e.g., 100-160
            # For now, we assume if it's an int, it's an ID attempt.
            return team_id_int
    except ValueError:
        # Not a string that directly converts to int, treat as name
        pass
    return TEAMS.get(identifier.lower().strip())


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
async def get_game_boxscore_and_details(game_pk: int) -> str:
    """
    Retrieves comprehensive game details including game metadata, boxscore, and linescore
    for a given MLB game_pk. This provides richer data for generating detailed recaps.
    Returns a JSON string of the extracted data.
    """
    if not isinstance(game_pk, int):
        try:
            game_pk = int(game_pk)
        except ValueError:
            return json.dumps({"error": "Invalid game_pk provided. It must be an integer."})

    logger.info(f"MCP Tool: get_game_boxscore_and_details called for game_pk={game_pk}")
    
    full_feed_data = await make_mlb_api_request(
        endpoint_version="v1.1",
        path=f"game/{game_pk}/feed/live"
    )

    if not full_feed_data:
        return json.dumps({"error": f"Could not retrieve comprehensive data for game_pk {game_pk}."})

    extracted_data = {}
    
    if "gameData" in full_feed_data:
        extracted_data["game_info"] = full_feed_data["gameData"]
    else:
        logger.warning(f"gameData not found in feed for game_pk {game_pk}")
        extracted_data["game_info"] = None

    live_data = full_feed_data.get("liveData", {})
    
    if "boxscore" in live_data:
        extracted_data["boxscore"] = live_data["boxscore"]
    else:
        logger.warning(f"liveData.boxscore not found in feed for game_pk {game_pk}")
        extracted_data["boxscore"] = None
        
    if "linescore" in live_data:
        extracted_data["linescore"] = live_data["linescore"]
    else:
        logger.warning(f"liveData.linescore not found in feed for game_pk {game_pk}")
        extracted_data["linescore"] = None
        
    # Optionally, include a small, fixed number of recent plays if needed for context,
    # but avoid allPlays to keep the payload size manageable.
    # all_plays = live_data.get("plays", {}).get("allPlays", [])
    # if all_plays:
    #     extracted_data["recent_plays_sample"] = all_plays[-5:] # Last 5 plays
    # else:
    #     extracted_data["recent_plays_sample"] = []


    if not extracted_data.get("game_info") and not extracted_data.get("boxscore") and not extracted_data.get("linescore"):
        return json.dumps({"error": f"No key data sections (game_info, boxscore, linescore) found in feed for game_pk {game_pk}."})

    logger.info(f"Successfully extracted game_info, boxscore, and linescore for game_pk {game_pk}.")
    # Use default=str to handle any non-serializable types like datetime objects if they sneak in
    return json.dumps(extracted_data, default=str)


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

# Add this function inside mcp_server/mlb_stats_server.py

def format_player_game_stats(player_data: Dict[str, Any], player_id: int) -> str:
    # (Implementation as before)
    if not player_data: return f"No data found for player ID {player_id} in this game."
    person_info = player_data.get("person", {})
    player_name = person_info.get("fullName", f"Player ID {player_id}")
    position = player_data.get("position", {}).get("abbreviation", "N/A")
    stats = player_data.get("stats", {})
    batting_stats = stats.get("batting", {})
    pitching_stats = stats.get("pitching", {})
    fielding_stats = stats.get("fielding", {})
    summary_lines = [f"Stats for {player_name} (Position: {position}):"]
    if batting_stats:
        b_summary = "  Batting: "
        b_fields = {"AB": batting_stats.get("atBats"), "R": batting_stats.get("runs"), "H": batting_stats.get("hits"), "RBI": batting_stats.get("rbi"), "BB": batting_stats.get("walks") or batting_stats.get("baseOnBalls"), "SO": batting_stats.get("strikeOuts"), "HR": batting_stats.get("homeRuns"), "AVG": batting_stats.get("avg"), "OBP": batting_stats.get("obp"), "SLG": batting_stats.get("slg"), "OPS": batting_stats.get("ops")}
        b_parts = [f"{k}: {v}" for k, v in b_fields.items() if v is not None and v != ""]
        b_summary += ", ".join(b_parts) if b_parts else "Did not bat or no batting stats recorded."
        summary_lines.append(b_summary)
    if pitching_stats:
        p_summary = "  Pitching: "
        p_fields = {"IP": pitching_stats.get("inningsPitched"), "H": pitching_stats.get("hits"), "R": pitching_stats.get("runs"), "ER": pitching_stats.get("earnedRuns"), "BB": pitching_stats.get("walks") or pitching_stats.get("baseOnBalls"), "SO": pitching_stats.get("strikeOuts"), "HR": pitching_stats.get("homeRunsAllowed") or pitching_stats.get("homeRuns"), "ERA": pitching_stats.get("era"), "WHIP": pitching_stats.get("whip"), "Pitches": pitching_stats.get("pitchesThrown"), "Strikes": pitching_stats.get("strikes")}
        p_parts = [f"{k}: {v}" for k, v in p_fields.items() if v is not None and v != ""]
        p_summary += ", ".join(p_parts) if p_parts else "Did not pitch or no pitching stats recorded."
        summary_lines.append(p_summary)
    if fielding_stats:
        f_summary = "  Fielding: "
        f_fields = {"PO": fielding_stats.get("putOuts"), "A": fielding_stats.get("assists"), "E": fielding_stats.get("errors"), "FPct": fielding_stats.get("fielding")}
        f_parts = [f"{k}: {v}" for k, v in f_fields.items() if v is not None and v != ""]
        if f_parts: f_summary += ", ".join(f_parts); summary_lines.append(f_summary)
    if len(summary_lines) == 1: return f"No specific batting, pitching, or fielding stats found for {player_name} in this game."
    return "\n".join(summary_lines)



# Add these imports at the top of mcp_server/mlb_stats_server.py
from datetime import datetime, timedelta

# Add this function inside mcp_server/mlb_stats_server.py
def format_game_schedule_entry(game_info: Dict[str, Any]) -> str:
    """Formats a single game from the schedule into a readable string."""
    game_pk = game_info.get("gamePk", "N/A")
    game_date = game_info.get("officialDate", "N/A") # Or gameDate for just date
    status = game_info.get("status", {}).get("detailedState", "Scheduled")
    teams = game_info.get("teams", {})
    away_team = teams.get("away", {}).get("team", {}).get("name", "Away Team")
    away_score = teams.get("away", {}).get("score", "")
    home_team = teams.get("home", {}).get("team", {}).get("name", "Home Team")
    home_score = teams.get("home", {}).get("score", "")
    venue = game_info.get("venue", {}).get("name", "N/A")

    score_str = ""
    if status not in ["Scheduled", "Pre-Game", "Postponed"]: # Only show score if game is in progress or final
        score_str = f" ({away_team} {away_score} - {home_team} {home_score})"


    return (f"Date: {game_date}, GamePK: {game_pk}\n"
            f"  Matchup: {away_team} @ {home_team}{score_str}\n"
            f"  Venue: {venue}\n"
            f"  Status: {status}")

@mcp.tool()
async def get_team_schedule(team_identifier: str, days_range: int = 7) -> str:
    """
    Retrieves a team's game schedule.
    Provide team name (e.g., "Dodgers") or team ID as a string (e.g., "119").
    Days_range is for N days from today (e.g., 7 for next 7 days, -3 for last 3 days).

    Args:
        team_identifier: Team name or ID (as a string).
        days_range: Number of days from today. Default 7.
    """
    team_id = internal_resolve_team_id(team_identifier)
    if team_id is None:
        return f"Team '{team_identifier}' not recognized. Please use a known team name or a numeric team ID."

    # days_range will be an int due to tool signature, or its default
    actual_days_range = days_range

    logger.info(f"MCP Tool: get_team_schedule for team_id={team_id} (from '{team_identifier}'), days_range={actual_days_range}")

    today = datetime.utcnow().date()
    if actual_days_range >= 0:
        start_date = today
        end_date = today + timedelta(days=actual_days_range)
        period_desc = f"next {actual_days_range+1} days" if actual_days_range > 0 else "today"
    else:
        start_date = today + timedelta(days=actual_days_range)
        end_date = today
        period_desc = f"last {-actual_days_range} days including today"

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    schedule_data = await make_mlb_api_request(
        endpoint_version="v1",
        path="schedule",
        params={"sportId": 1, "teamId": team_id, "startDate": start_date_str, "endDate": end_date_str}
    )

    if not schedule_data or not schedule_data.get("dates"):
        return f"No schedule found for {team_identifier} (ID: {team_id}) for {period_desc} ({start_date_str} to {end_date_str})."

    # ... (formatting logic from previous version)
    all_games_formatted = []
    for date_entry in schedule_data.get("dates", []):
        games_on_date = date_entry.get("games", [])
        for game in games_on_date:
            all_games_formatted.append(format_game_schedule_entry(game))
    if not all_games_formatted:
        return f"No games scheduled for {team_identifier} (ID: {team_id}) within {period_desc} ({start_date_str} to {end_date_str})."
    team_name_display = team_identifier.title()
    return f"Schedule for {team_name_display} ({period_desc}):\n" + "\n---\n".join(all_games_formatted)



# Add this function inside mcp_server/mlb_stats_server.py

def format_standings_record(team_record: Dict[str, Any], division_name: str) -> str:
    """Formats a single team's standings record."""
    team_info = team_record.get("team", {})
    team_name = team_info.get("name", "Unknown Team")
    wins = team_record.get("wins", "N/A")
    losses = team_record.get("losses", "N/A")
    pct = team_record.get("winningPercentage", "N/A")
    gb = team_record.get("gamesBack", "-")
    division_rank = team_record.get("divisionRank", "N/A")
    # Wildcard rank might be in a different part or need separate logic depending on endpoint
    # For /standings endpoint, 'wildCardRank' and 'wildCardGamesBack' are usually present
    wc_rank = team_record.get("wildCardRank", "N/A")
    wc_gb = team_record.get("wildCardGamesBack", "-")

    return (f"  {team_name:<25} | W: {wins:<3} L: {losses:<3} PCT: {pct:<6} GB: {gb:<5} "
            f"DivRank: {division_rank:<2} WC Rank: {wc_rank:<2} WC GB: {wc_gb}")


@mcp.tool()
async def get_league_standings(league_id: int, season: int) -> str:
    """
    Retrieves the current standings for a given MLB league (AL or NL).
    Provide the season year. If you are unsure of the current season, you can try providing the current calendar year.

    Args:
        league_id: The ID for the league (103 for American League, 104 for National League).
        season: The season year (e.g., 2024).
    """
    if league_id not in [103, 104]:
        return "Invalid league_id. Use 103 for American League or 104 for National League."

    if season is None:
        # Attempt to get current season, or default if API call fails
        season_data = await make_mlb_api_request(endpoint_version="v1", path="seasons/all", params={"sportId": 1})
        if season_data and season_data.get("seasons"):
            # Find the season with "hasWildcard": true and a current date range, or just the latest.
            # This logic can be complex. For simplicity, we'll try to find the latest marked as "current"
            # or just the absolute latest.
            # A more robust way is to get seasonId from /api/v1/schedule/games for today's date.
            # For now, let's just use the current year if API fails.
            current_year = datetime.now(UTC)
            latest_api_season = current_year
            for s_entry in reversed(season_data["seasons"]): # Check latest first
                if s_entry.get("seasonStartDate") and s_entry.get("regularSeasonEndDate"):
                    # A rough check if today is within a season, or if it's the most recent one
                    # This is not perfect for determining the "active" standings season mid-offseason.
                    # The API usually defaults to the latest completed or current season for standings.
                    latest_api_season = int(s_entry.get("seasonId")) # seasonId is the year
                    break
            season = latest_api_season
        else:
            season = datetime.now(UTC) # Fallback to current calendar year
        logger.info(f"MCP Tool: get_league_standings - season not provided, using: {season}")


    if not isinstance(season, int):
        try:
            season = int(season)
        except ValueError:
            return "Invalid season year. Must be an integer."


    logger.info(f"MCP Tool: get_league_standings called for league_id={league_id}, season={season}")
    standings_data = await make_mlb_api_request(
        endpoint_version="v1",
        path="standings",
        params={
            "leagueId": league_id,
            "season": season,
            "standingsTypes": "regularSeason,wildCard", # Get both division and wildcard views
            # "fields": "..." # Can specify fields to reduce payload
        }
    )

    if not standings_data or not standings_data.get("records"):
        return f"Could not retrieve standings for league ID {league_id} for season {season}."

    output_lines = [f"Standings for League ID {league_id} - Season {season}:"]
    league_name = "American League" if league_id == 103 else "National League"
    output_lines.append(f"--- {league_name} ---")

    for record_group in standings_data.get("records", []):
        # 'standingsType' can be 'wildCard', 'regularSeason' (division based)
        # 'division' or 'league' object provides context
        division_info = record_group.get("division", {})
        league_info = record_group.get("league", {}) # For Wild Card standings
        standings_type = record_group.get("standingsType", "N/A")


        group_name = division_info.get("nameShort") # e.g., AL East, NL West
        if not group_name and standings_type == "wildCard":
            group_name = f"{league_info.get('nameShort', 'League')} Wild Card"
        elif not group_name:
            group_name = "Unknown Group"


        output_lines.append(f"\n{group_name} ({standings_type}):")
        team_records = record_group.get("teamRecords", [])
        if not team_records:
            output_lines.append("  No team records found for this group.")
            continue

        for team_rec in team_records:
            output_lines.append(format_standings_record(team_rec, group_name))

    return "\n".join(output_lines)


def format_team_roster(
    roster_entries: List[Dict[str, Any]],
    team_name: str,
    team_id: int,
    season: int,
    roster_type: str
) -> str:
    """Formats team roster data into a readable string."""
    if not roster_entries:
        return f"No roster information found for {team_name} (ID: {team_id}) for the {season} season" \
               f"{f' (Type: {roster_type})' if roster_type else ''}."

    type_display = f" ({roster_type} Roster)" if roster_type else " Roster"
    summary_lines = [f"{team_name}{type_display} - {season} Season (Team ID: {team_id}):"]

    for player_entry in roster_entries:
        person = player_entry.get("person", {})
        player_name = person.get("fullName", "N/A")
        player_id = person.get("id", "N/A")
        jersey_number = player_entry.get("jerseyNumber", "--")
        position = player_entry.get("position", {}).get("abbreviation", "N/A")
        status = player_entry.get("status", {}).get("description", "N/A") # e.g., "Active", "Injured List"

        summary_lines.append(
            f"  - #{jersey_number:<3} {player_name:<25} (ID: {player_id}) | Pos: {position:<5} | Status: {status}"
        )

    return "\n".join(summary_lines)


@mcp.tool()
async def get_team_roster(team_identifier: str, season: int = 0, roster_type: str = "") -> str:
    """
    Retrieves a team's roster.
    Provide team name (e.g., "Dodgers") or team ID as a string (e.g., "119").
    Season is year (e.g., 2024); 0 or empty means current year.
    Roster type (e.g., "40Man", "active"); empty means default.

    Args:
        team_identifier: Team name or ID (as a string).
        season: Season year. Default 0 (uses current year).
        roster_type: Roster type. Default "" (uses API default).
    """
    team_id = internal_resolve_team_id(team_identifier)
    if team_id is None:
        return f"Team '{team_identifier}' not recognized. Please use a known team name or a numeric team ID."

    actual_season = season
    if not actual_season or actual_season == 0: # Check for 0 or potentially other sentinel if needed
        actual_season = CURRENT_YEAR
        logger.info(f"MCP Tool: get_team_roster - season not provided or 0, defaulting to {actual_season}")

    actual_roster_type = roster_type.strip()

    logger.info(f"MCP Tool: get_team_roster for team_id={team_id} (from '{team_identifier}'), season={actual_season}, roster_type='{actual_roster_type}'")

    params: Dict[str, Any] = {"season": actual_season}
    if actual_roster_type: # Only add if it's not an empty string
        params["rosterType"] = actual_roster_type

    roster_data_response = await make_mlb_api_request(
        endpoint_version="v1",
        path=f"teams/{team_id}/roster",
        params=params
    )

    official_team_name = team_identifier.title()
    team_info_data = await make_mlb_api_request(endpoint_version="v1", path=f"teams/{team_id}")
    if team_info_data and team_info_data.get("teams"):
        official_team_name = team_info_data["teams"][0].get("name", official_team_name)

    if not roster_data_response or "roster" not in roster_data_response:
        error_msg = f"Could not retrieve roster for {official_team_name} (ID: {team_id}) for season {actual_season}"
        if actual_roster_type: error_msg += f" (type: {actual_roster_type})"
        return error_msg + "."

    return format_team_roster(
        roster_data_response["roster"], official_team_name, team_id, actual_season, actual_roster_type if actual_roster_type else None
    )

# ... (rest of your mlb_stats_server.py, including if __name__ == "__main__":)
# You can add more tools here:
# - get_player_stats_for_game(game_pk: int, player_id: int)
# - get_team_schedule(team_id: int, days_ahead: int = 7)
# - get_league_standings()

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting MLB Stats MCP Server...")
    mcp.run(transport="stdio")