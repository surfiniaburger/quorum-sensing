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


# Add this function inside mcp_server/mlb_stats_server.py

def format_player_game_stats(player_data: Dict[str, Any], player_id: int) -> str:
    """Formats a player's game stats into a readable string."""
    if not player_data:
        return f"No data found for player ID {player_id} in this game."

    person_info = player_data.get("person", {})
    player_name = person_info.get("fullName", f"Player ID {player_id}")
    position = player_data.get("position", {}).get("abbreviation", "N/A")
    stats = player_data.get("stats", {})
    batting_stats = stats.get("batting", {})
    pitching_stats = stats.get("pitching", {})
    fielding_stats = stats.get("fielding", {})

    summary_lines = [f"Stats for {player_name} (Position: {position}):"]

    if batting_stats:
        # Common batting stats, you can add more from the API response
        # Example: atBats, runs, hits, rbi, walks, strikeOuts, homeRuns, baseOnBalls, avg, obp, slg, ops
        # The API often uses camelCase keys.
        b_summary = "  Batting: "
        b_fields = {
            "AB": batting_stats.get("atBats"), "R": batting_stats.get("runs"),
            "H": batting_stats.get("hits"), "RBI": batting_stats.get("rbi"),
            "BB": batting_stats.get("walks") or batting_stats.get("baseOnBalls"), # API sometimes uses 'walks', sometimes 'baseOnBalls'
            "SO": batting_stats.get("strikeOuts"), "HR": batting_stats.get("homeRuns"),
            "AVG": batting_stats.get("avg"), "OBP": batting_stats.get("obp"),
            "SLG": batting_stats.get("slg"), "OPS": batting_stats.get("ops")
        }
        b_parts = [f"{k}: {v}" for k, v in b_fields.items() if v is not None and v != ""] # Only show if stat exists and is not empty
        if not b_parts: # Handle cases like pitchers who didn't bat
            b_summary += "Did not bat or no batting stats recorded."
        else:
            b_summary += ", ".join(b_parts)
        summary_lines.append(b_summary)

    if pitching_stats:
        # Common pitching stats
        # Example: inningsPitched, runs, homeRuns, earnedRuns, strikeOuts, baseOnBalls, era, whips
        p_summary = "  Pitching: "
        p_fields = {
            "IP": pitching_stats.get("inningsPitched"), "H": pitching_stats.get("hits"),
            "R": pitching_stats.get("runs"), "ER": pitching_stats.get("earnedRuns"),
            "BB": pitching_stats.get("walks") or pitching_stats.get("baseOnBalls"),
            "SO": pitching_stats.get("strikeOuts"), "HR": pitching_stats.get("homeRunsAllowed") or pitching_stats.get("homeRuns"),
            "ERA": pitching_stats.get("era"), "WHIP": pitching_stats.get("whip"),
            "Pitches": pitching_stats.get("pitchesThrown"), "Strikes": pitching_stats.get("strikes")
        }
        p_parts = [f"{k}: {v}" for k, v in p_fields.items() if v is not None and v != ""]
        if not p_parts:
             p_summary += "Did not pitch or no pitching stats recorded."
        else:
            p_summary += ", ".join(p_parts)
        summary_lines.append(p_summary)

    if fielding_stats:
        # Common fielding stats
        f_summary = "  Fielding: "
        f_fields = {
            "PO": fielding_stats.get("putOuts"), "A": fielding_stats.get("assists"),
            "E": fielding_stats.get("errors"), "FPct": fielding_stats.get("fielding") # Fielding Percentage
        }
        f_parts = [f"{k}: {v}" for k, v in f_fields.items() if v is not None and v != ""]
        if f_parts: # Only show if there are any fielding stats
            f_summary += ", ".join(f_parts)
            summary_lines.append(f_summary)

    if len(summary_lines) == 1: # Only the header was added
        return f"No specific batting, pitching, or fielding stats found for {player_name} in this game."

    return "\n".join(summary_lines)

@mcp.tool()
async def get_player_stats_for_game(game_pk: int, player_id: int) -> str:
    """
    Retrieves a specific player's batting, pitching, and fielding stats for a given game.

    Args:
        game_pk: The unique identifier for the MLB game.
        player_id: The MLB official ID for the player.
    """
    if not isinstance(game_pk, int):
        try: game_pk = int(game_pk)
        except ValueError: return "Invalid game_pk. Must be an integer."
    if not isinstance(player_id, int):
        try: player_id = int(player_id)
        except ValueError: return "Invalid player_id. Must be an integer."

    logger.info(f"MCP Tool: get_player_stats_for_game called for game_pk={game_pk}, player_id={player_id}")
    game_data = await make_mlb_api_request(
        endpoint_version="v1.1",
        path=f"game/{game_pk}/feed/live"
    )

    if not game_data:
        return f"Could not retrieve game data for game_pk {game_pk}."

    try:
        boxscore = game_data.get("liveData", {}).get("boxscore", {})
        teams_data = boxscore.get("teams", {})
        player_found_data = None

        # Check both home and away teams
        for team_type in ["home", "away"]:
            team_players = teams_data.get(team_type, {}).get("players", {}) # This is a dict with "ID{player_id}" as keys
            player_key = f"ID{player_id}" # Player data in boxscore is keyed like "ID123456"
            if player_key in team_players:
                player_found_data = team_players[player_key]
                break # Found the player

        if player_found_data:
            return format_player_game_stats(player_found_data, player_id)
        else:
            # Fallback: Sometimes players might be in 'batters' or 'pitchers' lists if not in main 'players' boxscore structure for some reason (less common)
            # This part can be expanded if needed, but the 'players' dict is usually comprehensive.
            all_person_ids_in_game = boxscore.get("home", {}).get("batters", []) + \
                                     boxscore.get("home", {}).get("pitchers", []) + \
                                     boxscore.get("away", {}).get("batters", []) + \
                                     boxscore.get("away", {}).get("pitchers", [])
            if player_id not in all_person_ids_in_game : # Check if player ID was even part of lists
                 return f"Player ID {player_id} was not found in the boxscore for game PK {game_pk}."
            return f"Statistics for player ID {player_id} not found in the expected boxscore structure for game PK {game_pk}. They might have played but stats are not detailed here."

    except Exception as e:
        logger.error(f"Error processing player stats for game_pk {game_pk}, player_id {player_id}: {e}", exc_info=True)
        return f"Error processing player stats for player {player_id} in game {game_pk}."


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
async def get_team_schedule(team_id: int, days_range: int = 7) -> str:
    """
    Retrieves a team's game schedule for a specified range of days from today.
    Positive days_range for future games, negative for past games.

    Args:
        team_id: The MLB official ID for the team.
        days_range: Number of days from today. Default is 7 (next 7 days).
                    Use negative for past days (e.g., -3 for last 3 days).
    """
    if not isinstance(team_id, int):
        try: team_id = int(team_id)
        except ValueError: return "Invalid team_id. Must be an integer."
    if not isinstance(days_range, int):
        try: days_range = int(days_range)
        except ValueError: days_range = 7 # default

    logger.info(f"MCP Tool: get_team_schedule called for team_id={team_id}, days_range={days_range}")

    today = datetime.utcnow().date() # Use UTC for consistency with API if possible
    if days_range >= 0:
        start_date = today
        end_date = today + timedelta(days=days_range)
        period_desc = f"next {days_range+1} days" if days_range > 0 else "today"
    else: # days_range is negative
        start_date = today + timedelta(days=days_range) # days_range is negative, so this goes back
        end_date = today
        period_desc = f"last {-days_range} days including today"


    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    schedule_data = await make_mlb_api_request(
        endpoint_version="v1",
        path="schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date_str,
            "endDate": end_date_str,
            # "fields": "dates,date,games,gamePk,officialDate,status,detailedState,teams,away,home,team,name,score,venue,name" # Optional: limit fields
        }
    )

    if not schedule_data or not schedule_data.get("dates"):
        return f"No schedule found for team ID {team_id} for {period_desc} ({start_date_str} to {end_date_str})."

    all_games_formatted = []
    for date_entry in schedule_data.get("dates", []):
        games_on_date = date_entry.get("games", [])
        for game in games_on_date:
            all_games_formatted.append(format_game_schedule_entry(game))

    if not all_games_formatted:
        return f"No games scheduled for team ID {team_id} within {period_desc} ({start_date_str} to {end_date_str})."

    # Fetch team name for better output
    team_name = f"Team ID {team_id}"
    team_info_data = await make_mlb_api_request(endpoint_version="v1", path=f"teams/{team_id}")
    if team_info_data and team_info_data.get("teams"):
        team_name = team_info_data["teams"][0].get("name", team_name)


    return f"Schedule for {team_name} ({period_desc}):\n" + "\n---\n".join(all_games_formatted)


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
async def get_league_standings(league_id: int, season: Optional[int] = None) -> str:
    """
    Retrieves the current standings for a given MLB league (AL or NL).

    Args:
        league_id: The ID for the league (103 for American League, 104 for National League).
        season: The season year (e.g., 2024). Defaults to the current MLB season if not provided.
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
            current_year = datetime.utcnow().year
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
            season = datetime.utcnow().year # Fallback to current calendar year
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


# You can add more tools here:
# - get_player_stats_for_game(game_pk: int, player_id: int)
# - get_team_schedule(team_id: int, days_ahead: int = 7)
# - get_league_standings()

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting MLB Stats MCP Server...")
    mcp.run(transport="stdio")