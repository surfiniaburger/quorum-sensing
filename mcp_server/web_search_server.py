# mcp_server/web_search_server.py
import logging
import json
import os
from typing import Any, Dict, Optional, List
import asyncio

from tavily import TavilyClient
from google.cloud import secretmanager
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
TAVILY_SECRET_ID = os.getenv("TAVILY_SECRET_ID", "TAVILY_SEARCH")
TAVILY_SECRET_VERSION = "latest"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("web_search")

# --- Initialize Clients ---
tavily = None
secret_manager_client = None

try:
    secret_manager_client = secretmanager.SecretManagerServiceClient()

    # --- Helper Function to Access Secret Manager ---
    def access_secret_version(project_id: str, secret_id: str, version_id: str = "latest") -> Optional[str]:
        if not secret_manager_client: return None
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        try:
            response = secret_manager_client.access_secret_version(request={"name": name})
            payload = response.payload.data.decode("UTF-8")
            logger.info(f"Successfully accessed secret: {secret_id}")
            return payload
        except Exception as e:
            logger.error(f"Error accessing secret {secret_id}: {e}", exc_info=False)
            return None

    tavily_api_key = access_secret_version(GCP_PROJECT_ID, TAVILY_SECRET_ID, TAVILY_SECRET_VERSION)
    if tavily_api_key:
        tavily = TavilyClient(api_key=tavily_api_key)
        logger.info("Tavily client initialized successfully.")
    else:
        logger.warning("Tavily API key not found or accessible. Web search tool will not function.")

except Exception as e:
    logger.critical(f"Failed to initialize Secret Manager client: {e}", exc_info=True)
    # Allow server to start, but Tavily client will be None

# --- Internal Helper for Tavily Search ---
async def _call_tavily_search_async(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """Calls Tavily search API asynchronously."""
    if not tavily:
        logger.warning("Tavily client not initialized, skipping web search.")
        return []
    try:
        logger.info(f"Performing Tavily search for: '{query}'")
        # Use asyncio.to_thread for the synchronous Tavily client library call
        response = await asyncio.to_thread(
            tavily.search,
            query=query,
            max_results=max_results,
            include_raw_content=False,
            search_depth="basic" # Or "advanced"
        )
        results = response.get("results", [])
        logger.info(f" -> Tavily returned {len(results)} results.")
        return results # Return the list of result dictionaries
    except Exception as e:
        logger.error(f"Error calling Tavily API for query '{query}': {e}", exc_info=True)
        return []

# --- MCP Tool ---
@mcp.tool()
async def perform_web_search(query: str, max_results: int = 2) -> str:
    """
    Performs a web search using the Tavily API for the given query.
    Returns a JSON string list of result dictionaries (containing url, content, score, etc.).
    """
    if not tavily:
        return json.dumps({"error": "Tavily client not available."})
    if not query:
        return json.dumps({"error": "Search query cannot be empty."})

    logger.info(f"WEB_SEARCH_MCP: perform_web_search - Query: '{query}', MaxResults: {max_results}")

    search_results = await _call_tavily_search_async(query, max_results)

    # Return the list of result dictionaries as a JSON string
    return json.dumps(search_results)

if __name__ == "__main__":
    logger.info("Starting Web Search MCP Server (Tavily)...")
    mcp.run(transport="stdio")