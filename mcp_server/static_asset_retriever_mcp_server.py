# mcp_server/static_asset_retriever_mcp_server.py (REVISED)
import logging
import json
import os
from typing import Any, Dict, Optional, List
import asyncio
from google.cloud import storage
from mcp.server.fastmcp import FastMCP

GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCS_BUCKET_HEADSHOTS = "mlb-headshots"
GCS_PREFIX_HEADSHOTS = "headshots/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
mcp = FastMCP("static_retriever")
storage_client = None
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("GCS client initialized for static_retriever.")
except Exception as e:
    logger.critical(f"Failed to initialize GCS client for static_retriever: {e}")

async def _check_gcs_blob_exists_async(bucket_name: str, blob_name: str) -> bool:
    if not storage_client: return False
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return await asyncio.to_thread(blob.exists)
    except Exception as e:
        logger.error(f"Error checking GCS blob {bucket_name}/{blob_name}: {e}")
        return False

@mcp.tool()
async def get_headshot_uri_if_exists(player_id_str: str) -> str:
    """
    Checks if a headshot exists for the given player_id_str in GCS.
    Returns a JSON string with {"image_uri": "gs://...", "type": "headshot", "entity_id": player_id_str}
    or {"image_uri": null} if not found.
    """
    if not player_id_str:
        return json.dumps({"image_uri": None, "error": "Player ID is required."})
    if not storage_client:
        return json.dumps({"image_uri": None, "error": "Storage client not initialized."})

    logger.info(f"STATIC_RETRIEVER_MCP: Checking headshot for player_id: {player_id_str}")
    expected_blob_name = f"{GCS_PREFIX_HEADSHOTS}headshot_{player_id_str}.jpg"
    uri = f"gs://{GCS_BUCKET_HEADSHOTS}/{expected_blob_name}"

    if await _check_gcs_blob_exists_async(GCS_BUCKET_HEADSHOTS, expected_blob_name):
        logger.info(f"STATIC_RETRIEVER_MCP: Headshot found: {uri}")
        return json.dumps({
            "image_uri": uri,
            "type": "headshot",
            "entity_id": player_id_str
        })
    else:
        logger.warning(f"STATIC_RETRIEVER_MCP: Headshot not found for player_id {player_id_str} at {expected_blob_name}")
        return json.dumps({"image_uri": None})

if __name__ == "__main__":
    logger.info("Starting Static Asset Retriever MCP Server (for headshots)...")
    mcp.run(transport="stdio")