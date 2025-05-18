# mcp_server/image_embedding_server.py
# (Derived from image_embedding_pipeline2.py)

import logging
import json
import os
import re
import time
from typing import Any, Dict, Optional, List, Set
import asyncio

from google.cloud import bigquery, storage
from google.api_core import exceptions as google_exceptions
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from mcp.server.fastmcp import FastMCP

# --- Configuration (Same as pipeline script) ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "mlb_rag_data_2024")
EMBEDDING_TABLE_ID = os.getenv("EMBEDDING_TABLE_ID", "mlb_image_embeddings_sdk")
BQ_FULL_EMBEDDING_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
VERTEX_MULTIMODAL_MODEL_NAME = "multimodalembedding@001"
EMBEDDING_DIMENSIONALITY = 1408

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("image_embedding")

# --- Initialize Clients ---
bq_client = None
storage_client = None
multimodal_embedding_model = None

try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(VERTEX_MULTIMODAL_MODEL_NAME)
    logger.info("BQ, Storage, and Multimodal Embedding clients initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
    # Allow server to start but tools will fail

# --- BQ Helper (Copied from bq_vector_search_server) ---
async def _execute_bq_query_async(query: str) -> Optional[List[Dict]]:
    if not bq_client: return None
    try:
        logger.info(f"Executing BQ Query (async wrapper): {query[:200]}...")
        query_job = await asyncio.to_thread(bq_client.query, query)
        results_df = await asyncio.to_thread(query_job.to_dataframe)
        logger.info(f"BQ Query returned {len(results_df)} rows.")
        results_df = results_df.fillna('')
        return results_df.to_dict('records')
    except google_exceptions.NotFound:
        logger.warning(f"BQ resource not found for query: {query[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error executing BQ query: {query[:200]}... Error: {e}", exc_info=True)
        return None

# --- MCP Tools ---
@mcp.tool()
async def search_similar_images_by_text(query_text: str, top_k: int = 1, filter_image_type: str = "") -> str:
    """
    Performs vector search on the image embedding table using a text query.
    Returns a JSON string list of result dictionaries.
    """
    if not bq_client or not multimodal_embedding_model:
        return json.dumps({"error": "BQ or Embedding service not available."})
    if not query_text:
        return json.dumps({"error": "Query text cannot be empty."})

    logger.info(f"IMAGE_EMB_MCP: search_similar_images - Query: '{query_text}', TopK: {top_k}, Filter: {filter_image_type}")

    # 1. Generate query embedding
    query_embedding = None
    try:
        query_response = await asyncio.to_thread(
            multimodal_embedding_model.get_embeddings,
            contextual_text=query_text,
            dimension=EMBEDDING_DIMENSIONALITY,
        )
        query_embedding = query_response.text_embedding
        if not query_embedding: raise ValueError("SDK returned no text embedding.")
    except Exception as query_emb_err:
        logger.error(f"Failed to get query embedding via SDK: {query_emb_err}")
        # Could add BQ fallback here if needed, similar to pipeline script
        return json.dumps({"error": "Failed to generate query embedding."})

    query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

    # 2. Perform Vector Search (Corrected OPTIONS formatting)
    options_str = ""
    if filter_image_type and filter_image_type.lower() != "none" and filter_image_type.strip() != "":
        safe_filter_value = filter_image_type.replace("'", "''")
        # Correctly escape the quotes for the JSON string within the SQL string
        filter_json_value = json.dumps({"filter": f"image_type='{safe_filter_value}'"})
        options_str = f",\n                options => '{filter_json_value}'" # Pass JSON as escaped SQL string

    vector_search_sql = f"""
    SELECT base.image_uri, base.image_type, base.entity_id, base.entity_name, distance
    FROM VECTOR_SEARCH(
            TABLE `{BQ_FULL_EMBEDDING_TABLE_ID}`,
            'embedding',
            (SELECT {query_embedding_str} AS embedding),
            top_k => {int(top_k)},
            distance_type => 'COSINE'{options_str}
        );
    """
    # logger.debug(f"Executing Image Vector Search SQL:\n{vector_search_sql}") # Optional debug

    rows = await _execute_bq_query_async(vector_search_sql)

    if rows is None:
        return json.dumps({"error": "Error performing image vector search."})
    if not rows:
        logger.info(f"No similar images found for query: '{query_text}'.")
        return json.dumps([])

    logger.info(f"IMAGE_EMB_MCP: Image search returning {len(rows)} results.")
    return json.dumps(rows, default=str) # Handle potential datetime etc.


# --- Optional: Add tools for generating/loading embeddings if needed by agent ---
# @mcp.tool()
# async def generate_and_store_image_embedding(image_gcs_uri: str, image_type: str, ...) -> str:
#    # Logic from generate_embeddings_sdk and load_embeddings_to_bq for a single image
#    # ... implementation ...
#    pass


if __name__ == "__main__":
    logger.info("Starting Image Embedding MCP Server...")
    mcp.run(transport="stdio")