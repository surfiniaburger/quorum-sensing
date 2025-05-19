# mcp_server/image_embedding_server.py

import logging
import json
import os
from typing import Any, Dict, Optional, List
import asyncio

from google.cloud import bigquery, storage
from google.api_core import exceptions as google_exceptions
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel # Removed 'Image' as it wasn't used directly here
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "mlb_rag_data_2024")
EMBEDDING_TABLE_ID = os.getenv("EMBEDDING_TABLE_ID", "mlb_image_embeddings_sdk")
BQ_FULL_EMBEDDING_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
VERTEX_MULTIMODAL_MODEL_NAME = "multimodalembedding@001"
EMBEDDING_DIMENSIONALITY = 1408 # Ensure this matches your table's embedding dimension

# Use a more detailed format for logging, especially for server-side logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP("image_embedding")

# --- Initialize Clients ---
bq_client = None
storage_client = None # Though not used in search_similar_images_by_text, good practice if adding other tools
multimodal_embedding_model = None

try:
    logger.info("Initializing BQ client...")
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    logger.info("BQ client initialized.")

    logger.info("Initializing Storage client...")
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Storage client initialized.")

    logger.info(f"Initializing Vertex AI SDK for project {GCP_PROJECT_ID} and location {GCP_LOCATION}...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logger.info("Vertex AI SDK initialized.")

    logger.info(f"Loading Multimodal Embedding model: {VERTEX_MULTIMODAL_MODEL_NAME}...")
    multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(VERTEX_MULTIMODAL_MODEL_NAME)
    logger.info("Multimodal Embedding model loaded.")

    logger.info("All clients initialized successfully for image_embedding_server.")
except Exception as e:
    logger.critical(f"CRITICAL - Failed to initialize clients in image_embedding_server: {e}", exc_info=True)
    # Server might still start but tools will fail catastrophically.

# --- BQ Helper ---
async def _execute_bq_query_async(query: str, query_params: Optional[List[Any]] = None) -> Optional[List[Dict]]:
    if not bq_client:
        logger.error("BigQuery client not available in _execute_bq_query_async.")
        return None
    try:
        log_query = query.split('\n')[0] # Log first line for brevity if multi-line
        logger.info(f"Attempting to execute BQ Query (first line): {log_query}...")
        
        job_config = None
        if query_params:
            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            logger.info(f"BQ Query executing with parameters: {query_params}")

        # Use asyncio.to_thread for blocking BQ call
        query_job = await asyncio.to_thread(bq_client.query, query, job_config=job_config)
        logger.info(f"BQ Query job {query_job.job_id} created. Waiting for results...")
        
        results_df = await asyncio.to_thread(query_job.to_dataframe)
        logger.info(f"BQ Query job {query_job.job_id} finished. Returned {len(results_df)} rows.")
        
        results_df = results_df.fillna('') # Handle NaN values before converting to dict
        return results_df.to_dict('records')

    except google_exceptions.NotFound:
        logger.warning(f"BQ resource not found during query execution for query (first line): {log_query}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error executing BQ query (first line): {log_query}. Error: {e}", exc_info=True)
        return None

# --- MCP Tools ---
@mcp.tool()
async def search_similar_images_by_text(query_text: str, top_k: int = 1, filter_image_type: str = "") -> str:
    """
    Performs vector search on the image embedding table using a text query.
    Returns a JSON string list of result dictionaries or an error JSON.
    """
    tool_call_id = os.urandom(4).hex() # Simple unique ID for this tool call for logging
    logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Called search_similar_images_by_text.")
    logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Input - query_text='{query_text}', top_k={top_k}, filter_image_type='{filter_image_type}'")

    if not bq_client:
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: BigQuery client not initialized.")
        return json.dumps({"error": "Server-side BQ service not available."})
    if not multimodal_embedding_model:
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Multimodal Embedding model not initialized.")
        return json.dumps({"error": "Server-side Embedding service not available."})
    if not query_text:
        logger.warning(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Query text is empty.")
        return json.dumps({"error": "Query text cannot be empty."})

    # 1. Generate query embedding
    query_embedding_list: Optional[List[float]] = None
    try:
        logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Generating embedding for text: '{query_text}'")
        # The SDK's get_embeddings is synchronous, so wrap with asyncio.to_thread
        embedding_response = await asyncio.to_thread(
            multimodal_embedding_model.get_embeddings,
            contextual_text=query_text,
            dimension=EMBEDDING_DIMENSIONALITY # Ensure this matches your table
        )
        query_embedding_list = embedding_response.text_embedding
        if not query_embedding_list:
            # This case should ideally be caught by an exception from the SDK if it fails
            logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: SDK returned no text embedding for '{query_text}'.")
            raise ValueError("SDK returned no text embedding.")
        logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Successfully generated embedding for '{query_text}'. Length: {len(query_embedding_list)}")
    except Exception as query_emb_err:
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Failed to get query embedding for '{query_text}'. Error: {query_emb_err}", exc_info=True)
        return json.dumps({"error": f"Failed to generate query embedding: {str(query_emb_err)}"})

    # Convert embedding list to string format for SQL query
    query_embedding_str = f"[{','.join(map(str, query_embedding_list))}]"

    # 2. Perform Vector Search
    options_str = ""
    logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: No explicit 'image_type' filter being applied by this tool in VECTOR_SEARCH OPTIONS.")


    # Construct the BQ VECTOR_SEARCH query
    # Ensuring BQ_FULL_EMBEDDING_TABLE_ID is correctly formatted (already done at global scope)
    vector_search_sql = f"""
    SELECT
        base.image_uri,
        base.image_type,
        base.entity_id,
        base.entity_name,
        distance
    FROM
        VECTOR_SEARCH(
            TABLE `{BQ_FULL_EMBEDDING_TABLE_ID}`,
            'embedding', /* The column containing embeddings */
            (SELECT {query_embedding_str} AS embedding), /* The query vector */
            top_k => {int(top_k)},
            distance_type => 'COSINE'{options_str} 
            /* Note: No semicolon needed inside the SQL string passed to BQ client */
        )
    """
    logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Constructed Vector Search SQL (first 200 chars): {vector_search_sql[:200]}...")
    # For full query debugging:
    # logger.debug(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Full Vector Search SQL:\n{vector_search_sql}")

    bq_results: Optional[List[Dict]] = None
    try:
        bq_results = await _execute_bq_query_async(vector_search_sql)
    except Exception as bq_exec_err: # Catch errors from the execution wrapper itself
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Exception during BQ query execution wrapper for '{query_text}'. Error: {bq_exec_err}", exc_info=True)
        return json.dumps({"error": f"Server error during BigQuery execution: {str(bq_exec_err)}"})


    if bq_results is None:
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: BigQuery query execution returned None (likely an error) for '{query_text}'. Check BQ helper logs.")
        return json.dumps({"error": "Error performing image vector search in BigQuery."})
    
    if not bq_results: # Empty list from BQ
        logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: No similar images found in BQ for query: '{query_text}' with filter '{filter_image_type}'.")
        return json.dumps([]) # Return empty list JSON string

    # Successfully got results from BQ
    logger.info(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Image vector search for '{query_text}' returned {len(bq_results)} results.")
    try:
        # Ensure results are JSON serializable (default=str handles common types like datetime)
        response_json = json.dumps(bq_results, default=str)
        return response_json
    except Exception as json_err:
        logger.error(f"IMAGE_EMB_MCP_TOOL [{tool_call_id}]: Failed to serialize BQ results to JSON for '{query_text}'. Error: {json_err}", exc_info=True)
        return json.dumps({"error": "Failed to serialize search results."})


if __name__ == "__main__":
    logger.info("Attempting to start Image Embedding MCP Server...")
    # Test client initializations before starting server if not done already
    if not all([bq_client, storage_client, multimodal_embedding_model]):
        logger.error("One or more critical clients failed to initialize. MCP server may not function correctly.")
    else:
        logger.info("All critical clients seem initialized. Starting MCP server.")
    mcp.run(transport="stdio")