# mcp_server/bq_vector_search_server.py
import logging
import json
import os
import re
import pandas as pd
import hashlib
import uuid
import time
from typing import Any, Dict, Optional, List
import asyncio # For async execution
from datetime import datetime, UTC

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput # If embedding here
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "mlb_rag_data_2024")
BQ_RAG_TABLE_ID = os.getenv("BQ_RAG_TABLE_ID", "rag_documents")
BQ_PLAYS_TABLE_ID = os.getenv("BQ_PLAYS_TABLE_ID", "plays")
BQ_CRITIQUE_TABLE_ID = os.getenv("BQ_CRITIQUE_TABLE_ID", "agent_critiques")
PLAYER_METADATA_TABLE_ID = os.getenv("PLAYER_METADATA_TABLE_ID", "mlb_player_metadata")

BQ_FULL_RAG_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_RAG_TABLE_ID}"
BQ_FULL_PLAYS_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_PLAYS_TABLE_ID}"
BQ_FULL_CRITIQUE_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_CRITIQUE_TABLE_ID}"
BQ_FULL_PLAYER_METADATA_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}"

VECTOR_COLUMN_RAG = "embedding"
VECTOR_COLUMN_CRITIQUE = "critique_embedding"
CONTENT_COLUMNS_RAG = ["doc_id", "content", "doc_type", "game_id", "metadata"] # Example, adjust as needed

VERTEX_EMB_MODEL_NAME = os.getenv("VERTEX_EMB_MODEL", "text-embedding-004")
EMBEDDING_DIMENSIONALITY = 768 # Ensure consistency
CRITIQUE_EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_CRITIQUE_EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"
RAG_EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"

# Limits for vector search results (can be overridden by tool args)
DEFAULT_RAG_TOP_N = 5
DEFAULT_CRITIQUE_TOP_N = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("bq_vector_search")

# --- Initialize Clients ---
bq_client = None
emb_model = None
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    # Initialize embedding model if doing embeddings in this server
    emb_model = TextEmbeddingModel.from_pretrained(VERTEX_EMB_MODEL_NAME)
    logger.info(f"BigQuery client and Embedding model ({VERTEX_EMB_MODEL_NAME}) initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize BQ client or Embedding model: {e}", exc_info=True)
    # Allow server to start but tools will fail gracefully

# --- Internal Helper to get embeddings ---
async def _get_embeddings(texts: List[str], task_type: str) -> List[Optional[List[float]]]:
    if not emb_model:
        logger.error("Embedding model not initialized.")
        return [None] * len(texts)
    try:
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        # Using asyncio.to_thread for the blocking SDK call
        embeddings_result = await asyncio.to_thread(emb_model.get_embeddings, inputs)
        return [e.values for e in embeddings_result]
    except Exception as e:
        logger.error(f"Error getting embeddings (task: {task_type}): {e}", exc_info=True)
        return [None] * len(texts)

# --- Internal Helper for BQ Execution ---
async def _execute_bq_query_async(query: str) -> Optional[List[Dict]]:
    if not bq_client:
        logger.error("BigQuery client not initialized.")
        return None
    try:
        logger.info(f"Executing BQ Query (async wrapper): {query[:200]}...")
        # Use asyncio.to_thread for the synchronous BigQuery client library calls
        query_job = await asyncio.to_thread(bq_client.query, query)
        results_df = await asyncio.to_thread(query_job.to_dataframe)
        logger.info(f"BQ Query returned {len(results_df)} rows.")
        # Handle NaNs which cause JSON serialization errors
        results_df = results_df.fillna('') # Replace NaN with empty string or choose another placeholder
        return results_df.to_dict('records')
    except NotFound:
        logger.warning(f"BQ resource not found for query: {query[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error executing BQ query: {query[:200]}... Error: {e}", exc_info=True)
        return None

async def _insert_bq_rows_json_async(table_id: str, rows: List[Dict]) -> bool:
    if not bq_client:
        logger.error("BigQuery client not initialized.")
        return False
    try:
        logger.info(f"Inserting {len(rows)} rows into {table_id}...")
        # Use asyncio.to_thread for the synchronous BigQuery client library call
        errors = await asyncio.to_thread(bq_client.insert_rows_json, table_id, rows)
        if not errors:
            logger.info(f"Successfully inserted {len(rows)} rows.")
            return True
        else:
            logger.error(f"Errors occurred during BQ insertion: {errors}")
            return False
    except Exception as e:
        logger.error(f"Error inserting rows into BQ table {table_id}: {e}", exc_info=True)
        return False

# --- MCP Tools ---

@mcp.tool()
async def search_rag_documents(query_text: str, game_pk_str: str = "", top_n: int = DEFAULT_RAG_TOP_N) -> str:
    """
    Performs vector search on the RAG documents table (summaries, snippets).
    game_pk_str: The game_pk as a string. Leave empty or send "None" if not applicable.
    Returns a JSON string list of matching document contents.
    """
    if not bq_client or not emb_model:
        return json.dumps({"error": "BigQuery or Embedding service not available."})
    if not query_text:
        return json.dumps({"error": "Query text cannot be empty."})

    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try:
            game_pk = int(game_pk_str)
        except ValueError:
            logger.warning(f"BQ_MCP: Invalid game_pk_str '{game_pk_str}' received. Proceeding without game_pk filter.")
            # Return an error or proceed without game_pk
            # return json.dumps({"error": f"Invalid game_pk_str: {game_pk_str}. Must be an integer."})


    logger.info(f"BQ_MCP: search_rag_documents - Query: '{query_text[:50]}...', GamePK: {game_pk} (from str: '{game_pk_str}'), TopN: {top_n}")

    # ... rest of your existing logic using the parsed `game_pk` variable ...
    # Ensure the vector_search_query construction correctly handles `game_pk` being None

    # Example of how you might use the parsed game_pk:
    query_embedding_list_of_list = await _get_embeddings([query_text], task_type=RAG_EMBEDDING_TASK_TYPE)
    if not query_embedding_list_of_list or not query_embedding_list_of_list[0]:
        return json.dumps({"error": "Could not generate embedding for the RAG query."})
    query_embedding_str = f"[{','.join(map(str, query_embedding_list_of_list[0]))}]"

    initial_top_k = top_n * 5 + 10
    
    # Conditional game_pk filtering in the main query or post-filtering
    # For VECTOR_SEARCH, it might be better to fetch more and filter in Python if game_pk is used.
    # Or, if your BQ table's `game_id` column in the `base` struct is indexed, you might add a WHERE clause.
    # The current approach in search_rag_documents filters in Python, which is fine.

    vector_search_query = f"""
    SELECT
        base,
        distance
    FROM
        VECTOR_SEARCH(
            TABLE `{BQ_FULL_RAG_TABLE_ID}`,
            '{VECTOR_COLUMN_RAG}',
            (SELECT {query_embedding_str} AS {VECTOR_COLUMN_RAG}),
            top_k => {initial_top_k},
            distance_type => 'COSINE'
        )
    ORDER BY
        distance ASC
    LIMIT {initial_top_k}
    """
    logger.info(f"BQ_MCP: Executing RAG vector search: {vector_search_query[:300]}...")
    rows = await _execute_bq_query_async(vector_search_query)

    if rows is None:
        return json.dumps({"error": "Error performing RAG document search."})
    if not rows:
        logger.info(f"No RAG document candidates found for: '{query_text[:30]}...'.")
        return json.dumps([])

    results_list = []
    try:
        for row_dict in rows:
            nested_base_data = row_dict.get('base') # This is the STRUCT from BQ
            distance = row_dict.get('distance')

            if isinstance(nested_base_data, dict):
                 # Access fields within the 'base' STRUCT
                 row_game_pk_val = nested_base_data.get('game_id') # Assuming 'game_id' is a field in your base table
                 content = nested_base_data.get('content')

                 # Apply game_pk filter if game_pk (the parsed int) is available
                 if game_pk is not None:
                     try:
                         if row_game_pk_val is not None and int(row_game_pk_val) == game_pk:
                             if content: results_list.append({"content": content, "distance": distance})
                     except (TypeError, ValueError) as e:
                         logger.debug(f"Could not compare game_pk for filtering: {row_game_pk_val} vs {game_pk}. Error: {e}")
                         continue 
                 elif content: 
                     results_list.append({"content": content, "distance": distance})
            # ...
        results_list.sort(key=lambda x: x.get('distance', float('inf')))
        final_contents = [item['content'] for item in results_list[:top_n]]

        logger.info(f"BQ_MCP: RAG search returning {len(final_contents)} documents.")
        return json.dumps(final_contents)

    except Exception as e:
        logger.error(f"BQ_MCP: Error processing RAG search results: {e}", exc_info=True)
        return json.dumps({"error": "Error processing RAG search results."})

@mcp.tool()
async def get_game_summary_metadata(game_pk: int) -> str:
    """
    Fetches the game summary metadata for a given game_pk.
    Returns a JSON string of the metadata dictionary or an error object.
    """
    if not bq_client: return json.dumps({"error": "BigQuery service not available."})
    if not game_pk: return json.dumps({"error": "game_pk cannot be empty."})

    logger.info(f"BQ_MCP: get_game_summary_metadata - GamePK: {game_pk}")
    query = f"""
    SELECT game_id, doc_type, metadata, last_updated
    FROM `{BQ_FULL_RAG_TABLE_ID}`
    WHERE game_id = {int(game_pk)} AND doc_type = 'game_summary' LIMIT 1
    """
    rows = await _execute_bq_query_async(query)

    if rows is None:
        return json.dumps({"error": f"Error fetching metadata for game {game_pk}."})
    if not rows:
        return json.dumps({"error": f"No game summary metadata found for game_pk {game_pk}."})

    data_dict = rows[0]
    metadata = data_dict.get('metadata')

    # Parse metadata if it's a JSON string
    if isinstance(metadata, str):
        try:
            metadata_dict = json.loads(metadata)
            logger.info(f"BQ_MCP: Successfully fetched and parsed metadata for game {game_pk}.")
            return json.dumps(metadata_dict)
        except json.JSONDecodeError:
            logger.warning(f"BQ_MCP: Could not parse metadata JSON for game {game_pk}. Returning raw string.")
            # Return the raw string if parsing fails but data was found
            return json.dumps({"metadata_raw_string": metadata})
    elif isinstance(metadata, dict): # Already parsed by BQ driver?
         logger.info(f"BQ_MCP: Successfully fetched metadata for game {game_pk} (already dict).")
         return json.dumps(metadata)
    else:
         logger.warning(f"BQ_MCP: Metadata for game {game_pk} is not a string or dict. Type: {type(metadata)}")
         return json.dumps({"error": "Metadata found but in unexpected format."})


@mcp.tool()
async def get_filtered_play_by_play(game_pk: int, filter_criteria_sql: str = "1=1", limit: int = 500) -> str:
    """
    Fetches structured play-by-play data from the plays table, applying SQL filter criteria.
    Be cautious with filter_criteria_sql input. Returns a JSON string list of play dicts or an error.
    """
    if not bq_client: return json.dumps({"error": "BigQuery service not available."})
    if not game_pk: return json.dumps({"error": "game_pk cannot be empty."})

    logger.info(f"BQ_MCP: get_filtered_play_by_play - GamePK: {game_pk}, Filter: '{filter_criteria_sql}', Limit: {limit}")

    # Basic safety check - more robust validation/parameterization recommended for production
    # This is still vulnerable; prefer structured input if possible.
    safe_criteria = re.sub(r";|--|\/\*|\*\/", "", filter_criteria_sql) if filter_criteria_sql else "1=1"
    if safe_criteria != filter_criteria_sql:
        logger.warning(f"Potentially unsafe characters removed from filter criteria. Original: '{filter_criteria_sql}'")

    safe_limit = max(1, int(limit))

    # Select specific columns to avoid overly large responses
    query = f"""
    SELECT
        play_index, inning, halfInning, event_type, description,
        rbi, score_change, batting_team_id, fielding_team_id,
        batter_id, pitcher_id, start_time, end_time,
        pitch_data, hit_data, runners_after -- Include JSON columns if needed
    FROM `{BQ_FULL_PLAYS_TABLE_ID}`
    WHERE game_pk = {int(game_pk)} AND ({safe_criteria})
    ORDER BY play_index
    LIMIT {safe_limit}
    """
    rows = await _execute_bq_query_async(query)

    if rows is None:
        return json.dumps({"error": f"Error fetching play-by-play for game {game_pk}."})
    if not rows:
        logger.info(f"No plays found for game {game_pk} with filter '{safe_criteria}'.")
        return json.dumps([]) # Return empty list

    # Parse JSON columns if they exist in the results
    for record in rows:
        for col in ['pitch_data', 'hit_data', 'runners_after']:
            if col in record and isinstance(record[col], str):
                try:
                    record[col] = json.loads(record[col])
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse JSON in column '{col}' for play_index {record.get('play_index')}")
                    record[col] = None # Or keep raw string

    logger.info(f"BQ_MCP: Successfully fetched {len(rows)} plays.")
    return json.dumps(rows, default=str) # Use default=str for datetime/other types

@mcp.tool()
async def search_past_critiques(task_text: str, game_pk_str: str = "", top_n: int = DEFAULT_CRITIQUE_TOP_N) -> str:
    """
    Performs vector search on past critiques.
    game_pk_str: The game_pk as a string. Leave empty or send "None" if not applicable.
    """
    # Similar conversion for game_pk_str to Optional[int] game_pk
    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try:
            game_pk = int(game_pk_str)
        except ValueError:
            logger.warning(f"BQ_MCP: Invalid game_pk_str '{game_pk_str}' in search_past_critiques. Proceeding without game_pk filter.")
    
    logger.info(f"BQ_MCP: search_past_critiques - Task: '{task_text[:50]}...', GamePK: {game_pk} (from str: '{game_pk_str}'), TopN: {top_n}")
    # ... rest of the logic using the parsed game_pk ...
    if not bq_client or not emb_model:
        return json.dumps({"error": "BigQuery or Embedding service not available."})
    if not task_text:
        return json.dumps({"error": "Task text cannot be empty for critique search."})

    task_embedding_list_of_list = await _get_embeddings([task_text], task_type=QUERY_CRITIQUE_EMBEDDING_TASK_TYPE)
    if not task_embedding_list_of_list or not task_embedding_list_of_list[0]:
        return json.dumps({"error": "Could not generate embedding for the task text."})
    task_embedding_str = f"[{','.join(map(str, task_embedding_list_of_list[0]))}]"

    initial_top_k = top_n * 2 + 5
    # Construct the game_filter_clause based on the *parsed* integer game_pk
    game_filter_clause = f"AND base.game_pk = {game_pk}" if game_pk is not None else ""

    vector_search_query = f"""
    SELECT
        base.critique_text, 
        distance
    FROM
        VECTOR_SEARCH(
            TABLE `{BQ_FULL_CRITIQUE_TABLE_ID}`,
            '{VECTOR_COLUMN_CRITIQUE}',
            (SELECT {task_embedding_str} AS {VECTOR_COLUMN_CRITIQUE}),
            top_k => {initial_top_k},
            distance_type => 'COSINE'
        )
    WHERE 1=1 {game_filter_clause}
    ORDER BY distance ASC
    LIMIT {top_n}
    """
    # ... (execute query and process results) ...
    logger.info(f"BQ_MCP: Executing critique vector search: {vector_search_query[:300]}...")
    rows = await _execute_bq_query_async(vector_search_query)

    if rows is None: return json.dumps({"error": "Error performing critique search."})
    if not rows:
        logger.info(f"No relevant past critiques found for task: '{task_text[:30]}...'.")
        return json.dumps([])

    results_list = []
    try:
        for row_dict in rows:
             nested_base_data = row_dict.get('base')
             if isinstance(nested_base_data, dict):
                  critique_text = nested_base_data.get('critique_text')
                  if critique_text: results_list.append(critique_text)
             elif 'critique_text' in row_dict: 
                  if row_dict['critique_text']: results_list.append(row_dict['critique_text'])
        logger.info(f"BQ_MCP: Critique search returning {len(results_list)} critiques.")
        return json.dumps(results_list)
    except Exception as e:
        logger.error(f"BQ_MCP: Error processing critique search results: {e}", exc_info=True)
        return json.dumps({"error": "Error processing critique search results."})


@mcp.tool()
async def store_new_critique(critique_text: str, task_text: str, game_pk_str: str = "", revision_number_str: str = "") -> str:
    """
    Stores a new critique. game_pk_str and revision_number_str are optional.
    """
    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try:
            game_pk = int(game_pk_str)
        except ValueError: # Handle error or log
            logger.warning(f"Invalid game_pk_str '{game_pk_str}' in store_new_critique.")


    revision_number: Optional[int] = None
    if revision_number_str and revision_number_str.lower() != "none" and revision_number_str.strip() != "":
        try:
            revision_number = int(revision_number_str)
        except ValueError: # Handle error or log
            logger.warning(f"Invalid revision_number_str '{revision_number_str}' in store_new_critique.")

    logger.info(f"BQ_MCP: store_new_critique - Storing critique for task '{task_text[:50]}...', game_pk: {game_pk}, revision: {revision_number}")
    # ... rest of your existing logic using the parsed game_pk and revision_number ...
    if not bq_client or not emb_model:
        return json.dumps({"error": "BigQuery or Embedding service not available."})
    if not critique_text or not task_text:
        return json.dumps({"error": "Critique text and task text are required."})

    critique_embedding_list_of_list = await _get_embeddings([critique_text], task_type=CRITIQUE_EMBEDDING_TASK_TYPE)
    if not critique_embedding_list_of_list or not critique_embedding_list_of_list[0]:
        return json.dumps({"error": "Could not generate embedding for the critique text."})
    critique_embedding = critique_embedding_list_of_list[0]

    task_hash = hashlib.sha256(task_text.encode()).hexdigest()
    critique_id = str(uuid.uuid4())
    timestamp_now = datetime.now(UTC).isoformat()
    row_to_insert = {
        "critique_id": critique_id,
        "task_hash": task_hash,
        "game_pk": game_pk, # Use the parsed Optional[int] value
        "revision_number": revision_number, # Use the parsed Optional[int] value
        "critique_text": critique_text,
        "critique_embedding": critique_embedding,
        "timestamp": timestamp_now,
    }
    success = await _insert_bq_rows_json_async(BQ_FULL_CRITIQUE_TABLE_ID, [row_to_insert])
    if success:
        logger.info(f"BQ_MCP: Critique {critique_id} stored successfully.")
        return json.dumps({"status": "success", "critique_id": critique_id})
    else:
        logger.error(f"BQ_MCP: Failed to store critique for task hash {task_hash}.")
        return json.dumps({"error": "Failed to store critique in BigQuery."})
    
    
@mcp.tool()
async def get_player_metadata_lookup() -> str:
    """
    Loads the player ID to player name mapping from BigQuery.
    Returns JSON string of the lookup dictionary or an error object.
    """
    if not bq_client: return json.dumps({"error": "BigQuery service not available."})

    logger.info("BQ_MCP: get_player_metadata_lookup - Loading player metadata...")
    query = f"SELECT player_id, player_name FROM `{BQ_FULL_PLAYER_METADATA_TABLE_ID}`"
    rows = await _execute_bq_query_async(query)

    if rows is None:
        return json.dumps({"error": "Error fetching player metadata."})
    if not rows:
        logger.warning("BQ_MCP: Player metadata table appears empty.")
        return json.dumps({}) # Return empty dict

    lookup_dict = {}
    count = 0
    for row in rows:
        try:
            player_id = int(row['player_id'])
            player_name = row.get('player_name')
            if player_name: # Only add if name exists
                lookup_dict[player_id] = player_name
                count += 1
        except (TypeError, ValueError):
            logger.warning(f"BQ_MCP: Skipping invalid player_id format in metadata: {row.get('player_id')}")
            continue

    logger.info(f"BQ_MCP: Loaded {count} player names into lookup dictionary.")
    return json.dumps(lookup_dict)


if __name__ == "__main__":
    logger.info("Starting BigQuery Vector Search MCP Server...")
    # Optional: Run setup function here if needed on startup (usually done offline)
    # setup_critique_storage(...)
    mcp.run(transport="stdio")