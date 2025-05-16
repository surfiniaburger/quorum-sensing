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
import asyncio 
from datetime import datetime, UTC

from google.cloud import bigquery
from google.api_core.exceptions import NotFound, GoogleAPICallError # Import GoogleAPICallError
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "mlb_rag_data_2024")
BQ_RAG_TABLE_ID = os.getenv("BQ_RAG_TABLE_ID", "rag_documents")
BQ_PLAYS_TABLE_ID = os.getenv("BQ_PLAYS_TABLE_ID", "plays")
BQ_CRITIQUE_TABLE_ID = os.getenv("BQ_CRITIQUE_TABLE_ID", "agent_critiques") # Ensure this is correct
PLAYER_METADATA_TABLE_ID = os.getenv("PLAYER_METADATA_TABLE_ID", "mlb_player_metadata")

BQ_FULL_RAG_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_RAG_TABLE_ID}"
BQ_FULL_PLAYS_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_PLAYS_TABLE_ID}"
BQ_FULL_CRITIQUE_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_CRITIQUE_TABLE_ID}"
BQ_FULL_PLAYER_METADATA_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}"

VECTOR_COLUMN_RAG = "embedding"
VECTOR_COLUMN_CRITIQUE = "critique_embedding" # Ensure this column exists in agent_critiques table
CONTENT_COLUMNS_RAG = ["doc_id", "content", "doc_type", "game_id", "metadata"] 

VERTEX_EMB_MODEL_NAME = os.getenv("VERTEX_EMB_MODEL", "text-embedding-004")
EMBEDDING_DIMENSIONALITY = 768 
CRITIQUE_EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_CRITIQUE_EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"
RAG_EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"

DEFAULT_RAG_TOP_N = 5
DEFAULT_CRITIQUE_TOP_N = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("bq_vector_search_mcp") # Give it a distinct name

mcp = FastMCP("bq_search") # Ensure tool prefix matches (e.g. bq_search.search_past_critiques)

bq_client = None
emb_model = None
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    emb_model = TextEmbeddingModel.from_pretrained(VERTEX_EMB_MODEL_NAME)
    logger.info(f"BQ_MCP: BigQuery client and Embedding model ({VERTEX_EMB_MODEL_NAME}) initialized successfully.")
except Exception as e:
    logger.critical(f"BQ_MCP: CRITICAL - Failed to initialize BQ client or Embedding model: {e}", exc_info=True)
    # Server will start, but tools will fail.

async def _get_embeddings(texts: List[str], task_type: str) -> List[Optional[List[float]]]:
    if not emb_model:
        logger.error("BQ_MCP: Embedding model not initialized.")
        return [None] * len(texts)
    try:
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        embeddings_result = await asyncio.to_thread(emb_model.get_embeddings, inputs)
        return [e.values for e in embeddings_result]
    except Exception as e:
        logger.error(f"BQ_MCP: Error getting embeddings (task: {task_type}): {e}", exc_info=True)
        return [None] * len(texts)

async def _execute_bq_query_async(query: str, query_description: str = "BigQuery query") -> Optional[List[Dict]]:
    if not bq_client:
        logger.error(f"BQ_MCP: BigQuery client not initialized. Cannot execute {query_description}.")
        return None
    
    query_log_display = query if len(query) < 400 else query[:400] + "..." # Log more for debugging
    logger.info(f"BQ_MCP: Attempting to execute {query_description}: {query_log_display}")
    
    try:
        query_job = await asyncio.to_thread(bq_client.query, query)
        logger.info(f"BQ_MCP: Query job {query_job.job_id} created for {query_description}. Waiting for results...")
        
        # Wait for the job to complete and check for errors.
        # .result() will raise an exception if the job failed.
        await asyncio.to_thread(query_job.result) 
        
        # If query_job.result() did not raise, but there's an error_result (less common for SELECTs after .result())
        if query_job.error_result:
            logger.error(f"BQ_MCP: Query job {query_job.job_id} for '{query_description}' completed but reported an error: {query_job.error_result}")
            # Log detailed errors if available
            if query_job.errors:
                for job_error in query_job.errors:
                    logger.error(f"  - Reason: {job_error.get('reason', 'N/A')}, Message: {job_error.get('message', 'N/A')}, Location: {job_error.get('location', 'N/A')}")
            return None

        results_df = await asyncio.to_thread(query_job.to_dataframe)
        logger.info(f"BQ_MCP: {query_description} returned {len(results_df)} rows.")
        results_df = results_df.fillna('') 
        return results_df.to_dict('records')

    except NotFound as nf_error:
        logger.error(f"BQ_MCP: NotFound error for {query_description} (Query: {query_log_display}): {nf_error}", exc_info=True)
        return None
    except GoogleAPICallError as api_error: # Catch more specific BQ/Google API errors
        logger.error(f"BQ_MCP: GoogleAPICallError for {query_description} (Query: {query_log_display}): {api_error}", exc_info=True)
        return None
    except Exception as e: # Generic catch-all
        logger.error(f"BQ_MCP: Unexpected error executing {query_description} (Query: {query_log_display}): {e}", exc_info=True)
        return None

async def _insert_bq_rows_json_async(table_id: str, rows: List[Dict]) -> bool:
    # (Implementation as before, adding BQ_MCP to logs)
    if not bq_client:
        logger.error("BQ_MCP: BigQuery client not initialized for insert.")
        return False
    try:
        logger.info(f"BQ_MCP: Inserting {len(rows)} rows into {table_id}...")
        errors = await asyncio.to_thread(bq_client.insert_rows_json, table_id, rows)
        if not errors:
            logger.info(f"BQ_MCP: Successfully inserted {len(rows)} rows into {table_id}.")
            return True
        else:
            logger.error(f"BQ_MCP: Errors occurred during BQ insertion into {table_id}: {errors}")
            return False
    except Exception as e:
        logger.error(f"BQ_MCP: Error inserting rows into BQ table {table_id}: {e}", exc_info=True)
        return False

@mcp.tool()
async def search_past_critiques(task_text: str, game_pk_str: str = "", top_n: int = DEFAULT_CRITIQUE_TOP_N) -> str:
    """
    Performs vector search on past critiques.
    game_pk_str: The game_pk as a string. Leave empty or send "None" if not applicable.
    """
    logger.info(f"BQ_MCP: TOOL CALLED - search_past_critiques. Task: '{task_text[:60]}...', game_pk_str: '{game_pk_str}', top_n: {top_n}")

    if not bq_client or not emb_model:
        logger.error("BQ_MCP: search_past_critiques - BigQuery or Embedding service not available.")
        return json.dumps({"error": "BQ_MCP: BigQuery or Embedding service not available."})
    if not task_text:
        logger.warning("BQ_MCP: search_past_critiques - Task text cannot be empty.")
        return json.dumps({"error": "BQ_MCP: Task text cannot be empty for critique search."})

    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try:
            game_pk = int(game_pk_str)
        except ValueError:
            logger.warning(f"BQ_MCP: search_past_critiques - Invalid game_pk_str '{game_pk_str}'. Proceeding without game_pk filter.")
    
    logger.info(f"BQ_MCP: search_past_critiques - Parsed GamePK: {game_pk}")

    task_embedding_list_of_list = await _get_embeddings([task_text], task_type=QUERY_CRITIQUE_EMBEDDING_TASK_TYPE)
    if not task_embedding_list_of_list or not task_embedding_list_of_list[0]:
        logger.error("BQ_MCP: search_past_critiques - Could not generate embedding for the task text.")
        return json.dumps({"error": "BQ_MCP: Could not generate embedding for the task text."})
    task_embedding_str = f"[{','.join(map(str, task_embedding_list_of_list[0]))}]"
    logger.debug(f"BQ_MCP: search_past_critiques - Task embedding generated (first 10 chars): {task_embedding_str[:10]}...")

    initial_top_k = top_n * 2 + 5 
    
    # IMPORTANT: The WHERE clause needs to apply to the output of VECTOR_SEARCH.
    # If base.game_pk is a column directly in your agent_critiques table, this structure is okay.
    # The 'base' alias in VECTOR_SEARCH refers to the original table's columns.
    game_filter_sql_condition = f"AND base.game_pk = {game_pk}" if game_pk is not None else ""

    vector_search_query = f"""
    SELECT
        base.critique_text, 
        base.game_pk, -- For verification
        distance
    FROM
        VECTOR_SEARCH(
            TABLE `{BQ_FULL_CRITIQUE_TABLE_ID}`,
            '{VECTOR_COLUMN_CRITIQUE}',
            (SELECT {task_embedding_str} AS {VECTOR_COLUMN_CRITIQUE}), /* Alias must match column name */
            top_k => {initial_top_k},
            distance_type => 'COSINE'
        )
    WHERE 1=1 {game_filter_sql_condition} /* This filters results from VECTOR_SEARCH */
    ORDER BY distance ASC
    LIMIT {top_n}
    """
    
    rows = await _execute_bq_query_async(vector_search_query, query_description="Past Critiques Vector Search")

    if rows is None:
        logger.error("BQ_MCP: search_past_critiques - _execute_bq_query_async returned None (error occurred).")
        return json.dumps({"error": "BQ_MCP: Error performing critique vector search in BigQuery."}) # More specific
    if not rows:
        logger.info(f"BQ_MCP: search_past_critiques - No relevant past critiques found for task: '{task_text[:30]}...'. Query returned empty.")
        return json.dumps([]) 

    # The `rows` from _execute_bq_query_async are already dicts.
    # The `VECTOR_SEARCH` output, when fields are selected from `base.`, typically flattens them.
    # So, we might not have a nested `base` structure in `rows` here.
    results_list = []
    try:
        for row_dict in rows:
            # If 'base' is NOT a nested struct in the output of THIS query
            # (because we selected base.critique_text directly)
            critique_text = row_dict.get('critique_text') # Access directly
            if critique_text:
                results_list.append(critique_text)
            else:
                logger.warning(f"BQ_MCP: search_past_critiques - Row found without 'critique_text': {row_dict}")
        
        logger.info(f"BQ_MCP: search_past_critiques - Successfully returning {len(results_list)} critiques.")
        return json.dumps(results_list)
    except Exception as e:
        logger.error(f"BQ_MCP: search_past_critiques - Error processing BQ results: {e}", exc_info=True)
        return json.dumps({"error": "BQ_MCP: Error processing critique search results after fetch."})


# ... (rest of your bq_vector_search_server.py tools: search_rag_documents, store_new_critique, etc.)
# Ensure other tools also use the enhanced _execute_bq_query_async if they call it.

# Example of adapting another tool (search_rag_documents)
@mcp.tool()
async def search_rag_documents(query_text: str, game_pk_str: str = "", top_n: int = DEFAULT_RAG_TOP_N) -> str:
    logger.info(f"BQ_MCP: TOOL CALLED - search_rag_documents. Query: '{query_text[:60]}...', game_pk_str: '{game_pk_str}', top_n: {top_n}")
    if not bq_client or not emb_model: # Basic checks
        return json.dumps({"error": "BQ_MCP: BigQuery or Embedding service not available."})
    if not query_text:
        return json.dumps({"error": "BQ_MCP: Query text cannot be empty."})

    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try: game_pk = int(game_pk_str)
        except ValueError: logger.warning(f"BQ_MCP: search_rag_documents - Invalid game_pk_str '{game_pk_str}'.")
    
    logger.info(f"BQ_MCP: search_rag_documents - Parsed GamePK: {game_pk}")

    query_embedding_list_of_list = await _get_embeddings([query_text], task_type=RAG_EMBEDDING_TASK_TYPE)
    if not query_embedding_list_of_list or not query_embedding_list_of_list[0]:
        return json.dumps({"error": "BQ_MCP: Could not generate embedding for RAG query."})
    query_embedding_str = f"[{','.join(map(str, query_embedding_list_of_list[0]))}]"
    logger.debug(f"BQ_MCP: search_rag_documents - RAG query embedding generated.")

    initial_top_k = top_n * 5 + 10
    
    # The VECTOR_SEARCH query selects the 'base' struct and 'distance'
    # The Python code then filters the 'base' struct's 'game_id' field.
    vector_search_query = f"""
    SELECT
        base,  -- Select the entire base row as a STRUCT
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
    rows = await _execute_bq_query_async(vector_search_query, query_description="RAG Documents Vector Search")

    if rows is None:
        return json.dumps({"error": "BQ_MCP: Error performing RAG document search."})
    if not rows:
        logger.info(f"BQ_MCP: search_rag_documents - No RAG document candidates found for: '{query_text[:30]}...'.")
        return json.dumps([])

    results_list = []
    try:
        for row_dict in rows:
            nested_base_data = row_dict.get('base') # This 'base' is the STRUCT from BQ RAG table
            distance = row_dict.get('distance')

            if isinstance(nested_base_data, dict):
                # Access fields within the 'base' STRUCT (which is the original table row)
                row_game_pk_val = nested_base_data.get('game_id') # Assuming 'game_id' is a field in your BQ_RAG_TABLE_ID
                content = nested_base_data.get('content')

                if game_pk is not None: # Filter by parsed game_pk
                    try:
                        if row_game_pk_val is not None and int(row_game_pk_val) == game_pk:
                            if content: results_list.append({"content": content, "distance": distance})
                    except (TypeError, ValueError) as e:
                        logger.debug(f"BQ_MCP: search_rag_documents - Could not compare game_pk for filtering: {row_game_pk_val} vs {game_pk}. Error: {e}")
                        continue 
                elif content: 
                    results_list.append({"content": content, "distance": distance})
            else:
                logger.warning(f"BQ_MCP: search_rag_documents - Row found where 'base' is not a dict: {type(nested_base_data)}")
        
        results_list.sort(key=lambda x: x.get('distance', float('inf')))
        final_contents = [item['content'] for item in results_list[:top_n]]
        logger.info(f"BQ_MCP: search_rag_documents - Returning {len(final_contents)} documents.")
        return json.dumps(final_contents)
    except Exception as e:
        logger.error(f"BQ_MCP: search_rag_documents - Error processing RAG search results: {e}", exc_info=True)
        return json.dumps({"error": "BQ_MCP: Error processing RAG search results."})


@mcp.tool()
async def store_new_critique(critique_text: str, task_text: str, game_pk_str: str = "", revision_number_str: str = "") -> str:
    logger.info(f"BQ_MCP: TOOL CALLED - store_new_critique.")
    logger.info(f"  - Received critique_text (first 100 chars): '{critique_text[:100]}...'")
    logger.info(f"  - Received task_text (first 100 chars): '{task_text[:100]}...'")
    logger.info(f"  - Received game_pk_str: '{game_pk_str}'")
    logger.info(f"  - Received revision_number_str: '{revision_number_str}'")

    if not bq_client or not emb_model:
        logger.error("BQ_MCP: store_new_critique - BigQuery or Embedding service not available.")
        return json.dumps({"error": "BQ_MCP: BigQuery or Embedding service not available for storing critique."})
    if not critique_text or not task_text:
        logger.warning("BQ_MCP: store_new_critique - Critique text or task text is empty.")
        return json.dumps({"error": "BQ_MCP: Critique text and task text are required for storing critique."})

    game_pk: Optional[int] = None
    if game_pk_str and game_pk_str.lower() != "none" and game_pk_str.strip() != "":
        try:
            game_pk = int(game_pk_str)
            logger.info(f"  - Parsed game_pk as integer: {game_pk}")
        except ValueError:
            logger.warning(f"BQ_MCP: store_new_critique - Invalid game_pk_str '{game_pk_str}'. Storing with game_pk as NULL.")
            game_pk = None # Explicitly set to None if parsing fails
    else:
        logger.warning(f"BQ_MCP: store_new_critique - game_pk_str is empty or 'none'. Storing with game_pk as NULL.")
        game_pk = None


    revision_number: Optional[int] = None
    if revision_number_str and revision_number_str.lower() != "none" and revision_number_str.strip() != "":
        try:
            revision_number = int(revision_number_str)
            logger.info(f"  - Parsed revision_number as integer: {revision_number}")
        except ValueError:
            logger.warning(f"BQ_MCP: store_new_critique - Invalid revision_number_str '{revision_number_str}'. Storing with revision_number as NULL.")
            revision_number = None
    else:
        logger.info(f"BQ_MCP: store_new_critique - revision_number_str is empty or 'none'. Storing with revision_number as NULL.")
        revision_number = None
    
    logger.info(f"BQ_MCP: store_new_critique - Attempting to store with parsed game_pk: {game_pk}, revision: {revision_number}")

    critique_embedding_list_of_list = await _get_embeddings([critique_text], task_type=CRITIQUE_EMBEDDING_TASK_TYPE)
    
    critique_embedding: Optional[List[float]] = None
    if critique_embedding_list_of_list and critique_embedding_list_of_list[0] is not None:
        critique_embedding = critique_embedding_list_of_list[0]
        logger.info(f"BQ_MCP: store_new_critique - Critique embedding generated successfully (Dimensions: {len(critique_embedding)}).")
    else:
        logger.error("BQ_MCP: store_new_critique - Could not generate embedding for the critique text. Storing critique without embedding.")
        # Depending on table schema, storing None for a REPEATED field might be an issue,
        # or BQ might convert it to an empty array. Or it might fail insertion if the column is REQUIRED.
        # For now, we'll try to store it as None and see.
        # If your critique_embedding column in BQ is MODE="REPEATED" but not "NULLABLE" and also not "REQUIRED"
        # this could be an issue. A repeated field is implicitly nullable and defaults to empty array.

    task_hash = hashlib.sha256(task_text.encode()).hexdigest()
    critique_id = str(uuid.uuid4())
    timestamp_now = datetime.now(UTC).isoformat()
    
    row_to_insert = {
        "critique_id": critique_id,
        "task_hash": task_hash,
        "game_pk": game_pk, 
        "revision_number": revision_number, 
        "critique_text": critique_text,
        "critique_embedding": critique_embedding, # This will be None if embedding failed
        "timestamp": timestamp_now,
    }

    logger.info(f"BQ_MCP: store_new_critique - Prepared row for insertion: {json.dumps(row_to_insert, default=str)}") # Log the row

    success = await _insert_bq_rows_json_async(BQ_FULL_CRITIQUE_TABLE_ID, [row_to_insert])

    if success:
        logger.info(f"BQ_MCP: store_new_critique - Critique {critique_id} (game_pk: {game_pk}) insertion reported as successful by BQ client.")
        return json.dumps({"status": "success", "message": "Critique stored successfully.", "critique_id_stored": critique_id, "stored_game_pk": game_pk})
    else:
        logger.error(f"BQ_MCP: store_new_critique - BQ client reported errors storing critique for task hash {task_hash} (game_pk: {game_pk}).")
        return json.dumps({"error": "BQ_MCP: Failed to store critique in BigQuery (BQ client reported errors).", "attempted_game_pk": game_pk})

# --- Other tools like get_game_summary_metadata, get_filtered_play_by_play, get_player_metadata_lookup
# should also use the updated _execute_bq_query_async if they call it.

@mcp.tool()
async def get_game_summary_metadata(game_pk: int) -> str:
    # ... (use _execute_bq_query_async(query, "Game Summary Metadata Fetch")) ...
    logger.info(f"BQ_MCP: TOOL CALLED - get_game_summary_metadata for game_pk: {game_pk}")
    if not bq_client: return json.dumps({"error": "BQ_MCP: BigQuery service not available."})
    if not game_pk: return json.dumps({"error": "BQ_MCP: game_pk cannot be empty."})

    query = f"""
    SELECT game_id, doc_type, metadata, last_updated
    FROM `{BQ_FULL_RAG_TABLE_ID}`
    WHERE game_id = {int(game_pk)} AND doc_type = 'game_summary' LIMIT 1
    """
    rows = await _execute_bq_query_async(query, query_description="Game Summary Metadata Fetch")
    # ... (rest of the logic)
    if rows is None:
        return json.dumps({"error": f"BQ_MCP: Error fetching metadata for game {game_pk}."})
    if not rows:
        return json.dumps({"error": f"BQ_MCP: No game summary metadata found for game_pk {game_pk}."})
    # ... (metadata parsing logic)
    data_dict = rows[0]
    metadata = data_dict.get('metadata')
    if isinstance(metadata, str):
        try:
            metadata_dict = json.loads(metadata)
            return json.dumps(metadata_dict)
        except json.JSONDecodeError:
            return json.dumps({"metadata_raw_string": metadata}) # Return raw if parse fails
    elif isinstance(metadata, dict):
         return json.dumps(metadata)
    return json.dumps({"error": "BQ_MCP: Metadata found but in unexpected format."})


@mcp.tool()
async def get_filtered_play_by_play(game_pk: int, filter_criteria_sql: str = "1=1", limit: int = 500) -> str:
    # ... (use _execute_bq_query_async(query, "Filtered Play-by-Play Fetch")) ...
    logger.info(f"BQ_MCP: TOOL CALLED - get_filtered_play_by_play for game_pk: {game_pk}")
    if not bq_client: return json.dumps({"error": "BQ_MCP: BigQuery service not available."})
    # ... (safety checks for filter_criteria_sql and limit)
    safe_criteria = re.sub(r";|--|\/\*|\*\/", "", filter_criteria_sql) if filter_criteria_sql else "1=1"
    safe_limit = max(1, int(limit))
    query = f"""
    SELECT
        play_index, inning, halfInning, event_type, description,
        rbi, score_change, batting_team_id, fielding_team_id,
        batter_id, pitcher_id, start_time, end_time,
        pitch_data, hit_data, runners_after
    FROM `{BQ_FULL_PLAYS_TABLE_ID}`
    WHERE game_pk = {int(game_pk)} AND ({safe_criteria})
    ORDER BY play_index
    LIMIT {safe_limit}
    """
    rows = await _execute_bq_query_async(query, query_description="Filtered Play-by-Play Fetch")
    # ... (rest of the logic, JSON parsing for records)
    if rows is None:
        return json.dumps({"error": f"BQ_MCP: Error fetching play-by-play for game {game_pk}."})
    if not rows:
        return json.dumps([])
    for record in rows: # JSON parsing for certain columns
        for col in ['pitch_data', 'hit_data', 'runners_after']:
            if col in record and isinstance(record[col], str):
                try: record[col] = json.loads(record[col])
                except json.JSONDecodeError: record[col] = None
    return json.dumps(rows, default=str)


@mcp.tool()
async def get_player_metadata_lookup() -> str:
    # ... (use _execute_bq_query_async(query, "Player Metadata Lookup")) ...
    logger.info("BQ_MCP: TOOL CALLED - get_player_metadata_lookup")
    if not bq_client: return json.dumps({"error": "BQ_MCP: BigQuery service not available."})
    query = f"SELECT player_id, player_name FROM `{BQ_FULL_PLAYER_METADATA_TABLE_ID}`"
    rows = await _execute_bq_query_async(query, query_description="Player Metadata Lookup")
    # ... (rest of the logic)
    if rows is None:
        return json.dumps({"error": "BQ_MCP: Error fetching player metadata."})
    if not rows:
        return json.dumps({})
    lookup_dict = {}
    for row in rows:
        try:
            player_id = int(row['player_id'])
            player_name = row.get('player_name')
            if player_name: lookup_dict[player_id] = player_name
        except (TypeError, ValueError): continue # Skip invalid
    return json.dumps(lookup_dict)

if __name__ == "__main__":
    logger.info("BQ_MCP: Starting BigQuery Vector Search MCP Server...")
    mcp.run(transport="stdio")