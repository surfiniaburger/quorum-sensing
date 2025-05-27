# mcp_server/game_plotter_mcp_server.py
import logging
import json
import os
import io
from datetime import datetime, UTC
from typing import Any, Dict, Optional, List
import asyncio

import matplotlib
matplotlib.use('Agg') # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from google.cloud import storage
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "silver-455021") # Or your GCP project
GCS_BUCKET_GENERATED_GRAPHS = os.getenv("GCS_BUCKET_GENERATED_GRAPHS", "mlb_generated_graphs") # Create this bucket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("game_plotter_tool") # Toolset name

# --- Initialize GCS Client ---
storage_client = None
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("GCS client initialized for Game Plotter.")
except Exception as e:
    logger.critical(f"Failed to initialize GCS client for Game Plotter: {e}", exc_info=True)

# --- Helper to Save to GCS ---
async def _save_plot_to_gcs_async(
    plot_bytes: bytes,
    bucket_name: str,
    blob_name: str,
    content_type: str = 'image/png'
) -> Optional[str]:
    if not storage_client:
        logger.error("GCS storage client not initialized for plot saving.")
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(
            blob.upload_from_string,
            plot_bytes,
            content_type=content_type
        )
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved generated graph to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving graph to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

# --- MCP Tool for Graph Generation ---
@mcp.tool()
async def generate_graph(
    game_pk_str: str,
    graph_id: str,
    chart_type: str, # "line", "bar", "pie", "scatter"
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    # Data can be passed in various ways; JSON strings are versatile
    x_data_json: str = "[]", # e.g., '["Inning 1", "Inning 2"]' or '[1, 2, 3]'
    y_data_json: str = "[]", # e.g., '[0.55, 0.60]' or '[10, 12, 9]'
    data_series_json: str = "[]", # For multi-series plots: '[{"label": "TeamA", "y_values": [1,2,3]}, ...]'
                                         # x_values could be shared or per series.
    options_json: str = "{}" # For colors, linestyles, etc. e.g., '{"color": "blue"}'
) -> str:
    """
    Generates a graph image based on provided data and instructions, saves it to GCS,
    and returns the GCS URI.
    """
    logger.info(f"PLOTTER_MCP: Received graph request. X_data: '{x_data_json}', Y_data: '{y_data_json}', Series: '{data_series_json}', Options: '{options_json}'")
    if not storage_client:
        return json.dumps({"status": "error", "message": "Storage client not initialized."})

    logger.info(f"PLOTTER_MCP: Received request to generate graph '{graph_id}' for game '{game_pk_str}'. Type: '{chart_type}'. Title: '{title}'")

    try:
        x_data = json.loads(x_data_json) if x_data_json else []
        y_data = json.loads(y_data_json) if y_data_json else []
        data_series = json.loads(data_series_json) if data_series_json else []
        options = json.loads(options_json) if options_json else {}
    except json.JSONDecodeError as e:
        logger.error(f"PLOTTER_MCP: Invalid JSON in data/options for graph '{graph_id}': {e}")
        return json.dumps({"status": "error", "message": f"Invalid JSON input: {e}"})

    fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6))) # Default size 10x6 inches

    try:
        if chart_type == "line":
            if data_series: # Multi-line plot
                for series in data_series:
                    ax.plot(series.get("x_values", x_data), series["y_values"], label=series.get("label"), marker=series.get("marker", 'o'), linestyle=series.get("linestyle", '-'))
                if len(data_series) > 1:
                    ax.legend(title=options.get("legend_title"))
            elif x_data and y_data: # Single line plot
                ax.plot(x_data, y_data, marker=options.get("marker", 'o'), linestyle=options.get("linestyle", '-'), color=options.get("color"))
            else:
                raise ValueError("Line chart requires x_data and y_data, or data_series.")

        elif chart_type == "bar":
            if data_series: # Grouped or stacked bar - simplified to grouped for now
                num_series = len(data_series)
                num_categories = len(x_data) # Assumes x_data provides categories
                bar_width = 0.8 / num_series
                for i, series in enumerate(data_series):
                    positions = [x - (num_series/2 - i - 0.5) * bar_width for x in range(num_categories)]
                    ax.bar(positions, series["y_values"], width=bar_width, label=series.get("label"), color=series.get("color"))
                ax.set_xticks(range(num_categories))
                ax.set_xticklabels(x_data)
                if len(data_series) > 1:
                    ax.legend(title=options.get("legend_title"))
            elif x_data and y_data: # Simple bar chart
                ax.bar(x_data, y_data, color=options.get("color"))
            else:
                raise ValueError("Bar chart requires x_data and y_data, or data_series.")

        elif chart_type == "scatter":
            if x_data and y_data:
                ax.scatter(x_data, y_data, color=options.get("color"), marker=options.get("marker", 'o'))
            else:
                raise ValueError("Scatter plot requires x_data and y_data.")

        elif chart_type == "pie":
            if y_data and x_data: # y_data are sizes, x_data are labels for pie
                ax.pie(y_data, labels=x_data, autopct=options.get("autopct", '%1.1f%%'), startangle=options.get("startangle", 90), colors=options.get("colors"))
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            else:
                raise ValueError("Pie chart requires y_data (sizes) and x_data (labels).")
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        ax.set_title(title, fontsize=options.get("title_fontsize", 16))
        ax.set_xlabel(x_axis_label, fontsize=options.get("label_fontsize", 12))
        ax.set_ylabel(y_axis_label, fontsize=options.get("label_fontsize", 12))
        
        if chart_type not in ["pie"]: # Don't add grid to pie charts typically
            ax.grid(options.get("grid", True), linestyle='--', alpha=0.7)

        # Improve tick formatting if x_data is numeric for line/scatter/bar
        if all(isinstance(x, (int, float)) for x in x_data) and chart_type in ["line", "scatter", "bar"] and not data_series:
             ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=options.get("x_nbins", 'auto')))

        plt.tight_layout()

        img_bytes_io = io.BytesIO()
        plt.savefig(img_bytes_io, format='png', dpi=options.get("dpi", 100))
        img_bytes_io.seek(0)
        plot_bytes = img_bytes_io.getvalue()
        plt.close(fig) # Close the figure to free memory

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        clean_graph_id = "".join(c if c.isalnum() else "_" for c in graph_id)[:50] # Sanitize for filename
        blob_name = f"generated/game_{game_pk_str}/graphs/{clean_graph_id}_{timestamp}.png"
        
        gcs_uri = await _save_plot_to_gcs_async(plot_bytes, GCS_BUCKET_GENERATED_GRAPHS, blob_name)

        if gcs_uri:
            logger.info(f"PLOTTER_MCP: Successfully generated and saved graph '{graph_id}' to {gcs_uri}")
            return json.dumps({"status": "success", "graph_id": graph_id, "graph_image_uri": gcs_uri, "title": title, "type": "generated_graph"})
        else:
            logger.error(f"PLOTTER_MCP: Failed to save graph '{graph_id}' to GCS.")
            return json.dumps({"status": "error", "message": f"Failed to save graph '{graph_id}' to GCS."})

    except ValueError as ve: # Catch specific errors for bad chart types or data
        logger.error(f"PLOTTER_MCP: ValueError generating graph '{graph_id}': {ve}")
        return json.dumps({"status": "error", "message": f"Configuration error for graph '{graph_id}': {ve}"})
    except Exception as e:
        logger.error(f"PLOTTER_MCP: Error generating graph '{graph_id}': {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Failed to generate graph '{graph_id}': {e}"})
    finally:
        plt.close('all') # Ensure all figures are closed


if __name__ == "__main__":
    if not storage_client:
        logger.critical("Game Plotter MCP Server cannot start: GCS client failed to initialize.")
    else:
        logger.info("Starting Game Plotter MCP Server...")
        mcp.run(transport="stdio")