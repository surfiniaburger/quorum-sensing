# test_video_clip_mcp.py
import asyncio
import json
import os
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_SERVER_COMMAND = "python"
VIDEO_SERVER_ARGS = ["./mcp_server/video_clip_server.py"]
MCP_VIDEO_SERVER_NAME = "video_clip_generator_mcp" # From FastMCP("video_clip_generator")

async def main():
    logger.info("Starting test for Video Clip MCP Server...")
    server_params = StdioServerParameters(command=VIDEO_SERVER_COMMAND, args=VIDEO_SERVER_ARGS)
    exit_stack = None
    generate_video_tool_func = None

    try:
        mcp_tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
        logger.info(f"MCPToolset.from_server for video returned: {mcp_tools} (type: {type(mcp_tools)})")

        if isinstance(mcp_tools, list) and mcp_tools:
            tool_obj = mcp_tools[0]
            if hasattr(tool_obj, 'generate_video_clips_from_prompts'):
                generate_video_tool_func = tool_obj.generate_video_clips_from_prompts
        # Add other access patterns if needed (dict, direct object)

        if not generate_video_tool_func:
            logger.error("Could not find 'generate_video_clips_from_prompts' tool.")
            return

        logger.info("Successfully accessed 'generate_video_clips_from_prompts' tool.")

        test_video_prompts = [
            "A dramatic slow motion shot of a baseball player hitting a home run.",
            "Drone footage flying over a packed baseball stadium during a night game."
        ]
        # The MCP tool now expects a Python List[str] for 'prompts'
        
        game_pk_str = "test_video_game_pk"
        logger.info(f"Calling tool with prompts: {test_video_prompts}, game_pk: {game_pk_str}")

        # Call the tool with the Python list
        result_json_string = await generate_video_tool_func(prompts=test_video_prompts, game_pk_str=game_pk_str)
        logger.info(f"Raw result from video MCP tool: {result_json_string}")

        result_data = json.loads(result_json_string)
        logger.info(f"Parsed video result data: {json.dumps(result_data, indent=2)}")
        if isinstance(result_data, list) and result_data and result_data[0].startswith("gs://"):
            logger.info(f"Successfully generated {len(result_data)} video URIs.")
        elif isinstance(result_data, dict) and result_data.get("error"):
            logger.error(f"Video MCP Tool returned an error: {result_data['error']}")

    except Exception as e:
        logger.error(f"An error occurred in video test script: {e}", exc_info=True)
    finally:
        if exit_stack:
            await exit_stack.aclose()
            logger.info("Video MCP connection closed.")

if __name__ == "__main__":
    # Ensure video_clip_server.py is running
    asyncio.run(main())