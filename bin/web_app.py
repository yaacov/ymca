#!/usr/bin/env python3
"""
Web Chat Application - API server with static web interface.

Provides REST endpoints for chat interactions with tools, memory, and planning.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to import ymca package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ymca.core.model_handler import ModelHandler
from ymca.tools.memory.tool import MemoryTool
from ymca.tools.mcp.client import MCPClient
from ymca.chat.api import ChatAPI


# ==================== Helper Functions ====================

def load_system_prompt(prompt_arg: str) -> str:
    """
    Load system prompt from string or file.
    
    Args:
        prompt_arg: System prompt string or @filename to load from file
        
    Returns:
        System prompt string
    """
    if prompt_arg.startswith('@'):
        # Load from file
        file_path = Path(prompt_arg[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {file_path}")
        return file_path.read_text(encoding='utf-8')
    return prompt_arg


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    message: str
    enable_tools: bool = True
    enable_planning: bool = True
    max_iterations: int = 5
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


class StatusResponse(BaseModel):
    status: str
    message: str


# ==================== Global State ====================

# Chat API instance (initialized on startup)
chat_api: Optional[ChatAPI] = None


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: Initialize chat API
    if hasattr(app.state, 'args'):
        initialize_chat_api(app.state.args)
    
    yield
    
    # Shutdown: Cleanup MCP clients
    if chat_api and hasattr(chat_api, 'mcp_clients'):
        for name, client in chat_api.mcp_clients.items():
            try:
                client.stop()
            except:
                pass


# ==================== FastAPI App ====================

app = FastAPI(
    title="YMCA Chat API",
    description="Chat API with tools, memory, and planning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Redirect to static web interface."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/static/index.html">
    </head>
    <body>
        Redirecting to chat interface...
    </body>
    </html>
    """)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "chat_initialized": chat_api is not None
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message and get a response.
    
    Args:
        request: ChatRequest with message and options
        
    Returns:
        ChatResponse with assistant's response
    """
    if chat_api is None:
        raise HTTPException(status_code=500, detail="Chat API not initialized")
    
    try:
        response = chat_api.chat(
            user_message=request.message,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            enable_tools=request.enable_tools,
            enable_planning=request.enable_planning
        )
        
        return ChatResponse(response=response)
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear", response_model=StatusResponse)
async def clear_history():
    """Clear conversation history."""
    if chat_api is None:
        raise HTTPException(status_code=500, detail="Chat API not initialized")
    
    chat_api.clear_history()
    return StatusResponse(status="success", message="Conversation history cleared")


@app.get("/api/summary")
async def get_summary():
    """Get conversation summary."""
    if chat_api is None:
        raise HTTPException(status_code=500, detail="Chat API not initialized")
    
    summary = chat_api.get_history_summary()
    return {"summary": summary}


@app.get("/api/export")
async def export_conversation():
    """Export conversation history."""
    if chat_api is None:
        raise HTTPException(status_code=500, detail="Chat API not initialized")
    
    conversation = chat_api.export_conversation()
    return {"conversation": conversation}


# ==================== Initialization ====================

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress sentence-transformers verbose logging
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


def initialize_chat_api(args):
    """Initialize the chat API with model and tools."""
    global chat_api
    
    logging.info("Initializing Chat Application...")
    
    # Initialize model handler
    logging.info("Loading model handler...")
    model_handler = ModelHandler(
        model_path=args.model,
        n_ctx=args.context,
        n_gpu_layers=args.gpu_layers
    )
    actual_ctx = model_handler.llm.n_ctx()
    logging.info(f"Model handler ready (context: {actual_ctx:,} tokens)")
    
    # Initialize memory tool
    logging.info("Loading memory tool...")
    memory_tool = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=model_handler
    )
    total_chunks = len(memory_tool.storage.get_all_chunks())
    if total_chunks == 0:
        logging.warning("Memory store is empty (use memory_cli.py to add data)")
    else:
        logging.info(f"Memory tool ready ({total_chunks} chunks loaded)")
    logging.info(f"Memory directory: {args.memory_dir}")
    
    # System message for the assistant
    if args.system_prompt:
        system_message = load_system_prompt(args.system_prompt)
        logging.info("Custom system prompt loaded")
    else:
        system_message = (
            "You are a helpful technical assistant with access to documentation. "
            "When users ask technical questions, commands, or how-to questions, "
            "you MUST use the retrieve_memory tool to search the documentation first.\n\n"
            "To call a tool, use this exact XML format:\n"
            "<tool_call>\n"
            "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
            "</tool_call>\n\n"
            "Do NOT generate any other text when calling a tool. "
            "Wait for the tool result, then provide your answer based on the documentation.\n\n"
            "**CRITICAL - NO HALLUCINATION POLICY:**\n"
            "- Base your answers EXCLUSIVELY on information from tool results in this conversation\n"
            "- If the tool doesn't return relevant information, clearly state that\n"
            "- NEVER invent facts, commands, or details not provided by tools\n"
            "- If unsure, call retrieve_memory to get accurate documentation\n"
            "- Always cite which tool provided the information you're using\n\n"
            "**RESPONSE STYLE:**\n"
            "- Answer in 1-3 sentences, then add examples if helpful (examples are additional)\n"
            "- Be extremely brief - state only the essential facts\n"
            "- No pleasantries, just direct information"
        )
        logging.info("Default system prompt configured")
    
    # Initialize chat API
    logging.info("Initializing chat API...")
    chat_api = ChatAPI(
        model_handler=model_handler,
        max_history=15,
        system_message=system_message,
        max_tools_in_prompt=2,
        embedder=memory_tool.embedder if memory_tool else None
    )
    logging.info("Chat API ready (semantic tool selection: top-2 mode)")
    
    # Register memory retrieval tool
    logging.info("Registering memory retrieval tool...")
    tool_def = memory_tool.RETRIEVE_TOOL_DEF
    chat_api.register_tool(
        name=tool_def["name"],
        description=tool_def["description"],
        function=memory_tool.create_retrieve_tool_function(),
        parameters=tool_def["parameters"]
    )
    logging.info("Memory retrieval tool registered")
    
    # Register MCP servers if any
    if args.mcp_servers:
        logging.info(f"Registering {len(args.mcp_servers)} MCP server(s)...")
        for mcp_spec in args.mcp_servers:
            try:
                if ':' not in mcp_spec:
                    logging.warning(f"Invalid MCP spec (missing ':'): {mcp_spec}")
                    continue
                
                name, command_str = mcp_spec.split(':', 1)
                command = command_str.strip().split()
                
                logging.info(f"Starting MCP server '{name}'...")
                logging.info(f"Command: {' '.join(command)}")
                mcp_client = MCPClient(name=name, command=command)
                mcp_client.start()
                
                discovered_tools = mcp_client.get_tools()
                logging.info(f"Discovered {len(discovered_tools)} tools from MCP server")
                
                chat_api.register_mcp_server(mcp_client)
                
                registered_tool_names = [name for name in chat_api.tools.keys() if name.startswith(f"{mcp_client.name}.")]
                logging.info(f"MCP server '{name}' registered ({len(registered_tool_names)} tools)")
                
            except Exception as e:
                logging.error(f"Failed to register MCP server '{name}': {e}")
    
    logging.info("All systems ready!")


# ==================== Main ====================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Web Chat Application with Advanced Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web server
  python web_app.py

  # With custom model
  python web_app.py --model path/to/model.gguf

  # With MCP server
  python web_app.py --mcp-server "mtv:kubectl-mtv mcp-server"

  # With MCP server and custom system prompt
  python web_app.py --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

  # Multiple MCP servers
  python web_app.py --mcp-server "server1:cmd1" --mcp-server "server2:cmd2"

  # Custom host and port
  python web_app.py --host 0.0.0.0 --port 8080

  # Enable verbose logging
  python web_app.py --verbose --debug
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/ibm-granite_granite-4.0-h-tiny/gguf/ibm-granite_granite-4.0-h-tiny-q4_1.gguf",
        help="Path to GGUF model"
    )
    
    parser.add_argument(
        "--context",
        type=int,
        default=32768,
        help="Context size in tokens (default: 32768)"
    )
    
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (default: -1 = all, 0 = CPU only)"
    )
    
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="data/tools/memory",
        help="Memory storage directory"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--mcp-server",
        action="append",
        dest="mcp_servers",
        metavar="NAME:COMMAND",
        help="Register MCP server (format: 'name:command arg1 arg2')"
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        metavar="PROMPT",
        help="Custom system prompt (use @filename to load from file, e.g., @docs/mtv-system-prompt.txt)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose or args.debug)
    
    if args.debug:
        logging.getLogger('ymca').setLevel(logging.DEBUG)
        logging.getLogger('chat').setLevel(logging.DEBUG)
        logging.getLogger('memory').setLevel(logging.DEBUG)
    
    # Store args in app state for lifespan startup
    app.state.args = args
    
    # Mount static files directory
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Run server
    logging.info(f"Starting web server on http://{args.host}:{args.port}")
    logging.info(f"Open http://{args.host}:{args.port} in your browser")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if not args.debug else "debug"
    )


if __name__ == "__main__":
    main()

