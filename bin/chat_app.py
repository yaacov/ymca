#!/usr/bin/env python3
"""
Interactive Chat Application - Chat with tools, memory, and planning.

Features:
- Conversational AI with memory
- Tool calling
- Multi-step planning
- Memory integration
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import ymca package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ymca.core.model_handler import ModelHandler
from ymca.core.ui import ThinkingSpinner, print_markdown
from ymca.tools.memory.tool import MemoryTool
from ymca.tools.mcp.client import MCPClient
from ymca.chat.api import ChatAPI


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s' if not verbose else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress sentence-transformers verbose logging
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


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


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def interactive_chat(chat: ChatAPI):
    """Interactive chat mode."""
    print_section("INTERACTIVE CHAT")
    
    print("\n💬 Commands:")
    print("  - Type your message to chat")
    print("  - 'clear'   - Clear conversation history")
    print("  - 'summary' - Show conversation summary")
    print("  - 'export'  - Export conversation")
    print("  - 'quit'    - Exit (or Ctrl+C)")
    print("\n💡 Tips:")
    print("  - The assistant can search memory if data is loaded")
    print("  - Run with --debug for detailed logging")
    print("  - Run with --verbose for full error traces")
    print("\n" + "=" * 70)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                chat.clear_history()
                print("✓ Conversation history cleared")
                continue
            
            elif user_input.lower() == 'summary':
                print("\n📋 Conversation Summary:")
                print(chat.get_history_summary())
                continue
            
            elif user_input.lower() == 'export':
                conversation = chat.export_conversation()
                print(f"\n✓ Exported {len(conversation)} messages")
                print("(In a real app, this would save to a file)")
                continue
            
            # Regular chat
            with ThinkingSpinner("🤔 Thinking"):
                response = chat.chat(user_input, enable_tools=True, enable_planning=True)
            
            print("\n🤖 Assistant:")
            print_markdown(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Chat Application with Advanced Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive chat
  python chat_app.py

  # With custom model
  python chat_app.py --model path/to/model.gguf

  # With MCP server
  python chat_app.py --mcp-server "mtv:kubectl-mtv mcp-server"

  # With MCP server and custom system prompt
  python chat_app.py --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

  # Multiple MCP servers
  python chat_app.py --mcp-server "server1:cmd1" --mcp-server "server2:cmd2"

  # Enable verbose logging
  python chat_app.py --verbose --debug
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
        help="Context size in tokens (default: 32768, Granite supports up to 128K)"
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
        help="Memory storage directory (default: data/tools/memory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (shows context usage)"
    )
    
    parser.add_argument(
        "--mcp-server",
        action="append",
        dest="mcp_servers",
        metavar="NAME:COMMAND",
        help="Register MCP server (format: 'name:command arg1 arg2'). Can be used multiple times."
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
    
    # Enable debug logging for all ymca modules if requested
    if args.debug:
        logging.getLogger('ymca').setLevel(logging.DEBUG)
        logging.getLogger('chat').setLevel(logging.DEBUG)
        logging.getLogger('memory').setLevel(logging.DEBUG)
    
    try:
        print_section("INITIALIZING CHAT APPLICATION")
        
        # Initialize model handler
        print("\n1️⃣  Loading model handler...")
        model_handler = ModelHandler(
            model_path=args.model,
            n_ctx=args.context,
            n_gpu_layers=args.gpu_layers
        )
        actual_ctx = model_handler.llm.n_ctx()
        print(f"   ✓ Model handler ready (context: {actual_ctx:,} tokens)")
        
        # Initialize memory tool
        print("\n2️⃣  Loading memory tool...")
        memory_tool = MemoryTool(
            memory_dir=args.memory_dir,
            model_handler=model_handler
        )
        # Check memory status
        total_chunks = len(memory_tool.storage.get_all_chunks())
        if total_chunks == 0:
            print("   ⚠️  Memory store is empty (use memory_cli.py to add data)")
        else:
            print(f"   ✓ Memory tool ready ({total_chunks} chunks loaded)")
        print(f"   📁 Memory directory: {args.memory_dir}")
        
        # Initialize chat API
        print("\n3️⃣  Initializing chat API...")
        
        # System message for the assistant
        if args.system_prompt:
            system_message = load_system_prompt(args.system_prompt)
            print(f"   ✓ Custom system prompt loaded")
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
            print("   ✓ Default system prompt configured")
        
        chat = ChatAPI(
            model_handler=model_handler,
            max_history=15,  # Keep last ~3 complete interactions (user + tool calls + answers)
            system_message=system_message,
            max_tools_in_prompt=2,  # Show top 2 most relevant tools for better selection
            embedder=memory_tool.embedder if memory_tool else None  # Use memory embedder for tool selection
        )
        print("   ✓ Chat API ready (semantic tool selection: top-2 mode)")
        
        # Register memory retrieval tool (read-only)
        print("\n4️⃣  Registering memory retrieval tool...")
        tool_def = memory_tool.RETRIEVE_TOOL_DEF
        chat.register_tool(
            name=tool_def["name"],
            description=tool_def["description"],
            function=memory_tool.create_retrieve_tool_function(),
            parameters=tool_def["parameters"]
        )
        
        if args.debug:
            print(f"   Tool registered: {tool_def['name']}")
            print(f"   Description: {tool_def['description'][:100]}...")
            print(f"   Parameters: {list(tool_def['parameters'].get('properties', {}).keys())}")
        
        print("   ✓ Memory retrieval tool registered")
        
        # Register MCP servers if any
        if args.mcp_servers:
            print(f"\n5️⃣  Registering {len(args.mcp_servers)} MCP server(s)...")
            for mcp_spec in args.mcp_servers:
                try:
                    # Parse server specification: "name:command arg1 arg2"
                    if ':' not in mcp_spec:
                        print(f"   ⚠️  Invalid MCP spec (missing ':'): {mcp_spec}")
                        continue
                    
                    name, command_str = mcp_spec.split(':', 1)
                    command = command_str.strip().split()
                    
                    print(f"   Starting MCP server '{name}'...")
                    print(f"   Command: {' '.join(command)}")
                    mcp_client = MCPClient(name=name, command=command)
                    mcp_client.start()
                    
                    # Check what tools were discovered
                    discovered_tools = mcp_client.get_tools()
                    print(f"   Discovered {len(discovered_tools)} tools from MCP server")
                    
                    if args.debug:
                        print(f"     Tool names: {list(discovered_tools.keys())}")
                    
                    # Register with ChatAPI
                    chat.register_mcp_server(mcp_client)
                    
                    # Verify registration
                    registered_tool_names = [name for name in chat.tools.keys() if name.startswith(f"{mcp_client.name}.")]
                    print(f"   ✓ MCP server '{name}' registered ({len(registered_tool_names)} tools)")
                    
                    if args.debug and len(registered_tool_names) > 0:
                        print(f"     Registered as: {registered_tool_names[:5]}{'...' if len(registered_tool_names) > 5 else ''}")
                    
                except Exception as e:
                    print(f"   ✗ Failed to register MCP server '{name}': {e}")
        
        print("\n" + "=" * 70)
        print("🚀 All systems ready!")
        print("=" * 70)
        
        # Run interactive chat
        interactive_chat(chat)
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup MCP servers
        if 'chat' in locals() and hasattr(chat, 'mcp_clients'):
            for name, client in chat.mcp_clients.items():
                try:
                    client.stop()
                except:
                    pass


if __name__ == "__main__":
    main()

