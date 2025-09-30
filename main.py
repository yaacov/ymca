#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import List, Optional, Union

from modules.chat.planning_chat_manager import PlanningChatManager
from modules.config.config import Config
from modules.filesystem.filesystem_manager import FilesystemManager
from modules.llm.llm import LLM
from modules.memory.memory_manager import MemoryManager
from modules.memory.models import Memory, MemorySearchResult
from modules.web.web_browser import WebBrowser


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=numeric_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Suppress warnings from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    if log_level.upper() == "DEBUG":
        logging.getLogger("ymca").setLevel(logging.DEBUG)


def show_help() -> None:
    """Display available commands."""
    print("Commands:")
    print("  /quit - Exit the application")
    print("  /help - Show this help message")
    print("  /device - Show current device information")
    print("  /history - Show conversation history summary")
    print("  /clear - Clear conversation history")
    print("  /system <prompt> - Set or update system prompt")
    print("  /window <N> - Set history window size (number of conversation pairs)")
    print()
    print("Web Commands:")
    print("  /search <query> - Search the web")
    print("  /read <url> - Read and extract content from a webpage")
    print("  /smart_search <query> [criteria] - Intelligent web search with content extraction")
    print("  /web_session - Show browser session state")
    print("  /web_clear - Clear browser session")
    print()
    print("Memory Commands:")
    print("  /memory_store <content> - Store content as a memory")
    print("  /memory_search <query> - Search memories")
    print("  /memory_list [limit] - List stored memories")
    print("  /memory_get <id> - Get a specific memory by ID")
    print("  /memory_delete <id> - Delete a memory by ID")
    print("  /memory_stats - Show memory system statistics")
    print()
    print("Filesystem Commands:")
    print("  /fs_list [path] - List files in directory")
    print("  /fs_read <file_path> - Read content from a file")
    print("  /fs_write <file_path> <content> - Write content to a file")
    print("  /fs_delete <file_path> - Delete a file")
    print("  /fs_mkdir <dir_path> - Create a directory")
    print("  /fs_info <file_path> - Get file/directory information")
    print("  /fs_stats - Show filesystem configuration")
    print()
    print("Planning Commands:")
    print("  /plan <task> - Force planning (ignores current mode setting)")
    print("  /plan_status - Show current plan status")
    print("  /plan_list - List saved plans")
    print("  /plan_load <plan_id> - Load a saved plan")
    print("  /plan_resume <plan_id> - Resume execution of a paused plan")
    print("  /plan_mode [always|never|auto] - Set planning behavior")
    print("    • always: Use planning for all tasks")
    print("    • never: Only simple chat responses")
    print("    • auto: Intelligent detection (default)")
    print("  /tools - Show available tools")
    print("  /tools_stats - Show tool statistics")


def handle_command(
    command: str,
    llm: Optional[LLM] = None,
    chat_manager: Optional[PlanningChatManager] = None,
    web_browser: Optional[WebBrowser] = None,
    memory_manager: Optional[MemoryManager] = None,
    filesystem_manager: Optional[FilesystemManager] = None,
    system_prompt: str = "",
) -> Union[bool, str]:
    """Handle slash commands."""
    parts = command.split(" ", 1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit":
        return False
    elif cmd == "/help":
        show_help()
    elif cmd == "/device":
        if llm:
            device_info = llm.get_device_info()
            print(f"Current device: {llm.current_device}")
            print(f"Model loaded: {llm.is_loaded()}")
            optimizations = device_info.get("optimizations", [])
            if optimizations:
                print(f"Active optimizations: {', '.join(optimizations)}")
        else:
            print("LLM not available")
    elif cmd == "/history":
        if chat_manager:
            summary = chat_manager.get_history_summary()
            print(f"Conversation history: {summary['total_messages']} messages total")
            print(f"  User messages: {summary['user_messages']}")
            print(f"  Assistant messages: {summary['assistant_messages']}")
            print(f"  History window: {summary['history_window']} conversation pairs")
            print(f"  System prompt: {len(system_prompt)} characters")
        else:
            print("Chat manager not available")
    elif cmd == "/clear":
        if chat_manager:
            chat_manager.clear_history()
            print("Conversation history cleared")
        else:
            print("Chat manager not available")
    elif cmd == "/system":
        if chat_manager and args:
            chat_logger = logging.getLogger("ymca.chat")
            chat_logger.debug(f"System prompt updated: {args[:100]}{'...' if len(args) > 100 else ''}")
            print(f"System prompt updated: {args[:100]}{'...' if len(args) > 100 else ''}")
            return args  # Return the new system prompt
        elif chat_manager:
            print("Usage: /system <your system prompt>")
        else:
            print("Chat manager not available")
    elif cmd == "/window":
        if chat_manager and args:
            try:
                window_size = int(args)
                chat_manager.set_history_window(window_size)
                print(f"History window size set to {window_size} conversation pairs")
            except ValueError:
                print("Invalid window size. Please provide a number.")
            except Exception as e:
                print(f"Error setting window size: {e}")
        elif chat_manager:
            print("Usage: /window <number>")
        else:
            print("Chat manager not available")
    elif cmd == "/search":
        if web_browser and args:
            try:
                import asyncio

                results = asyncio.run(web_browser.search_web(args))
                print(f"Found {len(results)} search results for '{args}':")
                for i, web_result in enumerate(results[:5], 1):
                    score = f" (score: {web_result.relevance_score:.2f})" if web_result.relevance_score else ""
                    print(f"{i}. {web_result.title}{score}")
                    print(f"   {web_result.url}")
                    if web_result.description:
                        print(f"   {web_result.description[:100]}...")
                    print()
            except Exception as e:
                print(f"Search failed: {e}")
        elif web_browser:
            print("Usage: /search <query>")
        else:
            print("Web browser not available")
    elif cmd == "/read":
        if web_browser and args:
            try:
                import asyncio

                webpage = asyncio.run(web_browser.read_webpage(args))
                if webpage:
                    print(f"Title: {webpage.title}")
                    print(f"URL: {webpage.url}")
                    print(f"Content length: {len(webpage.content)} characters")
                    print(f"Links found: {len(webpage.links)}")
                    print("\nContent preview:")
                    print(webpage.content[:2500] + "..." if len(webpage.content) > 2500 else webpage.content)
                else:
                    print(f"Failed to read webpage: {args}")
            except Exception as e:
                print(f"Read failed: {e}")
        elif web_browser:
            print("Usage: /read <url>")
        else:
            print("Web browser not available")
    elif cmd == "/smart_search":
        if web_browser and args:
            try:
                import asyncio

                parts = args.split(" ", 1)
                query = parts[0]
                criteria = parts[1] if len(parts) > 1 else None

                search_result = asyncio.run(web_browser.smart_web_search(query, criteria))
                print(f"Smart search results for '{query}':")
                if criteria:
                    print(f"Criteria: {criteria}")

                for i, web_result in enumerate(search_result["results"], 1):
                    score = f" (score: {web_result.relevance_score:.2f})" if web_result.relevance_score else ""
                    print(f"{i}. {web_result.title}{score}")
                    print(f"   {web_result.url}")

                if search_result["pages"]:
                    print(f"\nExtracted content from {len(search_result['pages'])} pages:")
                    for page in search_result["pages"]:
                        print(f"- {page.title} ({len(page.content)} chars)")
            except Exception as e:
                print(f"Smart search failed: {e}")
        elif web_browser:
            print("Usage: /smart_search <query> [criteria]")
        else:
            print("Web browser not available")
    elif cmd == "/web_session":
        if web_browser:
            state = web_browser.get_session_state()
            print(f"Visited URLs: {len(state['visited_urls'])}")
            if state["visited_urls"]:
                print("Recent URLs:")
                for url in state["visited_urls"][-5:]:
                    print(f"  - {url}")
            print(f"Cookies: {len(state['cookies'])}")
            print(f"Current User-Agent: {state['current_user_agent'][:50]}...")
        else:
            print("Web browser not available")
    elif cmd == "/web_clear":
        if web_browser:
            web_browser.clear_session()
            print("Web browser session cleared")
        else:
            print("Web browser not available")
    elif cmd == "/memory_store":
        if memory_manager and args:
            try:
                memory_id = memory_manager.store_memory(args)
                print(f"Memory stored successfully with ID: {memory_id}")
            except Exception as e:
                print(f"Failed to store memory: {e}")
        elif memory_manager:
            print("Usage: /memory_store <content>")
        else:
            print("Memory manager not available")
    elif cmd == "/memory_search":
        if memory_manager and args:
            try:
                memory_results: List[MemorySearchResult] = memory_manager.search_memories(args)
                if memory_results:
                    print(f"Found {len(memory_results)} memory results for '{args}':")
                    for i, mem_result in enumerate(memory_results, 1):
                        print(f"{i}. ID: {mem_result.memory.id[:8]}... (score: {mem_result.relevance_score:.3f})")
                        print(f"   Summary: {mem_result.memory.summary}")
                        print(f"   Matched questions: {', '.join(mem_result.matched_questions[:2])}")
                        print()
                else:
                    print(f"No memories found for '{args}'")
            except Exception as e:
                print(f"Memory search failed: {e}")
        elif memory_manager:
            print("Usage: /memory_search <query>")
        else:
            print("Memory manager not available")
    elif cmd == "/memory_list":
        if memory_manager:
            try:
                limit = int(args) if args else 10
                memories = memory_manager.list_memories(limit=limit)
                if memories:
                    print(f"Showing {len(memories)} most recent memories:")
                    for i, memory in enumerate(memories, 1):
                        print(f"{i}. ID: {memory.id[:8]}... ({memory.created_at.strftime('%Y-%m-%d %H:%M')})")
                        print(f"   Summary: {memory.summary}")
                        if memory.tags:
                            print(f"   Tags: {', '.join(memory.tags)}")
                        print()
                else:
                    print("No memories stored yet")
            except ValueError:
                print("Invalid limit. Please provide a number.")
            except Exception as e:
                print(f"Failed to list memories: {e}")
        else:
            print("Memory manager not available")
    elif cmd == "/memory_get":
        if memory_manager and args:
            try:
                retrieved_memory: Optional[Memory] = memory_manager.get_memory(args)
                if retrieved_memory:
                    print(f"Memory ID: {retrieved_memory.id}")
                    print(f"Created: {retrieved_memory.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Updated: {retrieved_memory.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Summary: {retrieved_memory.summary}")
                    if retrieved_memory.tags:
                        print(f"Tags: {', '.join(retrieved_memory.tags)}")
                    print(f"\nContent:\n{retrieved_memory.content}")
                    if retrieved_memory.questions:
                        print("\nGenerated questions:")
                        for i, q in enumerate(retrieved_memory.questions, 1):
                            print(f"  {i}. {q.text}")
                else:
                    print(f"Memory not found: {args}")
            except Exception as e:
                print(f"Failed to get memory: {e}")
        elif memory_manager:
            print("Usage: /memory_get <memory_id>")
        else:
            print("Memory manager not available")
    elif cmd == "/memory_delete":
        if memory_manager and args:
            try:
                if memory_manager.delete_memory(args):
                    print(f"Memory {args} deleted successfully")
                else:
                    print(f"Memory not found: {args}")
            except Exception as e:
                print(f"Failed to delete memory: {e}")
        elif memory_manager:
            print("Usage: /memory_delete <memory_id>")
        else:
            print("Memory manager not available")
    elif cmd == "/memory_stats":
        if memory_manager:
            try:
                stats = memory_manager.get_stats()
                print("Memory System Statistics:")
                print(f"  Total memories: {stats.get('total_memories', 0)}")
                print(f"  Total questions: {stats.get('total_questions', 0)}")
                print(f"  Questions per memory: {stats.get('questions_per_memory', 2)}")
                print(f"  Index size: {stats.get('index_size', 0)}")
                print(f"  Database path: {stats.get('database_path', 'Unknown')}")
                if stats.get("most_recent_memory"):
                    print(f"  Most recent memory: {stats['most_recent_memory']}")
                embedding_info = stats.get("embedding_service", {})
                if embedding_info and not embedding_info.get("error"):
                    print(f"  Embedding model: {embedding_info.get('model_name', 'Unknown')}")
                    print(f"  Embedding dimension: {embedding_info.get('embedding_dimension', 'Unknown')}")
            except Exception as e:
                print(f"Failed to get memory stats: {e}")
        else:
            print("Memory manager not available")
    elif cmd == "/fs_list":
        if filesystem_manager:
            try:
                directory_path = args if args else ""
                listing = filesystem_manager.list_files(directory_path)
                print(f"Directory listing for '{listing.path}':")
                print(f"  Files: {listing.total_files}, Directories: {listing.total_directories}")
                print(f"  Total size: {listing.total_size_formatted}")

                if listing.directories:
                    print("\nDirectories:")
                    for directory in listing.directories:
                        print(f"  {directory.name}/ - {directory.modified_time_formatted}")

                if listing.files:
                    print("\nFiles:")
                    for file in listing.files:
                        print(f"  {file.name} - {file.size_formatted} - {file.modified_time_formatted}")

                if not listing.files and not listing.directories:
                    print("  (empty directory)")
            except Exception as e:
                print(f"Failed to list directory: {e}")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_read":
        if filesystem_manager and args:
            try:
                result = filesystem_manager.read_file(args)
                if result.success:
                    content = result.metadata["content"]
                    file_info = result.metadata["file_info"]
                    print(f"File: {file_info.name}")
                    print(f"Size: {file_info.size_formatted}")
                    print(f"Modified: {file_info.modified_time_formatted}")
                    print(f"Content length: {len(content)} characters")
                    print("\nContent:")
                    print("-" * 50)
                    print(content)
                    print("-" * 50)
                else:
                    print(f"Failed to read file: {result.message}")
            except Exception as e:
                print(f"Failed to read file: {e}")
        elif filesystem_manager:
            print("Usage: /fs_read <file_path>")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_write":
        if filesystem_manager and args:
            try:
                parts = args.split(" ", 1)
                if len(parts) < 2:
                    print("Usage: /fs_write <file_path> <content>")
                    return True

                file_path, content = parts
                result = filesystem_manager.write_file(file_path, content, overwrite=True)
                if result.success:
                    file_info = result.metadata["file_info"]
                    print(f"File written successfully: {file_path}")
                    print(f"Size: {file_info.size_formatted}")
                    print(f"Content length: {result.metadata['content_length']} characters")
                else:
                    print(f"Failed to write file: {result.message}")
            except Exception as e:
                print(f"Failed to write file: {e}")
        elif filesystem_manager:
            print("Usage: /fs_write <file_path> <content>")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_delete":
        if filesystem_manager and args:
            try:
                result = filesystem_manager.delete_file(args)
                if result.success:
                    print(f"File deleted successfully: {args}")
                else:
                    print(f"Failed to delete file: {result.message}")
            except Exception as e:
                print(f"Failed to delete file: {e}")
        elif filesystem_manager:
            print("Usage: /fs_delete <file_path>")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_mkdir":
        if filesystem_manager and args:
            try:
                result = filesystem_manager.create_directory(args)
                if result.success:
                    print(f"Directory created successfully: {args}")
                else:
                    print(f"Failed to create directory: {result.message}")
            except Exception as e:
                print(f"Failed to create directory: {e}")
        elif filesystem_manager:
            print("Usage: /fs_mkdir <dir_path>")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_info":
        if filesystem_manager and args:
            try:
                result = filesystem_manager.get_file_info(args)
                if result.success:
                    file_info = result.metadata["file_info"]
                    print(f"{'Directory' if file_info.is_directory else 'File'}: {file_info.name}")
                    print(f"Path: {file_info.path}")
                    print(f"Size: {file_info.size_formatted}")
                    print(f"Modified: {file_info.modified_time_formatted}")
                    print(f"Created: {file_info.created_time_formatted}")
                    if file_info.extension:
                        print(f"Extension: {file_info.extension}")
                else:
                    print(f"Failed to get file info: {result.message}")
            except Exception as e:
                print(f"Failed to get file info: {e}")
        elif filesystem_manager:
            print("Usage: /fs_info <file_path>")
        else:
            print("Filesystem manager not available")
    elif cmd == "/fs_stats":
        if filesystem_manager:
            print("Filesystem Configuration:")
            print(f"  Base directory: {filesystem_manager.base_dir}")
            print(f"  Max file size: {filesystem_manager.max_file_size_mb}MB")
            print(f"  Allowed extensions: {', '.join(sorted(filesystem_manager.allowed_extensions)) if filesystem_manager.allowed_extensions else 'All extensions allowed'}")
            print(f"  Subdirectories enabled: {filesystem_manager.enable_subdirectories}")
            print(f"  Base directory exists: {filesystem_manager.base_dir.exists()}")
        else:
            print("Filesystem manager not available")
    elif cmd == "/plan":
        if chat_manager and args:
            try:
                import asyncio

                planning_result: str = asyncio.run(chat_manager.force_planning(args))
                print(f"Planning result:\n{planning_result}")
            except Exception as e:
                print(f"Planning failed: {e}")
        elif chat_manager:
            print("Usage: /plan <task description>")
        else:
            print("Chat manager not available")
    elif cmd == "/plan_status":
        if chat_manager:
            status = chat_manager.get_current_plan_status()
            if status:
                print(f"Current Plan: {status['title']}")
                print(f"Status: {status['status']}")
                print(f"Progress: {status['progress']['completed']}/{status['progress']['total_steps']} steps")
                print(f"Created: {status['created_at']}")
            else:
                print("No active plan")
        else:
            print("Chat manager not available")
    elif cmd == "/plan_list":
        if chat_manager:
            plans = chat_manager.list_saved_plans(10)
            if plans:
                print("Saved Plans:")
                for plan in plans:
                    print(f"  {plan['id'][:8]}... - {plan['title']} ({plan['status']}) - {plan['steps_count']} steps")
            else:
                print("No saved plans found")
        else:
            print("Chat manager not available")
    elif cmd == "/plan_load":
        if chat_manager and args:
            loaded_plan = chat_manager.load_plan(args)
            if loaded_plan:
                print(f"Loaded plan: {loaded_plan.title}")
                print(f"Status: {loaded_plan.status.value}")
                if loaded_plan.result:
                    print(f"Result available: {len(loaded_plan.result)} characters")
                if loaded_plan.evolving_answer:
                    print(f"Evolving answer: {len(loaded_plan.evolving_answer)} characters")
            else:
                print(f"Plan not found: {args}")
        elif chat_manager:
            print("Usage: /plan_load <plan_id>")
        else:
            print("Chat manager not available")
    elif cmd == "/plan_resume":
        if chat_manager and args:
            try:
                import asyncio

                resume_result: str = asyncio.run(chat_manager.resume_plan(args))
                print(resume_result)
            except Exception as e:
                print(f"Failed to resume plan: {e}")
        elif chat_manager:
            print("Usage: /plan_resume <plan_id>")
        else:
            print("Chat manager not available")
    elif cmd == "/plan_mode":
        if chat_manager:
            if args.lower() == "always":
                chat_manager.set_planning_mode("always")
                print("Planning mode: ALWAYS - all tasks will use multi-step planning")
            elif args.lower() == "never":
                chat_manager.set_planning_mode("never")
                print("Planning mode: NEVER - will only use simple chat responses")
            elif args.lower() == "auto":
                chat_manager.set_planning_mode("auto")
                print("Planning mode: AUTO - will intelligently detect when to use planning")
            else:
                mode = chat_manager.get_planning_mode()
                print(f"Planning mode is currently: {mode.upper()}")
                print("Usage: /plan_mode [always|never|auto]")
                print("  - always: Use planning for all tasks")
                print("  - never: Only use simple chat")
                print("  - auto: Intelligent detection (default)")
        else:
            print("Chat manager not available")
    elif cmd == "/tools":
        if chat_manager:
            tools = chat_manager.get_available_tools()
            print("Available Tools:")
            for tool in tools:
                print(f"  - {tool}")
        else:
            print("Chat manager not available")
    elif cmd == "/tools_stats":
        if chat_manager:
            stats = chat_manager.get_tool_stats()
            print("Tool Statistics:")
            print(f"  Total tools: {stats['total_tools']}")
            print(f"  Enabled tools: {stats['enabled_tools']}")
            print("  By category:")
            for category, cat_stats in stats["by_category"].items():
                print(f"    {category}: {cat_stats['enabled']}/{cat_stats['total']} enabled")
        else:
            print("Chat manager not available")
    else:
        print("Unknown command. Type /help for available commands.")
    return True


def handle_user_request(chat_manager: PlanningChatManager, user_input: str, system_prompt: str) -> None:
    """Process user input and generate Assistant response."""
    try:
        import asyncio

        response = asyncio.run(chat_manager.send_message_with_planning(user_input, system_prompt=system_prompt))
        print(f"Assistant: {response}")
    except Exception as e:
        print(f"Error processing request: {e}")
        # Fallback to simple chat
        response = chat_manager.send_message(user_input, system_prompt=system_prompt)
        print(f"Assistant (planning off): {response}")


def run_chat_loop(llm: LLM, config: Config, chat_logger: logging.Logger) -> None:
    """Main interactive chat loop."""
    # Setup web browser
    web_logger = logging.getLogger("ymca.web")
    web_browser = WebBrowser(llm=llm, max_requests_per_second=config["WEB_MAX_REQUESTS_PER_SECOND"], max_requests_per_minute=config["WEB_MAX_REQUESTS_PER_MINUTE"], logger=web_logger)

    # Setup memory manager
    memory_logger = logging.getLogger("ymca.memory")
    memory_manager = MemoryManager(
        llm=llm,
        memory_db_path=config["MEMORY_DB_PATH"],
        embedding_model=config["MEMORY_EMBEDDING_MODEL"],
        num_questions_per_memory=config["MEMORY_QUESTIONS_PER_MEMORY"],
        cache_dir=config["MODEL_CACHE_DIR"],
        logger=memory_logger,
        max_chunk_size=config["MEMORY_MAX_CHUNK_SIZE"],
        chunk_overlap=config["MEMORY_CHUNK_OVERLAP"],
    )

    # Setup filesystem manager
    filesystem_manager = FilesystemManager(config)

    # Setup planning chat manager (enhanced chat manager with planning capabilities)
    chat_manager = PlanningChatManager(
        llm=llm,
        config=config,
        web_browser=web_browser,
        memory_manager=memory_manager,
        filesystem_manager=filesystem_manager,
        history_window=config["CHAT_HISTORY_WINDOW"],
        max_history_tokens=config["CHAT_MAX_HISTORY_TOKENS"],
        logger=chat_logger,
    )
    system_prompt = config["CHAT_SYSTEM_PROMPT"]

    print("Type /help for commands")
    print(f"System prompt: {system_prompt}")
    print(f"History window: {chat_manager.history_window} conversation pairs")
    planning_mode = chat_manager.get_planning_mode().upper()
    print(f"Planning mode: {planning_mode} - use /plan_mode to change")
    print()

    while True:
        try:
            user_input = input("User: ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                result = handle_command(user_input, llm, chat_manager, web_browser, memory_manager, filesystem_manager, system_prompt)
                if result is False:
                    break
                elif isinstance(result, str):  # New system prompt
                    system_prompt = result
                continue
            handle_user_request(chat_manager, user_input, system_prompt)

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("Goodbye!")


def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Local LLM Chat Application")
    parser.add_argument("--config", "-c", type=str, default="config.env", help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Setup loggers
    llm_logger = logging.getLogger("ymca.llm")
    chat_logger = logging.getLogger("ymca.chat")

    config = Config(args.config)
    llm = LLM(config, logger=llm_logger)

    print(f"Selected device: {llm.current_device}")
    device_info = llm.get_device_info()
    optimizations = device_info.get("optimizations", [])
    if optimizations:
        print(f"Active optimizations: {', '.join(optimizations)}")

    print("Loading model...")
    if not llm.load_model():
        print("Failed to load model")
        sys.exit(1)
    print("Model loaded successfully")

    run_chat_loop(llm, config, chat_logger)


if __name__ == "__main__":
    main()
