"""
Chat API - Main conversational interface.

Advanced chat API with conversation memory, tools, and planning.
"""

import logging
from typing import Dict, List, Callable, Optional

from .tool import Tool
from .history import ConversationHistory
from .chat_engine import ChatEngine
from .tool_selector import ToolSelector

logger = logging.getLogger(__name__)


class ChatAPI:
    """
    Advanced chat API with conversation memory, tools, and planning.
    
    This class provides a high-level interface for chat interactions,
    managing conversation history and tool registration. The actual
    chat loop execution is delegated to ChatEngine.
    """
    
    def __init__(
        self,
        model_handler,
        max_history: int = 20,
        system_message: Optional[str] = None,
        max_tools_in_prompt: int = 10,
        embedder=None,
        num_tool_queries: int = 5,
        max_history_in_prompt: int = 10
    ):
        """
        Initialize the chat API.
        
        Args:
            model_handler: ModelHandler instance for LLM
            max_history: Maximum conversation history to keep in memory (default: 20)
            system_message: Custom system message (default: "You are a helpful AI assistant.")
            max_tools_in_prompt: Maximum tools to include in system prompt (default: 10)
            embedder: Optional embedder for semantic tool selection
            num_tool_queries: Number of semantic summaries to generate per tool for embedding (default: 5)
            max_history_in_prompt: Maximum history messages to include in LLM prompt (default: 10)
        """
        self.model_handler = model_handler
        self.history = ConversationHistory(max_messages=max_history)
        self.max_history_in_prompt = max_history_in_prompt
        self.tools: Dict[str, Tool] = {}
        self.mcp_clients: Dict[str, 'MCPClient'] = {}  # MCP server clients
        self.base_system_message = system_message or (
            "You are a technical assistant. Answer the USER'S EXACT QUESTION using ONLY information from tool results.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY the user's actual question - never invent your own question\n"
            "2. Use ONLY facts explicitly stated in tool results - no assumptions or inventions\n"
            "3. If tool results don't contain the answer, say 'I don't have that information'\n"
            "4. NEVER add features, capabilities, or details not in the tool results\n"
            "5. LENGTH LIMIT: Your ENTIRE response including examples must fit in 10 lines maximum\n"
            "   - Core answer: 1-3 sentences\n"
            "   - Examples: Brief, only if needed\n"
            "   - Total: Maximum 20 lines, no exceptions\n\n"
            "TOOL QUERY BEST PRACTICES:\n"
            "When calling retrieve_memory or similar search tools, use detailed, specific queries:\n"
            "✓ GOOD: 'how to create authentication configuration step by step'\n"
            "✓ GOOD: 'troubleshooting database connection errors'\n"
            "✗ BAD: 'authentication' (too vague)\n"
            "✗ BAD: 'configuration' (too general)\n"
            "Include action words (how to, configure, troubleshoot, create, setup) and specific technical terms.\n\n"
            "FORMAT: Just answer directly. Don't create fake Q&A format. Don't add 'Question:' or 'Answer:' labels."
        )
        self.max_tools_in_prompt = max_tools_in_prompt
        
        # Initialize tool selector (pass model_handler for query generation)
        self.tool_selector = ToolSelector(embedder=embedder, model_handler=model_handler, num_queries=num_tool_queries)
        
        # Initialize chat engine with tool executor
        self.chat_engine = ChatEngine(
            model_handler=model_handler,
            tool_executor=self._execute_tool
        )
        
        # Set initial system message (will be updated when tools are registered)
        self.history.set_system_message(self.base_system_message)
        
        logger.info("ChatAPI initialized")
    
    # ==================== Tool Management ====================
    
    def register_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict
    ):
        """
        Register a new tool for use in chat.
        
        Args:
            name: Tool name (must be unique)
            description: Tool description for LLM
            function: Python function to execute when tool is called
            parameters: JSON schema defining the tool's parameters
        """
        self.tools[name] = Tool(
            name=name,
            description=description,
            function=function,
            parameters=parameters
        )
        
        # Reindex tools for semantic selection
        self.tool_selector.index_tools(self.tools)
        
        # Update system message (will be dynamically updated per query)
        self._update_system_message()
        
        logger.info(f"Registered tool: {name}")
    
    def register_mcp_server(self, mcp_client: 'MCPClient'):
        """
        Register an MCP server and all its tools.
        
        Args:
            mcp_client: Started MCPClient instance
        """
        server_name = mcp_client.name
        self.mcp_clients[server_name] = mcp_client
        
        # Register all tools from the MCP server
        mcp_tools = mcp_client.get_tools()
        for tool_name, tool_def in mcp_tools.items():
            # Prefix tool name with server name to avoid conflicts
            full_tool_name = f"{server_name}.{tool_name}"
            
            # Create a wrapper function for the MCP tool
            def make_mcp_wrapper(client, name):
                def mcp_tool_wrapper(**kwargs):
                    return client.call_tool(name, kwargs)
                return mcp_tool_wrapper
            
            self.tools[full_tool_name] = Tool(
                name=full_tool_name,
                description=f"[MCP: {server_name}] {tool_def['description']}",
                function=make_mcp_wrapper(mcp_client, tool_name),
                parameters=tool_def['parameters']
            )
            
            logger.debug(f"  Registered MCP tool: {full_tool_name}")
        
        # Reindex all tools for semantic selection
        self.tool_selector.index_tools(self.tools)
        
        # Update system message (will be dynamically updated per query)
        self._update_system_message()
        
        logger.info(f"Registered MCP server '{server_name}' with {len(mcp_tools)} tools")
    
    def _get_llm_tools(self) -> List[Dict]:
        """Get tools in OpenAI-compatible format for LLM."""
        return [tool.to_llm_format() for tool in self.tools.values()]
    
    def _update_system_message(self, selected_tool_names: Optional[List[str]] = None):
        """
        Update system message to include tool descriptions.
        
        Args:
            selected_tool_names: Optional list of tool names to include. 
                               If None, includes all tools.
        """
        if not self.tools:
            self.history.set_system_message(self.base_system_message)
            return
        
        # Determine which tools to include
        if selected_tool_names:
            tools_to_include = {name: self.tools[name] for name in selected_tool_names if name in self.tools}
        else:
            tools_to_include = self.tools
        
        if not tools_to_include:
            self.history.set_system_message(self.base_system_message)
            return
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in tools_to_include.values():
            params_desc = []
            props = tool.parameters.get('properties', {})
            required = tool.parameters.get('required', [])
            
            for param_name, param_info in props.items():
                is_required = " (required)" if param_name in required else ""
                param_desc = f"    {param_name}{is_required}: {param_info.get('description', '')}"
                params_desc.append(param_desc)
            
            # Create a concrete example with actual parameter names
            example_args = {}
            for param_name in props.keys():
                if param_name in required:
                    example_args[param_name] = f"<{param_name}_value>"
            
            import json
            example_json = json.dumps({"name": tool.name, "arguments": example_args}, indent=2)
            
            tool_desc = f"""
{tool.name}: {tool.description}
  Parameters:
{chr(10).join(params_desc)}
  
  Example usage:
  <tool_call>
{example_json}
  </tool_call>"""
            tool_descriptions.append(tool_desc)
        
        # Combine base message with tool descriptions
        enhanced_message = f"""{self.base_system_message}

# Available Tools
{chr(10).join(tool_descriptions)}

CRITICAL INSTRUCTIONS:

**Tool Calling:**
- Output ONLY the <tool_call> block with properly formatted JSON
- Do NOT add any text before or after the tool call
- Wait for the result before responding

**Answering Questions:**
- Answer the USER'S ACTUAL QUESTION - do not invent different questions
- Use ONLY information explicitly stated in tool results - no additions or assumptions
- If information isn't in the tool results, say 'I don't have that information'
- LENGTH LIMIT: Maximum 20 lines total including all examples
- Answer directly without adding extra formatting or structure
- Stay strictly within the scope of the retrieved information

**Remember:** Answer the actual question. Tool results only. No hallucinations. Maximum 20 lines total."""
        
        self.history.set_system_message(enhanced_message)
    
    def _execute_tool(self, name: str, arguments: Dict) -> str:
        """
        Execute a tool by name with given arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments dictionary
            
        Returns:
            Tool execution result as string
        """
        if name not in self.tools:
            error_msg = f"Error: Unknown tool '{name}'"
            logger.error(error_msg)
            return error_msg
        
        try:
            tool = self.tools[name]
            result = tool.function(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution error ({name}): {e}")
            
            # Provide helpful feedback for invalid parameters
            error_str = str(e)
            
            # Check for parameter validation errors
            if "invalid params" in error_str.lower() or "unexpected additional properties" in error_str.lower():
                return self._format_parameter_error(name, arguments, error_str)
            
            # Check for missing required parameters
            if "required" in error_str.lower() or "missing" in error_str.lower():
                return self._format_missing_parameter_error(name, arguments, error_str)
            
            # Generic error
            return f"Error executing {name}: {str(e)}"
    
    def _format_parameter_error(self, tool_name: str, arguments: Dict, error_str: str) -> str:
        """Format a helpful error message for invalid parameters."""
        tool = self.tools[tool_name]
        valid_params = tool.parameters.get('properties', {})
        required_params = tool.parameters.get('required', [])
        
        # Extract which parameters were invalid
        invalid_params = [param for param in arguments.keys() if param not in valid_params]
        
        error_msg = f"Tool call failed: {tool_name}\n\n"
        error_msg += f"Error: {error_str}\n\n"
        
        if invalid_params:
            error_msg += f"Invalid parameter(s): {', '.join(invalid_params)}\n\n"
        
        error_msg += "Valid parameters for this tool:\n"
        for param, info in valid_params.items():
            is_required = " (required)" if param in required_params else " (optional)"
            description = info.get('description', 'No description')
            error_msg += f"  • {param}{is_required}: {description}\n"
        
        error_msg += "\nPlease call the tool again with only the valid parameters listed above."
        
        return error_msg
    
    def _format_missing_parameter_error(self, tool_name: str, arguments: Dict, error_str: str) -> str:
        """Format a helpful error message for missing required parameters."""
        tool = self.tools[tool_name]
        valid_params = tool.parameters.get('properties', {})
        required_params = tool.parameters.get('required', [])
        
        # Find which required parameters are missing
        provided_params = set(arguments.keys())
        missing_required = [param for param in required_params if param not in provided_params]
        
        error_msg = f"Tool call failed: {tool_name}\n\n"
        error_msg += f"Error: {error_str}\n\n"
        
        if missing_required:
            error_msg += f"Missing required parameter(s): {', '.join(missing_required)}\n\n"
        
        error_msg += "Required parameters:\n"
        for param in required_params:
            info = valid_params.get(param, {})
            description = info.get('description', 'No description')
            error_msg += f"  • {param}: {description}\n"
        
        if len(valid_params) > len(required_params):
            error_msg += "\nOptional parameters:\n"
            for param, info in valid_params.items():
                if param not in required_params:
                    description = info.get('description', 'No description')
                    error_msg += f"  • {param}: {description}\n"
        
        error_msg += "\nPlease call the tool again with all required parameters."
        
        return error_msg
    
    # ==================== Chat Interface ====================
    
    def chat(
        self,
        user_message: str,
        max_iterations: int = 5,
        temperature: float = 0.5,
        enable_tools: bool = True,
        enable_planning: bool = True,
        refine_answer: bool = True
    ) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: User's message
            max_iterations: Maximum tool calling iterations
            temperature: LLM temperature
            enable_tools: Enable tool calling
            enable_planning: Enable multi-step planning (currently unused)
            refine_answer: Enable optional answer refinement step (default: True)
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.history.add_user_message(user_message)
        
        # Dynamically select relevant tools for this query
        selected_tool_names = None
        if enable_tools and self.tools:
            selected_tool_names = self.tool_selector.select_tools(
                query=user_message,
                tools=self.tools,
                max_tools=self.max_tools_in_prompt
            )
            logger.debug(f"Selected {len(selected_tool_names)} tools for this query")
            
            # Update system message with only selected tools
            self._update_system_message(selected_tool_names)
        
        # Get current conversation messages (including updated system message)
        all_messages = [dict(m) for m in self.history.get_messages()]
        
        # Limit history in prompt while keeping system message
        # System message is always first, then limit the recent conversation
        if len(all_messages) > self.max_history_in_prompt + 1:  # +1 for system message
            # Keep system message + most recent N messages
            messages = [all_messages[0]] + all_messages[-(self.max_history_in_prompt):]
            logger.debug(f"Limited prompt to system + {self.max_history_in_prompt} recent messages (from {len(all_messages)-1} total)")
        else:
            messages = all_messages
        
        # Prepare tools - only selected ones
        if enable_tools and self.tools and selected_tool_names:
            selected_tools = [self.tools[name].to_llm_format() for name in selected_tool_names if name in self.tools]
            tools = selected_tools
        else:
            tools = None
        
        # Run chat engine
        final_response, new_messages = self.chat_engine.run(
            messages=messages,
            tools=tools,
            max_iterations=max_iterations,
            temperature=temperature,
            max_tokens=512  # Room for detailed answer with examples
        )
        
        # Determine the final answer to return (and store in history)
        final_answer = final_response
        
        # Optional refinement step
        if refine_answer and final_response:
            logger.debug("Applying answer refinement...")
            try:
                refined_response = self.model_handler.refine_answer(
                    original_query=user_message,
                    raw_response=final_response,
                    temperature=0.3
                )
                logger.info("Answer refinement completed successfully")
                final_answer = refined_response
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Answer refinement failed: {error_msg}")
                
                # Log specific error types for debugging
                if "llama_decode" in error_msg:
                    logger.warning("Context window may be too full. Consider using --no-refine-answers for long conversations.")
                elif "out of memory" in error_msg.lower():
                    logger.warning("Out of memory during refinement. Try reducing context size or disabling refinement.")
                
                # Use original response as fallback
                logger.info("Returning original unrefined response")
                final_answer = final_response
        
        # Add messages to history with the final answer
        self._add_messages_to_history(new_messages, final_answer)
        
        return final_answer
    
    def _add_messages_to_history(self, messages: List[Dict], final_answer: str = None):
        """
        Add messages from chat engine to conversation history.
        
        Args:
            messages: List of messages from chat engine
            final_answer: The final answer to store (if different from raw response due to refinement)
        """
        for msg in messages:
            role = msg['role']
            content = msg.get('content', '')
            
            if role == 'assistant':
                tool_calls = msg.get('tool_calls')
                # Use final_answer if provided, otherwise use the original content
                assistant_content = final_answer if final_answer is not None else content
                self.history.add_assistant_message(assistant_content, tool_calls=tool_calls)
            elif role == 'tool':
                self.history.add_tool_message(
                    name=msg.get('name'),
                    content=content,
                    tool_call_id=msg.get('tool_call_id')
                )
    
    def plan_and_execute(self, user_message: str) -> str:
        """
        Create a plan for complex queries and execute it step by step.
        
        Args:
            user_message: User's complex query
            
        Returns:
            Final response after executing the plan
        """
        # First, ask the model to create a plan
        planning_prompt = f"""
User query: {user_message}

Please create a step-by-step plan to answer this query. Consider:
1. What information do you need?
2. What tools can help?
3. What order should steps be executed?

Format your plan as numbered steps, then execute each step.
"""
        
        return self.chat(planning_prompt, enable_planning=True, enable_tools=True)
    
    def get_history_summary(self) -> str:
        """Get a summary of the conversation."""
        return self.history.get_summary()
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")
    
    def export_conversation(self) -> List[Dict]:
        """Export conversation for saving/analysis."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "name": m.name
            }
            for m in self.history.messages
        ]

