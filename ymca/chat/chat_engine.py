"""
Chat Engine - Core chat loop and LLM interaction logic.

Handles the iterative tool-calling loop, message preparation, 
and interaction with the LLM.
"""

import json
import logging
from typing import Dict, List, Optional, Callable

from .tool_parser import parse_xml_tool_calls, has_incomplete_tool_call

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Core chat engine handling LLM interaction and tool execution loop.
    """
    
    def __init__(self, model_handler, tool_executor: Callable):
        """
        Initialize the chat engine.
        
        Args:
            model_handler: ModelHandler instance for LLM
            tool_executor: Function to execute tools (name, args) -> result
        """
        self.model_handler = model_handler
        self.execute_tool = tool_executor
    
    def run(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        max_iterations: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 256
    ) -> tuple[str, List[Dict]]:
        """
        Run the chat loop with tool calling support.
        
        Args:
            messages: Conversation messages
            tools: Available tools in LLM format
            max_iterations: Maximum tool calling iterations
            temperature: LLM temperature
            max_tokens: Maximum tokens for completion
            
        Returns:
            Tuple of (final_response, new_messages_to_add_to_history)
        """
        iteration = 0
        new_messages = []
        final_response = ""
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prepare API call
            kwargs = self._prepare_llm_kwargs(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Log context info
            self._log_context_info(messages, kwargs, iteration)
            
            # Reset model state before each iteration to avoid KV cache contamination
            # This ensures clean state for both new queries and tool result processing
            self._reset_model_state()
            
            try:
                # Get LLM response
                content, tool_calls = self._get_llm_response(kwargs)
                
                # Handle tool calls
                if tool_calls:
                    tool_messages = self._handle_tool_calls(content, tool_calls)
                    new_messages.extend(tool_messages)
                    
                    # Add tool messages to context for next iteration
                    messages.extend(tool_messages)
                    continue
                
                # No tool calls - we have the final response
                final_response = content
                new_messages.append({
                    "role": "assistant",
                    "content": content
                })
                break
                
            except Exception as e:
                error_response = self._handle_error(e, iteration, messages)
                new_messages.append({
                    "role": "assistant",
                    "content": error_response
                })
                final_response = error_response
                break
        
        # Handle max iterations
        if iteration >= max_iterations and not final_response:
            final_response = "I've reached the maximum number of steps. Let me summarize what I found..."
            new_messages.append({
                "role": "assistant",
                "content": final_response
            })
        
        # Log final usage statistics
        self._log_usage_stats(messages, new_messages, iteration)
        
        return final_response, new_messages
    
    def _prepare_llm_kwargs(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """Prepare kwargs for LLM API call."""
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Don't pass tools parameter - Granite models don't support OpenAI-style tool calling
        # Tool descriptions are already in the system prompt
        if tools and logger.isEnabledFor(logging.DEBUG):
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
            logger.debug(f"Available tools (in prompt only): {tool_names}")
        
        return kwargs
    
    def _log_context_info(self, messages: List[Dict], kwargs: Dict, iteration: int):
        """Log context and token information for debugging."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        approx_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
        model_ctx = self.model_handler.llm.n_ctx()
        
        logger.debug(
            f"Iteration {iteration}: ~{approx_tokens} tokens in context "
            f"({total_chars} chars), {kwargs['max_tokens']} reserved, "
            f"model has {model_ctx} available"
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            for i, msg in enumerate(messages):
                content_preview = str(msg.get("content", ""))[:100]
                logger.debug(f"  Message {i} ({msg['role']}): {content_preview}...")
    
    def _log_usage_stats(self, messages: List[Dict], new_messages: List[Dict], iterations: int):
        """Log usage statistics after response generation."""
        # Count messages by type
        tool_calls = sum(1 for m in new_messages if m.get('role') == 'assistant' and m.get('tool_calls'))
        tool_responses = sum(1 for m in new_messages if m.get('role') == 'tool')
        
        # Calculate token usage
        all_messages = messages + new_messages
        total_chars = sum(len(str(m.get("content", ""))) for m in all_messages)
        approx_tokens = total_chars // 4
        model_ctx = self.model_handler.llm.n_ctx()
        
        # Count history vs new content
        history_chars = sum(len(str(m.get("content", ""))) for m in messages)
        new_chars = sum(len(str(m.get("content", ""))) for m in new_messages)
        history_tokens = history_chars // 4
        new_tokens = new_chars // 4
        
        # Only print usage statistics in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            print(f"\n📊 Usage Statistics:")
            print(f"   Iterations: {iterations}")
            print(f"   Tool calls: {tool_calls}")
            print(f"   History: ~{history_tokens:,} tokens ({len(messages)} messages)")
            print(f"   New content: ~{new_tokens:,} tokens ({len(new_messages)} messages)")
            print(f"   Total: ~{approx_tokens:,} / {model_ctx:,} tokens ({(approx_tokens/model_ctx*100):.1f}% used)\n")
    
    def _reset_model_state(self):
        """Reset model state to avoid KV cache corruption."""
        try:
            self.model_handler.llm.reset()
        except AttributeError:
            # reset() might not be available in all versions
            pass
    
    def _get_llm_response(self, kwargs: Dict) -> tuple[str, List[Dict]]:
        """
        Get response from LLM and parse tool calls.
        
        Returns:
            Tuple of (content, tool_calls)
        """
        response = self.model_handler.llm.create_chat_completion(**kwargs)
        choice = response['choices'][0]
        message = choice['message']
        content = message.get('content', '')
        
        # Debug: Show LLM response
        if logger.isEnabledFor(logging.DEBUG):
            content_preview = content[:300] + ("..." if len(content) > 300 else "")
            logger.debug(f"LLM response (first 300 chars): {content_preview}")
            logger.debug(f"Response has 'tool_calls' field: {'tool_calls' in message}")
            logger.debug(f"Response contains '<tool_call>': {'<tool_call>' in content}")
        
        # Check for tool calls in both OpenAI format and Granite XML format
        tool_calls = self._parse_tool_calls(message, content, kwargs)
        
        if logger.isEnabledFor(logging.DEBUG):
            if tool_calls:
                logger.debug(f"Parsed {len(tool_calls)} tool call(s)")
                for i, tc in enumerate(tool_calls):
                    logger.debug(f"  Tool call {i+1}: {tc['function']['name']}")
            else:
                logger.debug("No tool calls detected")
        
        return content, tool_calls
    
    def _parse_tool_calls(
        self,
        message: Dict,
        content: str,
        kwargs: Dict
    ) -> List[Dict]:
        """
        Parse tool calls from LLM response.
        Supports both OpenAI format and Granite XML format.
        """
        tool_calls = []
        
        # Check OpenAI format first
        if 'tool_calls' in message and message['tool_calls']:
            return message['tool_calls']
        
        # Check Granite XML format
        if content and '<tool_call>' in content:
            tool_calls = parse_xml_tool_calls(content)
            
            # Handle incomplete tool calls
            if not tool_calls and has_incomplete_tool_call(content):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Incomplete tool call detected. First response (full): {content}")
                tool_calls = self._retry_incomplete_tool_call(kwargs, content)
        
        return tool_calls
    
    def _retry_incomplete_tool_call(self, kwargs: Dict, first_attempt: str = "") -> List[Dict]:
        """Retry LLM call with more tokens for incomplete tool calls."""
        logger.warning("Detected incomplete tool call, retrying with more tokens")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"First attempt length: {len(first_attempt)} chars")
            logger.debug(f"First attempt ends with: ...{first_attempt[-100:] if len(first_attempt) > 100 else first_attempt}")
        
        try:
            # Reset model state before retry to avoid decode errors
            self._reset_model_state()
            
            kwargs["max_tokens"] = 1536  # Give it plenty of room for tool call + arguments
            response = self.model_handler.llm.create_chat_completion(**kwargs)
            choice = response['choices'][0]
            message = choice['message']
            content = message.get('content', '')
            
            # Debug: Show full retry response
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Retry response length: {len(content)} chars")
                logger.debug(f"Retry response (full): {content}")
            
            if '<tool_call>' in content:
                parsed = parse_xml_tool_calls(content)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Parsed {len(parsed)} tool calls from retry")
                return parsed
            else:
                logger.warning("Retry response does not contain <tool_call>")
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            # Return empty list to fall back to regular response
        
        return []
    
    def _handle_tool_calls(self, content: str, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute tool calls and return messages to add to history.
        
        Returns:
            List of messages (assistant message + tool response messages)
        """
        messages = []
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls
        })
        
        # Execute each tool
        for tool_call in tool_calls:
            func = tool_call['function']
            name = func['name']
            
            # Parse arguments (handle both string and dict)
            arguments = self._parse_tool_arguments(func)
            
            # Execute tool
            logger.info(f"Calling tool: {name}")
            result = self.execute_tool(name, arguments)
            
            # Debug: Show tool call details and result (only in debug mode)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔧 Tool: {name}")
                logger.debug(f"  Tool arguments: {arguments}")
                logger.debug(f"  Tool result ({len(str(result))} chars):")
                logger.debug(f"  {result}")
            
            # Add tool response message
            tool_message = {
                "role": "tool",
                "content": str(result),
                "name": name,
                "tool_call_id": tool_call['id']
            }
            messages.append(tool_message)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Added tool response to context ({len(str(result))} chars)")
        
        return messages
    
    def _parse_tool_arguments(self, func: Dict) -> Dict:
        """Parse tool arguments from function call."""
        args_str = func.get('arguments', '{}')
        
        if isinstance(args_str, str):
            return json.loads(args_str)
        else:
            return args_str
    
    def _handle_error(self, error: Exception, iteration: int, messages: List[Dict]) -> str:
        """
        Handle errors during chat execution.
        
        Returns:
            Error message to show to user
        """
        logger.error(f"Chat error: {error}")
        error_str = str(error)
        
        # Handle llama_decode errors
        if "llama_decode" in error_str:
            self._handle_context_error()
            
            # Check if it's actually a context overflow or something else
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            approx_tokens = total_chars // 4
            model_ctx = self.model_handler.llm.n_ctx()
            
            # If we're actually near the context limit, show context error
            if approx_tokens > model_ctx * 0.8:  # 80% of context
                return self._create_context_error_message(messages)
            
            # Otherwise, it's likely a model/decode issue
            if iteration > 1:
                return "I've completed the tool operations. The information has been processed."
            else:
                return (
                    f"⚠️  Model decode error (this is not a context size issue)\n"
                    f"   Tokens in use: ~{approx_tokens:,} / {model_ctx:,}\n\n"
                    f"Possible causes:\n"
                    f"  • Model compatibility issue\n"
                    f"  • Tool call format issue\n"
                    f"  • Model needs to be reset\n\n"
                    f"Try: 'clear' to reset conversation and try again"
                )
        
        # Handle actual context overflow errors
        if "context" in error_str.lower() and "size" in error_str.lower():
            return self._create_context_error_message(messages)
        
        return f"I encountered an error: {str(error)}"
    
    def _handle_context_error(self):
        """Handle context-related errors by resetting model state."""
        try:
            self.model_handler.llm.reset()
            logger.info("Reset model state after decode error")
        except:
            pass
    
    def _create_context_error_message(self, messages: List[Dict]) -> str:
        """Create detailed error message for context issues."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        approx_tokens = total_chars // 4
        model_ctx = self.model_handler.llm.n_ctx()
        
        return (
            f"⚠️  Context issue detected!\n"
            f"   Estimated tokens: ~{approx_tokens}\n"
            f"   Model context: {model_ctx:,} tokens\n"
            f"   Messages: {len(messages)}\n\n"
            f"Try: 'clear' to reset conversation"
        )

