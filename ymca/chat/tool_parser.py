"""
Tool Call Parser - Parse tool calls from various LLM formats.

Supports:
- OpenAI format (native from llama-cpp-python)
- Granite XML format
"""

import json
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


def parse_xml_tool_calls(content: str) -> List[Dict]:
    """
    Parse tool calls from Granite XML format with error recovery.
    
    Granite models output tool calls as:
    <tool_call>
    {"name": "tool_name", "arguments": "{...}"}
    </tool_call>
    
    This parser is robust against common LLM mistakes:
    - Double opening tags: <tool_call><tool_call>{...}
    - Extra whitespace/newlines
    - Minor JSON formatting issues
    
    Args:
        content: Response content that may contain XML tool calls
        
    Returns:
        List of tool call dicts in OpenAI format with 'id', 'type', and 'function'
    """
    tool_calls = []
    
    # More flexible pattern - handles multiple opening tags
    # Matches content between first <tool_call> and last </tool_call>
    pattern = r'<tool_call>+\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches):
        try:
            # Clean up the match
            cleaned = _clean_tool_call_content(match)
            
            # Try to parse JSON
            tool_data = json.loads(cleaned)
            
            # Validate required fields
            if 'name' not in tool_data:
                logger.warning(f"Tool call missing 'name' field, skipping: {cleaned[:100]}")
                continue
            
            # Convert to OpenAI format
            tool_calls.append({
                'id': f'call_{i}',
                'type': 'function',
                'function': {
                    'name': tool_data.get('name', ''),
                    'arguments': tool_data.get('arguments', '{}')
                }
            })
            logger.info(f"Parsed tool call: {tool_data.get('name')}")
            
        except json.JSONDecodeError as e:
            # Try to recover by extracting JSON more aggressively
            recovered = _extract_json_from_text(match)
            if recovered:
                try:
                    tool_data = json.loads(recovered)
                    tool_calls.append({
                        'id': f'call_{i}',
                        'type': 'function',
                        'function': {
                            'name': tool_data.get('name', ''),
                            'arguments': tool_data.get('arguments', '{}')
                        }
                    })
                    logger.warning(f"Recovered malformed tool call: {tool_data.get('name')}")
                    continue
                except:
                    pass
            
            logger.error(f"Failed to parse tool call JSON: {e}\nContent: {match[:200]}")
            continue
    
    return tool_calls


def _clean_tool_call_content(text: str) -> str:
    """
    Clean up tool call content for JSON parsing.
    
    Removes:
    - Extra <tool_call> tags that might be duplicated
    - Excessive whitespace
    - Common formatting issues
    
    Args:
        text: Raw tool call content
        
    Returns:
        Cleaned content ready for JSON parsing
    """
    # Remove any remaining <tool_call> or </tool_call> tags
    cleaned = re.sub(r'</?tool_call>', '', text)
    
    # Strip whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_json_from_text(text: str) -> str:
    """
    Aggressively extract JSON from text with potential formatting issues.
    
    Finds the outermost {} braces and extracts content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON string or empty string if not found
    """
    # Find first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    
    return ""


def has_incomplete_tool_call(content: str) -> bool:
    """
    Check if content has incomplete tool call (opening tag but no complete structure).
    
    Args:
        content: Response content to check
        
    Returns:
        True if there's an incomplete tool call
    """
    return '<tool_call>' in content and '</tool_call>' not in content

