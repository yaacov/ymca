"""Synthesis tools for combining and analyzing information."""

from typing import Any, Callable, Dict, List, Tuple, cast


def create_synthesis_tools(llm: Any) -> List[Tuple[Dict[str, Any], Callable[..., Any]]]:
    """Create synthesis tools for combining information."""
    tools = []

    # Synthesize information tool
    synthesize_tool = {
        "name": "synthesize_information",
        "description": (
            "Synthesize and combine information from multiple sources into a "
            "comprehensive, well-structured answer. Use this after gathering "
            "information from web searches, memory searches, or file reads to "
            "create a coherent final response. Particularly useful when you have "
            "collected data from multiple tools and need to organize it into a "
            "clear, comprehensive answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {"type": "string", "description": "The original question or topic to be addressed"},
                "information_sources": {"type": "string", "description": "Raw information gathered from previous tool calls (web searches, memory searches, file reads, etc.)"},
            },
            "required": ["original_question"],
        },
        "category": "synthesis",
        "enabled": True,
    }

    async def synthesize_handler(original_question: str, information_sources: str = "") -> str:
        """Synthesize information using LLM to create comprehensive answer."""
        if not information_sources or not information_sources.strip():
            return f"No information available to synthesize for: {original_question}"

        synthesis_prompt = f"""Based on the gathered information, provide a comprehensive answer to the question.

Question: {original_question}

Gathered Information:
{information_sources}

Please synthesize this information into a clear, well-structured, and comprehensive answer.
Focus on accuracy and completeness while organizing the information logically.

Comprehensive Answer:"""

        try:
            # Generate context for debug filename
            safe_question = "".join(c for c in original_question[:30] if c.isalnum() or c in " _-").replace(" ", "_")
            context_desc = f"synthesis_{safe_question}"
            
            synthesized_answer, debug_filename = llm.generate_response_with_debug(synthesis_prompt, context_desc)
            result = str(synthesized_answer).strip()
            
            # Add debug info to result if available (for logging)
            if debug_filename:
                result = f"{result}\n\n[Debug: Synthesis prompt saved to {debug_filename}]"
            
            return result
        except Exception as e:
            return f"Synthesis failed: {e}. Raw information: {information_sources[:500]}..."

    tools.append((synthesize_tool, cast(Callable[..., Any], synthesize_handler)))
    return tools
