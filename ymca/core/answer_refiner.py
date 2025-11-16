"""
Answer Refiner - Post-process LLM responses for clarity and accuracy.

This module provides functionality to refine raw LLM responses by:
- Improving clarity and formatting
- Prioritizing practical examples
- Ensuring answers directly address the original question
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerRefiner:
    """
    Refines raw LLM responses to improve clarity while preserving accuracy.
    
    The refiner takes a raw response and polishes it with better formatting,
    structure, and readability while ensuring technical accuracy and that
    the answer directly addresses the user's original question.
    """
    
    def __init__(self, llm):
        """
        Initialize the answer refiner.
        
        Args:
            llm: Llama instance for refinement generation
        """
        self.llm = llm
    
    def refine(
        self, 
        original_query: str, 
        raw_response: str, 
        temperature: float = 0.3,
        reset_state_fn: Optional[callable] = None
    ) -> str:
        """
        Refine a raw LLM response to improve clarity, formatting, and readability.
        
        This optional post-processing step takes the original user query and raw response,
        then asks the LLM to polish the answer while:
        - Ensuring it directly answers the original question
        - Prioritizing practical examples and technical details
        - Improving formatting and structure
        - Preserving complete technical accuracy
        
        Args:
            original_query: The user's original question
            raw_response: The raw response from the LLM
            temperature: Lower temperature for more consistent refinement (default: 0.3)
            reset_state_fn: Optional function to reset model state before refinement
            
        Returns:
            Refined response string
            
        Raises:
            Exception: If refinement fails (caller should handle and return original)
        """
        # Reset model state to clear KV cache from previous interactions
        # This prevents context overflow issues during refinement
        if reset_state_fn:
            try:
                reset_state_fn(deep_clean=False)
            except Exception as e:
                logger.warning(f"Failed to reset model state before refinement: {e}")
        
        # Truncate very long responses to prevent context overflow
        max_response_length = 8000  # chars (~2000 tokens) - increased to handle longer responses
        if len(raw_response) > max_response_length:
            logger.warning(
                f"Raw response too long ({len(raw_response)} chars), "
                f"truncating to {max_response_length}"
            )
            raw_response = raw_response[:max_response_length] + "\n\n[Response truncated for refinement]"
        
        refinement_prompt = self._build_refinement_prompt(original_query, raw_response)
        
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": refinement_prompt}],
            max_tokens=1024,  # Allow room for well-formatted response with examples
            temperature=temperature,
            stop=None
        )
        
        refined_text = response['choices'][0]['message']['content'].strip()
        logger.debug(f"Refined response ({len(raw_response)} → {len(refined_text)} chars)")
        
        return refined_text
    
    def _build_refinement_prompt(self, original_query: str, raw_response: str) -> str:
        """
        Build the refinement prompt with emphasis on examples and answering the actual question.
        
        Args:
            original_query: The user's original question
            raw_response: The raw response to refine
            
        Returns:
            Formatted refinement prompt
        """
        return f"""You are an expert technical writer. Your task is to refine and improve the following response while maintaining complete technical accuracy.

Original User Query:
{original_query}

Raw Response:
{raw_response}

CRITICAL REFINEMENT RULES:

1. **Be Brief But Complete**:
   - Keep the answer SHORT - aim for 10-15 lines total including examples
   - Get to the point immediately - no long introductions
   - Remove ALL unnecessary words and filler text
   - But DO keep all examples and technical details (these are essential)
   - Think: "What's the minimum needed to answer + show how?"

2. **Answer the ACTUAL Question**: 
   - Address EXACTLY what the user asked - not a similar or related question
   - If the user asks "How do I X?", show how to do X specifically
   - If the user asks "What is Y?", explain Y, not something else
   - Stay laser-focused on the original query

3. **Prioritize Practical Examples**:
   - Include concrete, runnable examples for every key point
   - Show actual commands, code, or tool calls with real parameters
   - Provide multiple examples when helpful (before/after, simple/advanced)
   - Examples should be DETAILED and COMPLETE, not abbreviated

4. **Preserve Technical Details**:
   - Keep ALL technical facts, commands, parameters, and specifications
   - Maintain exact parameter names, values, and syntax
   - Include all prerequisites, constraints, and important notes
   - Do NOT simplify away important technical nuances

5. **Improve Structure and Clarity**:
   - Use clear markdown formatting (lists, code blocks) use paragraphs and try to avoid headers
   - Organize information logically (overview → examples → details)
   - Make it easy to scan and find information quickly
   - Use bold/italic for emphasis on key terms

6. **Format Code Properly**:
   - Use proper code blocks with language tags
   - Include inline code formatting for parameters and commands
   - Ensure examples are properly indented and readable

CRITICAL OUTPUT RULES:
- Output ONLY the refined answer itself - nothing else
- Do NOT include any preamble like "Here's the refined version" or "The answer is:"
- Do NOT repeat the question or prompt
- Do NOT add meta-commentary like "I've refined this to..." or "This addresses..."
- Do NOT explain what you did or how you improved it
- Start IMMEDIATELY with the actual answer to: "{original_query}"
- If the answer needs code, start with the explanation or code directly
- Think: "If I were writing this in a document, what would the reader see?" - they should see ONLY the answer
"""

