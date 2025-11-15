"""
Model Handler - Manages LLM for question generation and other tasks.
"""

from pathlib import Path
from typing import List, Optional
import logging
import gc
import time

from llama_cpp import Llama
from .question_parser import parse_questions_from_text

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handle LLM operations for memory tool."""
    
    def __init__(self, model_path: str, n_ctx: int = 32768, n_threads: int = 4, n_gpu_layers: int = -1):
        """
        Initialize the model handler.
        
        Args:
            model_path: Path to GGUF model (required)
            n_ctx: Context size (default: 32768, Granite 4.0 supports up to 128K but GGUF may have limits)
            n_threads: Number of threads (default: 4)
            n_gpu_layers: Number of layers to offload to GPU (default: -1 = all layers, 0 = CPU only)
                         Falls back to CPU if GPU support is not available
        """
        if not model_path:
            raise ValueError("model_path is required")
        
        if not Path(model_path).exists():
            raise ValueError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {Path(model_path).name}...")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # -1 = all layers to GPU, 0 = CPU only
            verbose=False
        )
        
        # Log acceleration info
        gpu_info = "GPU acceleration enabled" if n_gpu_layers != 0 else "CPU only"
        if n_gpu_layers != 0:
            gpu_info += " (will fall back to CPU if GPU unavailable)"
        logger.info(f"Model loaded (context: {n_ctx} tokens, {gpu_info})")
    
    def reset_state(self, deep_clean: bool = False):
        """
        Reset the model's KV cache to clear any accumulated state.
        
        Args:
            deep_clean: If True, also run garbage collection and brief pause
        """
        try:
            # Reset llama.cpp state
            self.llm.reset()
            logger.debug("Model state reset")
        except AttributeError:
            # Fallback for older llama-cpp-python versions
            logger.debug("Model reset not available")
            pass
        
        # Deep clean if requested
        if deep_clean:
            # Force garbage collection to free memory
            gc.collect()
            # Brief pause to let system settle
            time.sleep(0.1)
            logger.debug("Deep clean completed")
    
    def generate_questions(self, chunk: str, num_questions: int = 3) -> List[str]:
        """
        Generate semantic summaries for the chunk (optimized for retrieval).
        
        Uses the LLM to generate declarative statements that capture key information.
        These summaries are embedded and matched against user queries for retrieval.
        
        Args:
            chunk: Text chunk to generate summaries for
            num_questions: Number of semantic summaries to generate (default: 3)
            
        Returns:
            List of generated semantic summaries (declarative statements)
            
        Raises:
            ValueError: If parsing fails or not enough summaries generated
            Exception: If LLM call fails
        """
        prompt = self._build_question_prompt(chunk, num_questions)
        
        # Generate semantic summaries via LLM
        # Note: max_tokens needs to be high enough for detailed summaries
        # Detailed summaries (15-30 words) ~= 20-40 tokens each
        # For 2-3 summaries: 40-120 tokens + overhead = ~150-250 tokens needed
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Increased to support detailed summaries (15-30 words each)
            temperature=0.2,  # Low temperature for technical accuracy and consistency
            stop=["\n\n", "Text:", "---", "Instructions:"],  # Stop tokens to prevent rambling
            repeat_penalty=1.1  # Reduce repetition
        )
        
        # Extract and parse generated text (will raise ValueError if insufficient)
        generated_text = response['choices'][0]['message']['content'].strip()
        summaries = parse_questions_from_text(generated_text, num_questions)
        
        # Validate and log summary quality
        for i, summary in enumerate(summaries, 1):
            word_count = len(summary.split())
            if word_count < 10:
                logger.warning(f"Summary {i} is very short ({word_count} words): {summary}")
            elif word_count > 40:
                logger.warning(f"Summary {i} is very long ({word_count} words): {summary}")
        
        return summaries
    
    def _build_question_prompt(self, chunk: str, num_questions: int) -> str:
        """
        Build prompt for semantic summary generation optimized for retrieval.
        
        Creates declarative statements that capture the key information in the chunk.
        These summaries are semantically rich and match well against user queries.
        
        Args:
            chunk: Text chunk
            num_questions: Number of semantic summaries to generate
            
        Returns:
            Formatted prompt string
        """
        return f"""Given the following technical documentation, generate exactly {num_questions} comprehensive semantic summaries that capture the key information.

Text:
{chunk}

Instructions:
- Generate exactly {num_questions} DECLARATIVE statements that summarize DIFFERENT aspects of the content
- Each statement should be a complete, self-contained description of a key concept or fact
- Focus on WHAT, HOW, WHY aspects: what things are, how they work, why they matter
- Include specific technical details: commands, parameters, configurations, procedures
- Each statement should capture unique information that would help answer user queries
- If the text contains procedures, describe the steps and their purpose
- If the text contains lists, summarize what the list represents and key patterns
- If the text contains configurations, describe what they control and their effects
- Statements should be rich and detailed (15-30 words each)
- Each statement must be on a separate line
- Do not number the statements
- Use present tense and be factual

Semantic summaries:"""
    

    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.2) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def chat(
        self, 
        messages: List[dict], 
        max_tokens: int = 200, 
        temperature: float = 0.2,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None
    ) -> dict:
        """
        Chat completion with optional tool calling support.
        
        Args:
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional list of tool definitions for function calling
            tool_choice: Tool selection mode ("auto", "none", or specific tool name)
            
        Returns:
            Full response dictionary from llama_cpp
        """
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        
        return self.llm.create_chat_completion(**kwargs)
    
    def refine_answer(self, original_query: str, raw_response: str, temperature: float = 0.3) -> str:
        """
        Refine a raw LLM response to improve clarity, formatting, and readability.
        
        This optional post-processing step takes the original user query and raw response,
        then asks the LLM to polish the answer while preserving technical accuracy.
        
        Args:
            original_query: The user's original question
            raw_response: The raw response from the LLM
            temperature: Lower temperature for more consistent refinement (default: 0.3)
            
        Returns:
            Refined response string
            
        Raises:
            Exception: If refinement fails (caller should handle and return original)
        """
        # Reset model state to clear KV cache from previous interactions
        # This prevents context overflow issues during refinement
        try:
            self.reset_state(deep_clean=False)
        except Exception as e:
            logger.warning(f"Failed to reset model state before refinement: {e}")
        
        # Truncate very long responses to prevent context overflow
        max_response_length = 2000  # chars (~500 tokens)
        if len(raw_response) > max_response_length:
            logger.warning(f"Raw response too long ({len(raw_response)} chars), truncating to {max_response_length}")
            raw_response = raw_response[:max_response_length] + "\n\n[Response truncated for refinement]"
        
        refinement_prompt = f"""You are an expert technical writer. Your task is to refine and improve the following response while maintaining complete technical accuracy.

Original User Query:
{original_query}

Raw Response:
{raw_response}

Instructions for refinement:
1. **Improve Clarity**: Make the response easier to understand without losing technical precision
2. **Better Formatting**: Use clear structure with appropriate markdown (headers, lists, code blocks)
3. **Conciseness**: Remove redundancy while keeping all essential information
4. **Polish Language**: Improve readability and flow
5. **Preserve Accuracy**: Do NOT change any technical facts, commands, parameters, or examples
6. **Keep Examples**: Maintain all code examples and technical details exactly as provided

Output ONLY the refined response, with no preamble or explanation. Begin immediately with the improved answer."""

        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": refinement_prompt}],
            max_tokens=1024,  # Allow room for well-formatted response
            temperature=temperature,
            stop=None
        )
        
        refined_text = response['choices'][0]['message']['content'].strip()
        logger.debug(f"Refined response ({len(raw_response)} → {len(refined_text)} chars)")
        
        return refined_text