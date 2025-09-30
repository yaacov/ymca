"""Plan response processing and result cleaning."""

import logging
from typing import Optional

from ..llm.llm import LLM
from .models import Plan


class PlanResponseProcessor:
    """Handles plan response generation and result cleaning for user-friendly output."""

    def __init__(
        self,
        llm: LLM,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm = llm
        self.logger = logger or logging.getLogger("ymca.plan.response_processor")

    def generate_plan_response(self, plan: Plan) -> str:
        """Generate a comprehensive response based on the executed plan."""
        progress = plan.get_progress_summary()

        response_parts = []

        # Clean, direct result presentation
        result_content = None
        if plan.result:
            result_content = plan.result
        elif plan.evolving_answer:
            result_content = plan.evolving_answer

        if result_content:
            # Ensure the result is user-friendly and not raw technical data
            cleaned_result = self.ensure_user_friendly_result(plan.description, result_content, plan)
            
            # Always present the answer directly - let the synthesis handle proper formatting
            response_parts.append(cleaned_result)
        
        response_parts.append("")
        response_parts.append("---")
        response_parts.append(f"✅ Plan completed successfully")
        response_parts.append(f"📊 Progress: {progress['completed']}/{progress['total_steps']} steps ({progress['progress_percentage']:.1f}%)")
        
        if progress["failed"] > 0:
            response_parts.append(f"⚠️ Warning: {progress['failed']} steps failed")

        response_parts.append(f"🆔 Plan ID: {plan.id} (saved for future reference)")

        return "\n".join(response_parts)

    def _clean_debug_markers(self, text: str) -> str:
        """Remove debug markers and technical formatting from text."""
        import re
        
        # Remove debug patterns
        debug_patterns = [
            r'🔎 CRITICAL EVALUATION:.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\*\*NEW INPUT:\*\*.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\*\*Reasoning:\*\*.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\*\*Action:\*\*.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\*\*Parameters:\*\*.*?(?=\n\n|\n[A-Z]|\Z)',
            r'Finding \d+:.*?(?=\n\n|\n[A-Z]|\Z)',
            r'DISCOVERED URLS.*?(?=\n\n|\n[A-Z]|\Z)',
            r'Current accumulated information:.*?(?=\n\n|\n[A-Z]|\Z)',
            r'Search Results for:.*?(?=\n\n|\n[A-Z]|\Z)',
            r'- Current accumulated information.*?(?=\n|\Z)',
            r'- Task: .*?(?=\n|\Z)',
        ]
        
        cleaned_text = text
        for pattern in debug_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
        
        # Clean up multiple newlines and whitespace
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # If we're left with very little content, return a helpful message
        if len(cleaned_text) < 20:
            return "I found some relevant information but need to gather more details to provide a complete answer."
        
        return cleaned_text

    def ensure_user_friendly_result(self, original_question: str, raw_result: str, plan: Optional[Plan] = None) -> str:
        """Convert technical/raw data into a clean, user-friendly answer."""
        
        # First, detect potential hallucination patterns
        hallucination_risk = self._detect_hallucination_risk(original_question, raw_result)
        if hallucination_risk["is_likely_hallucination"]:
            return self._handle_potential_hallucination(original_question, raw_result, hallucination_risk)
        
        # Check if the result looks like raw data that needs synthesis
        raw_data_indicators = [
            # Debug output patterns (CRITICAL to filter)
            "🔎 CRITICAL EVALUATION:",
            "**NEW INPUT:**",
            "**Reasoning:**", 
            "**Action:**",
            "**Parameters:**",
            "Finding 1:", "Finding 2:", "Finding 3:",
            "DISCOVERED URLS (ready for read_webpage):",
            "Current accumulated information:",
            "Search Results for:",
            "DuckDuckGo",
            # Web content indicators
            "Smart Search Results for:",
            "URL:",
            "[ ](",  # HTML links
            "Opens a new window",
            "Main Versions  Licenses",
            "You must be signed in",
            "</",  # HTML tags
            ">",  # More HTML
            # Multi-source result indicators
            "...\n\n2.",  # Multiple numbered results
            "...\n\n3.",
            "1. ",
            "2. ",
            "3. ",  # Numbered list format from tools
            # File/data structure indicators
            "File: /",
            "Directory: /",
            "Error reading file:",
            "Stored in memory with ID:",
            # API/JSON indicators
            '{"',
            '"success":',
            '"result":',
            '"error":',
            # Technical output patterns
            "Tool '",
            "Action '",
            "succeeded in",
            "failed after",
        ]

        # Also check for overly technical language vs user-friendly responses
        technical_patterns = [
            "based on the research for",  # Our own tool output format
            "gathered information:",
            "search results:",
            "tool execution",
            "call_id:",
        ]

        # Check length and structure - very long responses with lots of technical details often need synthesis
        is_very_long = len(raw_result) > 800
        has_raw_indicators = any(indicator in raw_result for indicator in raw_data_indicators)
        has_technical_patterns = any(pattern in raw_result.lower() for pattern in technical_patterns)
        has_multiple_sections = raw_result.count("\n\n") > 3

        needs_synthesis = has_raw_indicators or has_technical_patterns or (is_very_long and has_multiple_sections)

        if not needs_synthesis:
            # Result looks clean, but still clean any debug markers
            return self._clean_debug_markers(raw_result)
        elif len(raw_result) < 200 and not has_raw_indicators:
            # Too short to need synthesis but clean debug markers
            return self._clean_debug_markers(raw_result)

        self.logger.info("🧹 Converting technical/raw data to user-friendly format...")

        try:
            # Build execution context if plan is available
            context_info = ""
            if plan:
                tools_used = [exec.action for exec in plan.executions if exec.action]
                context_info = f"""

EXECUTION CONTEXT:
- Total iterations: {len(plan.executions)}
- Tools used: {', '.join(tools_used) if tools_used else 'None (no tools were executed)'}
- Task completed in: {len(plan.executions)} iteration(s)
- URL tracking: {len(plan.read_urls)} URLs were processed
"""

            # Simple, direct answer guidance
            tone_guidance = """
🎯 ANSWER REQUIREMENTS:
- Give the direct answer to what they asked - nothing more
- No placeholder text like <replace-this> - use realistic examples
- No "Important Notes" or "Caveats" sections unless critical
- No links to documentation unless they specifically help
- Write as if you're quickly helping a colleague, not writing documentation"""

            synthesis_prompt = f"""The user asked: "{original_question}"

I used an automated planning system that executed various tools (web search, file reading, databases, APIs, etc.) to gather information, but the result contains raw data, technical output, or formatting that makes it difficult for a user to read and understand.
{context_info}
Raw system output:
{raw_result}

⚠️ CRITICAL SYNTHESIS REQUIREMENTS:
- DO NOT make up information that wasn't in the raw data
- NEVER invent CLI command syntax, flags, or parameters that weren't explicitly found in the raw data
- If the raw data shows the system didn't gather enough information, acknowledge this
- If tools failed or weren't used, mention this as context for why information might be incomplete
- For CLI commands, only provide syntax that was verified from actual sources, not invented
- If asked for command syntax but none was found, clearly state "No specific command syntax was found in the sources"
- Do NOT provide example commands unless they were explicitly shown in the documentation

{tone_guidance}

Transform the raw data into a clean, direct answer to their specific question.

WHAT TO DO:
- Extract exactly what they asked for from the raw data
- Present it clearly and simply 
- Use real examples, not placeholder text
- Keep it conversational and helpful

WHAT NOT TO DO:
- Don't include technical metadata, tool outputs, or raw data
- Don't add unnecessary warnings, caveats, or "Important Notes"
- Don't use placeholder text like <replace-this> or <your-value-here>
- Don't write documentation - just answer their question
- Don't explain how you found the information

Example: If they ask for a command, just give them the actual command with real values, maybe one line of context.

User-friendly answer:"""

            # Generate context for debug filename
            context_desc = f"user_friendly_{original_question[:20].replace(' ', '_')}"
            
            synthesized, debug_filename = self.llm.generate_response_with_debug(synthesis_prompt, context_desc)
            clean_result = synthesized.strip()

            if debug_filename:
                self.logger.info(f"🧹 Generated user-friendly answer ({len(clean_result)} chars, debug: {debug_filename})")
            else:
                self.logger.info(f"🧹 Generated user-friendly answer ({len(clean_result)} chars)")
            
            return clean_result

        except Exception as e:
            self.logger.error(f"🧹 User-friendly conversion failed: {e}, using fallback cleanup")
            # Fallback: try to extract just the essential info manually
            return self._fallback_result_cleanup(original_question, raw_result)

    def _fallback_result_cleanup(self, original_question: str, raw_result: str) -> str:
        """Fallback method to clean up results when user-friendly conversion fails."""

        # Try to extract key information patterns
        lines = raw_result.split("\n")
        cleaned_lines = []

        # Look for lines with important information
        important_keywords = ["version", "v0.", "v1.", "v2.", "published", "latest", "current", "available", "download", "install", "release", "update"]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip obvious navigation/technical elements
            skip_patterns = ["[ ]", "Opens a new window", "You must be signed in", "Tool '", "Action '", "succeeded in", "failed after"]
            if any(skip in line for skip in skip_patterns):
                continue

            # Keep lines with important info
            if any(keyword in line.lower() for keyword in important_keywords):
                cleaned_lines.append(line)

        if cleaned_lines:
            intro = f"Based on the gathered information about '{original_question}':"
            return f"{intro}\n\n" + "\n".join(cleaned_lines[:5])  # Limit to top 5 relevant lines
        else:
            # Last resort - return a shortened version
            if len(raw_result) > 500:
                return f"Here's what I found regarding '{original_question}':\n\n{raw_result[:400]}..."
            return raw_result

    def extract_clean_answer(self, plan: Plan) -> str:
        """Extract just the clean final answer for chat history storage."""

        # Try to get the clean final answer from the plan
        result_content = None
        if plan.result:
            self.logger.info("📚 Using plan.result for chat history")
            result_content = plan.result
        elif plan.evolving_answer:
            self.logger.info("📚 Using plan.evolving_answer for chat history")
            result_content = plan.evolving_answer
        elif plan.knowledge_pieces:
            # If we have knowledge pieces but no clean final answer,
            # use the last (most complete) knowledge piece
            self.logger.info(f"📚 Using last knowledge piece for chat history ({len(plan.knowledge_pieces)} pieces available)")
            result_content = plan.knowledge_pieces[-1]
        else:
            # Fallback - this shouldn't happen in normal operation
            self.logger.warning("📚 No clean answer found, using fallback for chat history")
            return f"I couldn't find complete information about: {plan.description}"

        # Apply the same user-friendly conversion to ensure clean chat history
        if result_content:
            return self.ensure_user_friendly_result(plan.description, result_content)

        return f"I couldn't find complete information about: {plan.description}"
    
    def _detect_hallucination_risk(self, original_question: str, raw_result: str) -> dict:
        """Detect if result might be hallucinated based on patterns and context."""
        import re
        
        # Patterns that suggest hallucination
        specific_claim_patterns = [
            r"version\s+(?:is\s+)?(\d+\.\d+\.\d+|\d+\.\d+)",  # Version claims
            r"latest\s+(?:version\s+)?(?:is\s+)?(\d+\.\d+\.\d+|\d+\.\d+)", 
            r"released\s+(?:on\s+)?(\d{4}-\d{2}-\d{2}|\w+\s+\d+,?\s+\d{4})",  # Release dates
            r"as\s+per\s+the\s+(.+?)\s+(?:releases|documentation|website)",  # "As per X" claims
        ]
        
        # Context checks
        has_search_results_only = "Search Results for:" in raw_result and "Webpage Content:" not in raw_result
        has_discovered_urls = "DISCOVERED URLS" in raw_result
        has_specific_claims = any(re.search(pattern, raw_result, re.IGNORECASE) for pattern in specific_claim_patterns)
        
        # Red flags for hallucination
        red_flags = []
        
        if has_search_results_only and has_specific_claims:
            red_flags.append("specific_claims_from_search_only")
            
        if "as per the" in raw_result.lower() and "github releases" in raw_result.lower() and "Webpage Content:" not in raw_result:
            red_flags.append("unverified_github_claims")
            
        if has_discovered_urls and has_specific_claims and "read_webpage" not in raw_result:
            red_flags.append("claims_without_reading_pages")
            
        # Version pattern check for common hallucination
        if "version" in original_question.lower() and re.search(r"\d+\.\d+\.\d+", raw_result) and not any(
            indicator in raw_result for indicator in ["Webpage Content:", "downloaded", "file content", "API response"]
        ):
            red_flags.append("version_without_source_verification")
            
        is_likely_hallucination = len(red_flags) >= 2 or "unverified_github_claims" in red_flags
        
        return {
            "is_likely_hallucination": is_likely_hallucination,
            "red_flags": red_flags,
            "confidence": len(red_flags) / 4.0,  # Scale 0-1
            "has_search_only": has_search_results_only,
            "has_specific_claims": has_specific_claims
        }
    
    def _handle_potential_hallucination(self, original_question: str, raw_result: str, hallucination_info: dict) -> str:
        """Handle cases where result might be hallucinated."""
        
        self.logger.warning(f"🚫 Potential hallucination detected: {hallucination_info['red_flags']}")
        
        # Extract what we actually found vs claimed
        found_urls = []
        if "DISCOVERED URLS" in raw_result:
            urls_section = raw_result[raw_result.find("DISCOVERED URLS"):]
            lines = urls_section.split('\n')
            for line in lines[1:6]:  # Get up to 5 URLs
                if line.strip() and line.startswith((' ', '\t')) and 'http' in line:
                    found_urls.append(line.strip())
        
        honest_response_parts = [
            f"I searched for information about '{original_question}' but need to be honest about what I actually found:"
        ]
        
        if found_urls:
            honest_response_parts.extend([
                "",
                "✅ What I successfully found:",
                f"- {len(found_urls)} relevant URLs that likely contain the information you need:",
            ])
            for i, url in enumerate(found_urls[:3], 1):
                honest_response_parts.append(f"  {i}. {url}")
            
            if len(found_urls) > 3:
                honest_response_parts.append(f"  ... and {len(found_urls) - 3} more URLs")
                
            honest_response_parts.extend([
                "",
                "❌ What I didn't do:",
                "- I didn't actually read the content of these web pages",
                "- I only have search result snippets, not the full information",
                "",
                "💡 To get the actual information you need:",
                "- I should read the content of these discovered URLs",
                "- This would give you accurate, verified information instead of guesses"
            ])
        else:
            honest_response_parts.extend([
                "",
                "❌ What happened:",
                "- My search didn't return clear, actionable information",
                "- I don't have access to verified details to answer your question",
                "",
                "💡 What you could try:",
                "- Try a more specific search query",
                "- Check the official documentation or repository directly"
            ])
        
        return "\n".join(honest_response_parts)

