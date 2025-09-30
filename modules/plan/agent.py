"""Simplified ReAct planning agent with goal focus."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm.llm import LLM
from ..tools.registry import ToolRegistry
from .models import Goal, Plan, PlanExecution, PlanStatus, ToolFailure


class PlanningAgent:
    """Simplified ReAct agent focused on goal achievement using tool manager."""

    def __init__(self, llm: LLM, tool_registry: ToolRegistry, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.tool_registry = tool_registry
        self.logger = logger or logging.getLogger("ymca.plan.agent")
        self.max_iterations = 10  # Prevent infinite loops
        self.current_plan: Optional[Plan] = None  # Track current plan for data flow

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Plan:
        """Execute a complete task using ReAct pattern."""
        self.logger.info(f"Starting ReAct execution: {task_description}")

        # Create goal from task description
        goal = self._create_goal(task_description)

        # Create plan
        plan = Plan(
            title=f"ReAct: {task_description[:50]}...",
            description=task_description,
            goal=goal,
            context=context or {},
        )

        # Execute plan
        await self.execute_plan(plan)
        return plan

    async def execute_plan(self, plan: Plan) -> None:
        """Execute an existing plan using ReAct pattern."""
        plan.status = PlanStatus.IN_PROGRESS
        plan.started_at = datetime.now()

        self.logger.info(f"Starting ReAct execution: {plan.title}")

        try:
            # ReAct loop: Reason -> Act -> Observe
            result = await self._react_loop(plan.description, plan.goal, plan.context, plan)

            plan.result = result
            plan.status = PlanStatus.COMPLETED
            if plan.goal:
                plan.goal.achieved = True

        except Exception as e:
            self.logger.error(f"ReAct execution failed: {e}")
            plan.error = str(e)
            plan.status = PlanStatus.FAILED

        plan.completed_at = datetime.now()

    async def _react_loop(self, task: str, goal: Optional[Goal], context: Dict[str, Any], plan: Plan) -> str:
        """Execute ReAct loop: Reason -> Act -> Observe."""
        observations: list[str] = []
        self.current_plan = plan  # Store reference for accumulated information

        for iteration in range(self.max_iterations):
            iteration_num = iteration + 1
            plan.current_iteration = iteration_num
            self.logger.info(f"🔄 ReAct iteration {iteration_num}/{self.max_iterations}")

            # REASON: Determine what to do next
            self.logger.info("🧠 REASON: Determining next action...")
            reasoning_prompt = self._build_reasoning_prompt(task, goal, observations, context, plan.evolving_answer, plan)
            reasoning_result, reasoning_response, debug_filename = await self._reason(task, goal, observations, context, plan.evolving_answer, plan)
            
            # Create execution record
            execution = PlanExecution(
                iteration=iteration_num,
                timestamp=datetime.now(),
                reasoning_prompt=reasoning_prompt,
                reasoning_response=reasoning_response,  # We'll set this later
                reasoning_result=reasoning_result,
                action=reasoning_result.get("action", ""),
                action_parameters=reasoning_result.get("parameters", {}),
                action_result={},  # We'll set this after action execution
                observation="",  # We'll set this after observation
                success=False,  # We'll update this based on action result
                error=None
            )
            
            # Add debug filename if available
            if debug_filename:
                execution.reasoning_result["debug_prompt_file"] = debug_filename

            if reasoning_result.get("complete", False):
                final_answer = str(reasoning_result.get("final_answer", "Task completed"))
                self.logger.info("✅ Task marked as COMPLETE by reasoning")
                self.logger.info(f"   Final answer length: {len(final_answer)} chars")
                
                # Update execution record
                execution.observation = f"Task completed with final answer ({len(final_answer)} chars)"
                execution.success = True
                plan.executions.append(execution)
                
                # Update evolving answer one final time
                await self._update_evolving_answer(plan, final_answer, is_final=True)
                return final_answer

            # ACT: Execute the chosen action
            self.logger.info("🎬 ACT: Executing chosen action...")
            action_result = await self._act(reasoning_result, plan)
            
            # Update execution record with action result
            execution.action_result = action_result

            # OBSERVE: Record the result and update evolving answer
            self.logger.info("👁️  OBSERVE: Creating observation from action result...")
            observation = await self._observe(reasoning_result, action_result, plan)
            observations.append(observation)
            execution.observation = observation
            execution.success = action_result.get("success", False)
            if not execution.success and action_result.get("error"):
                execution.error = str(action_result.get("error"))
            
            # Store the execution record
            plan.executions.append(execution)
            
            self.logger.info(f"   Total observations so far: {len(observations)}")

            # UPDATE: Build up the evolving answer with new information
            if action_result.get("success"):
                self.logger.info("📝 UPDATE: Adding successful result to evolving answer...")
                await self._update_evolving_answer(plan, action_result.get("result", ""))
            else:
                self.logger.info("⏭️  UPDATE: Skipping failed result, will retry in next iteration")

        # Max iterations reached
        return self._synthesize_final_answer(task, observations, plan.evolving_answer)

    def _build_reasoning_prompt(self, task: str, goal: Optional[Goal], observations: List[str], context: Dict[str, Any], current_answer: str = "", plan: Optional[Plan] = None) -> str:
        """Build the reasoning prompt for LLM with strategic guidance."""
        available_tools = self.tool_registry.get_tools_for_llm()
        available_tool_names = [tool['name'] for tool in available_tools]

        # Build reasoning prompt
        prompt_parts = [
            f"Task: {task}",
        ]

        if current_answer:
            prompt_parts.append(f"Current accumulated answer: {current_answer}")
            prompt_parts.append("(You can build upon this or refine it with new information)")
            
        # Add summary of key information gathered so far
        if plan and plan.knowledge_pieces:
            recent_findings = plan.knowledge_pieces[-2:]  # Last 2 findings to keep context manageable
            if recent_findings:
                findings_summary = self._summarize_recent_findings(recent_findings, task)
                if findings_summary:
                    prompt_parts.append(f"Key information gathered recently:\n{findings_summary}")
                    
        # Add context about visited URLs to avoid redundant research
        if plan and hasattr(plan, 'read_urls') and plan.read_urls:
            recent_urls = list(plan.read_urls)[-3:]  # Show last 3 visited URLs
            prompt_parts.append(f"Recently visited sources: {', '.join(recent_urls)}")
            prompt_parts.append("(Avoid re-visiting these unless necessary. Look for new sources if current information is insufficient.)")

        if goal:
            prompt_parts.append(f"Goal: {goal.description}")
            if goal.success_criteria:
                criteria_text = "\n".join([f"- {criterion}" for criterion in goal.success_criteria])
                prompt_parts.append(f"Success criteria:\n{criteria_text}")

        # Add strategic guidance based on tool failure patterns
        if plan:
            strategy_guidance = plan.get_strategy_guidance(available_tool_names)
            if strategy_guidance:
                prompt_parts.append(strategy_guidance)

            # Add information about tool retry limits
            failed_tools = plan.get_repeatedly_failing_tools()
            if failed_tools:
                retry_info = []
                for tool_name in failed_tools:
                    if not plan.should_retry_tool(tool_name):
                        retry_info.append(f"❌ {tool_name}: Max retries reached, avoid using")
                    else:
                        failure_count = plan.tool_failures[tool_name].failure_count
                        retry_info.append(f"⚠️  {tool_name}: {failure_count} failures, use with caution")
                
                if retry_info:
                    prompt_parts.append(f"Tool Status:\n" + "\n".join(retry_info))

        if observations:
            prompt_parts.append("Previous observations:")
            for i, obs in enumerate(observations[-3:], 1):  # Last 3 observations
                prompt_parts.append(f"{i}. {obs}")
                
            # Add analysis of what information might still be missing
            if plan and current_answer:
                missing_info_guidance = self._analyze_information_gaps(task, current_answer, observations)
                if missing_info_guidance:
                    prompt_parts.append(f"Information gap analysis: {missing_info_guidance}")

        # Add available tools with enhanced descriptions
        tool_descriptions = []
        for tool in available_tools:
            # Mark tools that have failed
            status_marker = ""
            if plan and tool['name'] in plan.get_repeatedly_failing_tools():
                if not plan.should_retry_tool(tool['name']):
                    status_marker = " [❌ MAX RETRIES REACHED]"
                else:
                    status_marker = " [⚠️  PREVIOUSLY FAILED]"
            
            tool_descriptions.append(f"- {tool['name']}: {tool['description']}{status_marker}")

        if tool_descriptions:
            prompt_parts.append("Available tools:")
            prompt_parts.extend(tool_descriptions)

        # Add strategic examples and guidance
        prompt_parts.append("""
🎯 STRATEGY GUIDANCE:
- If search tools fail repeatedly, try reading specific documentation URLs
- If you have gathered some information but tools are failing, consider using synthesize_information
- If multiple approaches fail, provide a helpful final answer explaining what you attempted
- Focus on making progress rather than retrying the same failing approach
- When in doubt, switch to a different tool or strategy

⚠️ CRITICAL SEARCH RESULT HANDLING:
- Search results (from search_web) are just the FIRST STEP, not the final answer
- After search_web succeeds, you MUST use read_webpage on discovered URLs to get actual information
- DO NOT complete tasks based only on search result snippets - they lack the detailed information users need
- Look for "DISCOVERED URLS" section in search results and follow up with read_webpage
- Only complete when you have ACTUAL CONTENT from webpages, not just search snippets

🔄 DATA INTEGRATION REQUIREMENTS:
- If your observations mention "NEXT STEP: Use read_webpage", you MUST do that before completing
- If you have DISCOVERED URLS but haven't read any pages, the task is NOT complete
- Prefer reading the most relevant URL (like GitHub repos for version info) before generic ones
- Look for specific information (version numbers, dates, etc.) not just general descriptions
- Only use synthesize_information AFTER you have read actual webpage content, not just search results

📋 SUCCESSFUL STRATEGY EXAMPLES:
- Search fails → Try known documentation URLs with read_webpage
- Multiple web tools fail → Use synthesize_information with accumulated knowledge
- Cannot find exact answer → Provide best available guidance with caveats
- search_web succeeds → ALWAYS follow up with read_webpage on the most relevant discovered URLs
- Found URLs in search → Use read_webpage to get actual content before completing
- Observation says "NEXT STEP" → Do exactly that next step before considering completion""")

        return self._add_json_requirements(prompt_parts)

    def _add_json_requirements(self, prompt_parts: List[str]) -> str:
        """Add JSON response requirements to prompt."""
        prompt_parts.append(
            "\nReason about the task and decide what to do next. "
            "If you have enough information to complete the task, set complete=true and provide final_answer. "
            "Otherwise, set complete=false and choose a tool to gather more information."
            "\n\n🚫 CRITICAL: NO DEBUG OUTPUT 🚫"
            "\nDO NOT start your response with phrases like:"
            "\n- '🔎 CRITICAL EVALUATION:'"  
            "\n- '**NEW INPUT:**'"
            "\n- '**Reasoning:**'"
            "\n- '**Action:**'"
            "\n- Any evaluation or meta-commentary"
            "\n\n⚠️ CRITICAL JSON RESPONSE REQUIREMENTS ⚠️"
            "\n1. Your response MUST be EXACTLY ONE JSON object"
            "\n2. Start response with { (no spaces, no text before)"
            "\n3. End response with } (no text after)"
            "\n4. NO explanations, examples, or additional text"
            "\n5. NO markdown code blocks (```)"
            "\n6. NO multiple JSON objects"
            "\n7. VALID JSON syntax only"
            "\n8. If tools fail, do NOT hallucinate - set action='' and explain in reasoning"
            "\n9. DO NOT complete tasks based only on search results - search is just the first step"
            "\n10. If search_web found URLs, use read_webpage before completing"
            "\n\nRequired JSON format:"
            "\n{"
            '\n  "reasoning": "Your reasoning about what to do next",'
            '\n  "complete": true,'
            '\n  "action": "tool_name_if_not_complete_or_empty_string_if_no_valid_tool",'
            '\n  "parameters": {"param": "value"},'
            '\n  "final_answer": "only_if_complete_is_true"'
            "\n}"
            "\n\nExample responses:"
            "\n- To continue working: {\"reasoning\": \"I need to search for information\", \"complete\": false, \"action\": \"search_web\", \"parameters\": {\"query\": \"kubectl-mtv\"}, \"final_answer\": \"\"}"
            "\n- To finish task: {\"reasoning\": \"I have all needed information\", \"complete\": true, \"action\": \"\", \"parameters\": {}, \"final_answer\": \"kubectl-mtv create migration-plan ...\"}"
            "\n\n🚫 NEVER HALLUCINATE INFORMATION:"
            "\n- NEVER make up command syntax or tool usage without verifying from actual sources"
            "\n- For CLI commands, you MUST verify syntax from official documentation or reliable sources"
            "\n- DO NOT guess command parameters or flags - search for accurate information first"
            "\n- If you haven't read actual webpage content, don't invent details"
            "\n- If search results show URLs but you haven't read them, follow up with read_webpage"
            "\n- If tools fail repeatedly, admit what you attempted rather than making up answers"
            "\n- Search result snippets are NOT sufficient for completing tasks"
            "\n\n⚠️ COMPLETION CRITERIA:"
            "\n- Do NOT set complete=true if your last observation says 'REQUIRED NEXT STEP' or 'CRITICAL: Task is NOT complete'"
            "\n- Do NOT complete based only on search results, even if they seem comprehensive"
            "\n- Only complete when you have read actual webpage content with specific information"
            "\n- If asked for version/specific info, you need actual content showing that info, not just links to it"
            "\n\n🔍 INFORMATION GAP DETECTION:"
            "\n- After reading a webpage, assess if it contains the SPECIFIC information requested"
            "\n- For CLI commands: Does the page show actual command syntax with real flags/parameters?"
            "\n- If the page lacks specific details, look for documentation links, examples, or tutorial links ON THAT PAGE"
            "\n- GitHub repositories often have documentation in docs/ folders or example files"
            "\n- Don't complete with generic/placeholder information - find actual verified syntax"
        )

        return "\n\n".join(prompt_parts)
    
    def _summarize_recent_findings(self, knowledge_pieces: List[str], task: str) -> Optional[str]:
        """Summarize recent findings using intelligent context compression."""
        try:
            if not knowledge_pieces:
                return None
                
            # Combine all knowledge pieces  
            findings_text = "\n---\n".join(knowledge_pieces)
            
            # Check if content is primarily search results - avoid expensive compression for structured data
            if self._is_search_results_content(findings_text):
                self.logger.info(f"📊 Detected search results content ({len(findings_text):,} chars) - using fast URL extraction instead of compression")
                return self._extract_search_results_summary(knowledge_pieces, task)
            
            # Use intelligent compression if content is large and not search results
            elif len(findings_text) > 1000:  # Use compression for substantial content
                compressed_summary, extracted_links = self._intelligent_context_compression(
                    findings_text, 
                    task, 
                    max_context_chars=800
                )
                self.logger.info(f"📊 Compressed recent findings from {len(findings_text):,} to {len(compressed_summary):,} chars")
                return compressed_summary
            else:
                # For small content, format as bullet points
                summary_lines = []
                for i, piece in enumerate(knowledge_pieces, 1):
                    # Extract key points from each piece 
                    preview = piece[:150].replace('\n', ' ').strip()
                    if len(piece) > 150:
                        preview += "..."
                    summary_lines.append(f"• Finding {i}: {preview}")
                
                return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.warning(f"Failed to summarize recent findings: {e}")
            return None
    
    def _analyze_information_gaps(self, task: str, current_answer: str, observations: List[str]) -> Optional[str]:
        """Analyze what information might still be missing for completing the task."""
        try:
            # Look for common patterns that suggest incomplete information
            incomplete_indicators = [
                "placeholder", "<", ">", "...", "TBD", "TODO", "example",
                "generic", "template", "replace", "substitute", "configure"
            ]
            
            # Check if current answer has incomplete indicators
            has_incomplete_info = any(indicator in current_answer.lower() for indicator in incomplete_indicators)
            
            # Check if recent observations suggest finding more specific information
            recent_obs = " ".join(observations[-2:]) if len(observations) >= 2 else ""
            found_relevant_links = "DISCOVERED RELEVANT LINKS" in recent_obs or "AUTOMATICALLY VISITED" in recent_obs
            
            guidance_parts = []
            
            if has_incomplete_info:
                guidance_parts.append("Current answer contains placeholders/generic information - look for more specific details")
                
            if found_relevant_links:
                guidance_parts.append("Relevant links were found and visited - assess if additional sources are needed")
            
            # For CLI command tasks, check for specific completeness
            if any(cmd_word in task.lower() for cmd_word in ["command", "cli", "kubectl", "syntax"]):
                if not any(flag in current_answer for flag in ["--", "-", "kubectl"]):
                    guidance_parts.append("Task involves CLI commands - ensure actual command syntax with flags is provided")
                    
            if guidance_parts:
                return "; ".join(guidance_parts)
                
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze information gaps: {e}")
            return None

    async def _reason(self, task: str, goal: Optional[Goal], observations: List[str], context: Dict[str, Any], current_answer: str = "", plan: Optional[Plan] = None) -> tuple[Dict[str, Any], str, Optional[str]]:
        """Reason about what to do next. Returns (reasoning_result, raw_response, debug_filename)."""
        prompt = self._build_reasoning_prompt(task, goal, observations, context, current_answer, plan)

        try:
            self.logger.info("🧠 Sending reasoning prompt to LLM")
            
            # Generate reasoning context for debug filename
            iteration = plan.current_iteration if plan else 1
            context_desc = f"reasoning_iter{iteration}_{task[:30].replace(' ', '_')}"
            
            # Use debug-capable LLM method
            response, debug_filename = self.llm.generate_response_with_debug(prompt, context_desc)
            
            self.logger.info(f"🧠 LLM response received ({len(response)} chars)")
            if debug_filename:
                self.logger.info(f"🐛 Prompt saved to debug file: {debug_filename}")
            self.logger.debug(f"   Full LLM response: {response}")

            reasoning_result = await self._parse_reasoning_response(response, task, observations)
            self.logger.info(f"🧠 Reasoning result: action='{reasoning_result.get('action', 'None')}', complete={reasoning_result.get('complete', False)}")

            return reasoning_result, response, debug_filename
        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}")
            # Don't assume completion on reasoning failure - let the system retry
            fallback_reasoning = {
                "reasoning": f"Reasoning process failed with error: {e}. Need to retry or use different approach.",
                "complete": False,
                "action": "",
                "parameters": {}
            }
            return fallback_reasoning, f"Error: {e}", None

    async def _act(self, reasoning_result: Dict[str, Any], plan: Optional[Plan] = None) -> Dict[str, Any]:
        """Execute the chosen action with failure tracking."""
        action = reasoning_result.get("action", "").strip()
        parameters = reasoning_result.get("parameters", {})

        if not action:
            self.logger.warning("⚠️  No action specified in reasoning result")
            return {
                "success": False, 
                "error": "No action specified", 
                "guidance": "This indicates the LLM could not determine a valid next step. Consider providing a final answer based on available information."
            }

        # Check if this tool should be retried based on failure history
        if plan and not plan.should_retry_tool(action):
            self.logger.warning(f"🚫 Tool '{action}' has exceeded retry limit, blocking execution")
            return {
                "success": False,
                "error": f"Tool '{action}' has failed too many times and is temporarily blocked",
                "guidance": f"Try a different approach. Tool failure count: {plan.tool_failures.get(action, ToolFailure(action)).failure_count}"
            }

        self.logger.info(f"🎯 Planning to execute action: {action}")

        # Check for duplicate URL reads for web-based tools
        if action == "read_webpage" and plan and parameters.get("url"):
            url = parameters["url"]
            if plan.is_url_already_read(url):
                self.logger.warning(f"🔄 URL already read, skipping: {url}")
                return {
                    "success": False,
                    "error": f"URL already read: {url}",
                    "guidance": "This URL has already been read. Try a different URL or use synthesize_information to combine existing knowledge."
                }
            # Mark URL as read before execution
            plan.mark_url_as_read(url)
            self.logger.info(f"📝 Marked URL as read: {url}")

        try:
            # For synthesis tool, ensure all required parameters are provided
            if action == "synthesize_information":
                # Always ensure original_question is provided (required parameter)
                if "original_question" not in parameters:
                    if plan and plan.description:
                        parameters["original_question"] = plan.description
                        self.logger.info(f"🔄 Added original_question: {plan.description}")
                    else:
                        parameters["original_question"] = "information synthesis"
                        self.logger.warning(f"🔄 Using fallback original_question")
                
                # Add accumulated information if not provided
                if "information_sources" not in parameters:
                    accumulated = self._get_accumulated_information()
                    parameters["information_sources"] = accumulated
                    self.logger.info(f"🔄 Added {len(accumulated)} chars of accumulated information to synthesis")

            tool_call = {"name": action, "parameters": parameters, "call_id": f"{action}_{datetime.now().timestamp()}"}

            self.logger.info(f"📞 Calling tool registry with: {tool_call['name']}")
            self.logger.debug(f"   Complete tool call: {tool_call}")

            result = await self.tool_registry.execute_tool(tool_call)

            self.logger.info(f"📋 Tool execution result: success={result.get('success', False)}")
            
            # Record tool failure for strategic planning
            if not result.get("success") and plan:
                error_msg = result.get("error", "Unknown error")
                plan.record_tool_failure(action, error_msg)
                self.logger.info(f"📊 Recorded failure for tool '{action}': {plan.tool_failures[action].failure_count} total failures")
            
            if result.get("success"):
                result_length = len(str(result.get("result", "")))
                self.logger.info(f"   Result length: {result_length} characters")
                self.logger.debug(f"   Full result: {result.get('result', '')}")
            else:
                self.logger.warning(f"   Error: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Action execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _observe(self, reasoning_result: Dict[str, Any], action_result: Dict[str, Any], plan: Optional[Plan] = None) -> str:
        """Create observation from action result."""
        try:
            action = reasoning_result.get("action", "unknown")

            if action_result.get("success"):
                result_text = str(action_result.get("result", ""))
                execution_time = action_result.get("execution_time", 0)

                # Special handling for search results with DISCOVERED URLS - automatically visit them  
                if action == "search_web" and "DISCOVERED URLS" in result_text:
                    urls_section = result_text[result_text.find("DISCOVERED URLS"):]
                    urls_count = result_text.count("http")
                    
                    # Extract the most relevant URL (usually the first one for the specific query)
                    urls_lines = urls_section.split('\n')[1:4]  # Get first 3 URLs
                    relevant_urls = [line.strip() for line in urls_lines if line.strip() and 'http' in line]
                    
                    if len(result_text) > 500:
                        result_text = result_text[:500] + "... (truncated)"
                    
                    if relevant_urls:
                        top_url = relevant_urls[0].split('. ', 1)[-1] if '. ' in relevant_urls[0] else relevant_urls[0]
                        observation = (
                            f"Action '{action}' succeeded in {execution_time:.2f}s and found {urls_count} URLs. "
                            f"⚠️ CRITICAL: Task is NOT complete - you have only search snippets, not actual content. "
                            f"REQUIRED NEXT STEP: Use read_webpage on '{top_url}' to get the actual information needed. "
                            f"Search results: {result_text}"
                        )
                    else:
                        observation = (
                            f"Action '{action}' succeeded in {execution_time:.2f}s and found {urls_count} URLs. "
                            f"NEXT STEP: Use read_webpage on the most relevant discovered URLs to get actual content. "
                            f"Search results: {result_text}"
                        )
                    
                    self.logger.info(f"👁️  Search observation with critical follow-up guidance created")
                    return observation
                
                # Special handling for webpage results - use LLM to extract relevant links from content
                elif action == "read_webpage" and "Content (" in result_text and "characters):" in result_text:
                    try:
                        # Check if we have a plan to get the original query context
                        original_query = plan.description if plan else reasoning_result.get("task", "")
                        
                        self.logger.info(f"🔗 Webpage found links, checking for relevant ones for query: {original_query}")
                        
                        # Extract links from the result
                        relevant_links = await self._extract_relevant_links_from_webpage_result(result_text, original_query, plan)
                        
                        if relevant_links:
                            # Proactively visit relevant links to gather more information
                            additional_info = await self._visit_relevant_links(relevant_links, original_query, plan)
                            
                            if additional_info:
                                # Combine original result with additional information
                                if len(result_text) > 400:
                                    result_text = result_text[:400] + "... (truncated)"
                                
                                combined_result = f"{result_text}\n\n🔗 AUTOMATICALLY VISITED RELEVANT LINKS:\n{additional_info}"
                                observation = f"Action '{action}' succeeded in {execution_time:.2f}s: {combined_result}"
                                self.logger.info(f"👁️  Webpage observation with additional information from {len(relevant_links)} relevant links")
                                return observation
                            else:
                                # Links were visited but didn't provide useful additional information
                                links_text = ", ".join(relevant_links[:3])
                                additional_note = f"\n🔗 Note: Visited {len(relevant_links)} relevant links ({links_text}) but they didn't contain additional useful information for this query."
                                
                                if len(result_text) > 400:
                                    result_text = result_text[:400] + "... (truncated)"
                                
                                observation = f"Action '{action}' succeeded in {execution_time:.2f}s: {result_text}{additional_note}"
                                self.logger.info(f"👁️  Webpage observation with attempted link following")
                                return observation
                        else:
                            self.logger.info(f"🔗 No relevant links found for query: {original_query}")
                    
                    except Exception as e:
                        self.logger.error(f"❌ Error in link following logic: {e}")
                        # Continue with standard observation if link following fails
                
                # Standard observation processing
                if len(result_text) > 500:
                    result_text = result_text[:500] + "..."

                observation = f"Action '{action}' succeeded in {execution_time:.2f}s: {result_text}"
                self.logger.info(f"👁️  Observation created: {observation[:100]}{'...' if len(observation) > 100 else ''}")
                return observation
            else:
                error = action_result.get("error", "Unknown error")
                execution_time = action_result.get("execution_time", 0)

                observation = f"Action '{action}' failed after {execution_time:.2f}s: {error}"
                self.logger.warning(f"👁️  Observation (failure): {observation}")
                return observation
        
        except Exception as e:
            self.logger.error(f"❌ Critical error in _observe method: {e}")
            # Fallback observation to prevent null observations
            action = reasoning_result.get("action", "unknown")
            if action_result.get("success"):
                return f"Action '{action}' completed but observation processing failed: {e}"
            else:
                return f"Action '{action}' failed: {action_result.get('error', 'Unknown error')}"

    async def _extract_relevant_links_from_webpage_result(self, webpage_result: str, original_query: str, plan: Optional[Plan] = None) -> List[str]:
        """Extract relevant links from webpage content using intelligent LLM analysis with batching."""
        try:
            # Extract current URL and content from webpage result
            current_url, webpage_content = self._parse_webpage_result(webpage_result)
            
            if not webpage_content:
                self.logger.warning("🔗 No content found in webpage result for link extraction")
                return []
            
            # Use intelligent context compression for large content before link extraction
            if len(webpage_content) > 8000:  # Use compression for large content
                compressed_content, compression_links = self._intelligent_context_compression(webpage_content, original_query, max_context_chars=6000)
                self.logger.info(f"📊 Using compressed content for link extraction")
                # Extract additional links from compressed content and combine with compression links
                batch_extracted_links = await self._extract_links_in_batches(compressed_content, original_query, current_url)
                # Combine links from compression phase with links from batch extraction
                all_links = compression_links + batch_extracted_links
                extracted_links = list(dict.fromkeys(all_links))  # Remove duplicates while preserving order
                if compression_links:
                    self.logger.info(f"🔗 Combined {len(compression_links)} compression links + {len(batch_extracted_links)} batch links = {len(extracted_links)} total links")
            else:
                # Use original batching method for smaller content
                extracted_links = await self._extract_links_in_batches(webpage_content, original_query, current_url)
            
            if not extracted_links:
                self.logger.info("🔗 No valid URLs found in LLM analysis")
                return []
            
            # Check link following limits before processing more links
            if not self._can_follow_more_links(plan, original_query):
                return []
            
            # Check if we already have sufficient information before following more links
            if await self._has_sufficient_information_for_query(original_query, plan):
                self.logger.info("✅ Query appears to be sufficiently answered - skipping additional link following")
                return []
            
            # Filter out already visited URLs
            filtered_links = self._filter_unvisited_links(extracted_links, current_url, plan)
            
            # Log results
            filtered_count = len(extracted_links) - len(filtered_links)
            if filtered_count > 0:
                self.logger.info(f"🚫 Filtered out {filtered_count} already-visited URLs")
            
            if filtered_links:
                # Use LLM-based content prioritization for best link selection
                prioritized_links = await self._prioritize_links_by_content(filtered_links, original_query, plan)
                
                if prioritized_links:
                    selected_links = prioritized_links[:2]  # Take top 2 after content analysis
                    self.logger.info(f"🎯 Content-prioritized selection: {len(selected_links)} links:")
                    for i, link in enumerate(selected_links, 1):
                        self.logger.info(f"   {i}. {link}")
                    return selected_links
                else:
                    # Fallback to first 2 if content prioritization fails
                    fallback_links = filtered_links[:2]
                    self.logger.warning(f"⚠️  Content prioritization failed, using fallback: {len(fallback_links)} links")
                    return fallback_links
            else:
                self.logger.warning(f"⚠️  All {len(extracted_links)} LLM-extracted links were already visited")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract relevant links: {e}")
            return []
    
    async def _prioritize_links_by_content(self, links: List[str], original_query: str, plan: Optional[Plan]) -> List[str]:
        """Prioritize links by actually reading their content and using LLM to evaluate relevance."""
        if len(links) <= 2:
            return links  # No need to prioritize if we have 2 or fewer
        
        self.logger.info(f"📚 Starting content-based prioritization of {len(links)} links for query: {original_query}")
        
        link_summaries = []
        
        # Read and summarize each link
        for i, link in enumerate(links[:6], 1):  # Limit to first 6 to avoid too many requests
            try:
                self.logger.info(f"🔍 Reading link {i}/{min(len(links), 6)}: {link}")
                
                # Read the webpage content
                tool_call = {
                    "name": "read_webpage",
                    "parameters": {"url": link, "extract_links": False},
                    "call_id": f"prioritize_{i}_{int(time.time())}"
                }
                
                result = await self.tool_registry.execute_tool(tool_call)
                
                if result.get("success"):
                    webpage_content = str(result.get("result", ""))
                    
                    # Mark URL as visited
                    if plan:
                        plan.mark_url_as_read(link)
                    
                    # Get content-based summary for this link
                    summary = await self._summarize_link_content(webpage_content, original_query, link)
                    
                    if summary:
                        link_summaries.append({
                            "url": link,
                            "summary": summary,
                            "relevance_score": 0  # Will be set by LLM
                        })
                        self.logger.info(f"✅ Summarized {link}: {summary[:100]}...")
                    else:
                        self.logger.warning(f"⚠️  No relevant content found in {link}")
                else:
                    self.logger.warning(f"❌ Failed to read {link}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error reading link {link}: {e}")
                continue
        
        if not link_summaries:
            self.logger.warning("⚠️  No links could be read for content prioritization")
            return links
        
        # Use LLM to rank links by relevance based on their actual content
        prioritized_links = await self._rank_links_by_content(link_summaries, original_query)
        
        self.logger.info(f"🏆 Content prioritization complete: selected {len(prioritized_links)} best links")
        return prioritized_links
    
    async def _summarize_link_content(self, webpage_content: str, original_query: str, link_url: str) -> Optional[str]:
        """Summarize webpage content in context of the original query."""
        try:
            # Extract main content from webpage result
            _, content = self._parse_webpage_result(webpage_content)
            
            if not content:
                return None
            
            # Limit content for LLM processing
            content_preview = content[:3000] if len(content) > 3000 else content
            
            summary_prompt = f"""Summarize how this webpage content helps answer: "{original_query}"

Webpage: {link_url}
Content preview:
{content_preview}

Provide a 1-2 sentence summary focusing on:
- What specific information/commands/examples it contains
- How directly it answers the user's query
- If it's not relevant, say "Not relevant to query"

Summary:"""
            
            response = self.llm.generate_response(summary_prompt)
            
            if "not relevant" in response.lower():
                return None
                
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to summarize content for {link_url}: {e}")
            return None
    
    async def _rank_links_by_content(self, link_summaries: List[dict], original_query: str) -> List[str]:
        """Use LLM to rank links by relevance based on their actual content summaries."""
        try:
            summaries_text = "\n\n".join([
                f"{i+1}. {item['url']}\n   Content: {item['summary']}"
                for i, item in enumerate(link_summaries)
            ])
            
            ranking_prompt = f"""Rank these links by how well they help answer: "{original_query}"

Links with content summaries:
{summaries_text}

Rank them from most to least helpful (1 = most helpful).
Consider:
- Direct relevance to the query
- Practical value (commands, examples, tutorials)
- Completeness of information

Respond with only the numbers in order of preference (e.g., "3 1 2"):"""
            
            response = self.llm.generate_response(ranking_prompt)
            
            # Parse ranking response
            ranked_indices = []
            for token in response.strip().split():
                if token.isdigit():
                    idx = int(token) - 1  # Convert to 0-based
                    if 0 <= idx < len(link_summaries):
                        ranked_indices.append(idx)
            
            # Return links in ranked order
            if ranked_indices:
                ranked_links = [link_summaries[i]['url'] for i in ranked_indices]
                self.logger.info(f"📊 Ranked links: {ranked_links}")
                return ranked_links
            else:
                # Fallback to original order
                return [item['url'] for item in link_summaries]
                
        except Exception as e:
            self.logger.error(f"❌ Failed to rank links by content: {e}")
            return [item['url'] for item in link_summaries]
    
    def _parse_webpage_result(self, webpage_result: str) -> tuple[Optional[str], Optional[str]]:
        """Parse webpage result to extract current URL and full content."""
        current_url = None
        webpage_content = None
        
        lines = webpage_result.split('\n')
        
        # Extract current URL
        for line in lines:
            if line.startswith('URL: ') and 'http' in line:
                current_url = line.replace('URL: ', '').strip()
                break
        
        # Extract content (skip metadata headers) - FIXED to get ALL content
        content_started = False
        content_lines = []
        for line in lines:
            if content_started:
                content_lines.append(line)
            elif line.startswith('Content (') and 'characters):' in line:
                content_started = True
                continue
        
        if content_lines:
            webpage_content = '\n'.join(content_lines)
            self.logger.info(f"📊 Extracted {len(webpage_content):,} characters from webpage result")
            
            # Debug: Check if we're getting the documentation section
            if 'README-usage.md' in webpage_content or 'README_demo.md' in webpage_content:
                self.logger.info("✅ Found documentation links in extracted content")
            else:
                self.logger.warning("⚠️  Documentation section may be missing from extracted content")
                # Show a sample to debug
                sample = webpage_content[-500:] if len(webpage_content) > 500 else webpage_content
                self.logger.warning(f"📝 Content end sample: ...{sample}")
        
        return current_url, webpage_content

    def _intelligent_context_compression(self, large_content: str, original_query: str, max_context_chars: int = 4000) -> tuple[str, List[str]]:
        """
        Use LLM-based chunking and summarization to compress large content intelligently.
        Each chunk is summarized in context of the original query, and irrelevant summaries are filtered out.
        Additionally extracts relevant links found during the summarization process.
        
        Args:
            large_content: The large text content to compress
            original_query: The original user query for context-aware summarization  
            max_context_chars: Maximum characters for the final compressed context
            
        Returns:
            Tuple of (compressed context, extracted links list)
        """
        if len(large_content) <= max_context_chars:
            return large_content, []
            
        self.logger.info(f"🧠 Compressing {len(large_content):,} chars using intelligent LLM summarization")
        
        # Step 1: Chunk the content intelligently
        chunks = self._create_intelligent_chunks(large_content, chunk_size=2000, overlap=200)
        self.logger.info(f"📄 Created {len(chunks)} intelligent chunks")
        
        # Step 2: Summarize each chunk and extract links in context of original query
        summaries = []
        extracted_links = []
        processed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            processed_chunks += 1
            summary, chunk_links = self._summarize_and_extract_links_from_chunk(chunk, original_query, chunk_index=i+1)
            if summary and summary.strip():
                summaries.append(summary)
            if chunk_links:
                extracted_links.extend(chunk_links)
            
            # Early termination check: After processing several chunks, check if we have enough information
            if processed_chunks >= 3 and processed_chunks % 2 == 1:  # Check every 2 chunks after the first 3
                should_continue = self._should_continue_summarizing(summaries, original_query, total_chunks=len(chunks), processed=processed_chunks)
                if not should_continue:
                    remaining_chunks = len(chunks) - processed_chunks
                    self.logger.info(f"🎯 Early termination: Found sufficient information after {processed_chunks}/{len(chunks)} chunks (skipping {remaining_chunks} remaining chunks)")
                    break
                
        self.logger.info(f"📝 Generated {len(summaries)} chunk summaries from {processed_chunks}/{len(chunks)} processed chunks")
        if extracted_links:
            unique_links = list(dict.fromkeys(extracted_links))  # Remove duplicates while preserving order
            self.logger.info(f"🔗 Simultaneously extracted {len(unique_links)} links during compression")
        
        # Step 3: Filter out irrelevant summaries using LLM  
        relevant_summaries = self._filter_relevant_summaries(summaries, original_query)
        self.logger.info(f"🎯 Filtered to {len(relevant_summaries)} relevant summaries")
        
        # Step 4: Concatenate relevant summaries into compact context
        compressed_context = self._concatenate_summaries(relevant_summaries, max_context_chars)
        self.logger.info(f"✅ Compressed context: {len(compressed_context):,} chars (reduction: {100*(1-len(compressed_context)/len(large_content)):.1f}%)")
        
        # Return both compressed context and extracted links
        unique_links = list(dict.fromkeys(extracted_links)) if extracted_links else []
        return compressed_context, unique_links
    
    def _create_intelligent_chunks(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Create intelligent chunks that respect sentence/paragraph boundaries."""
        chunks = []
        content_length = len(content)
        start = 0
        
        while start < content_length:
            end = start + chunk_size
            
            if end >= content_length:
                # Last chunk
                chunks.append(content[start:])
                break
                
            # Try to end at a natural boundary (paragraph, sentence, or word)
            chunk_text = content[start:end]
            
            # Look for paragraph boundary within overlap distance
            paragraph_boundary = chunk_text.rfind('\n\n')
            if paragraph_boundary > len(chunk_text) - overlap:
                end = start + paragraph_boundary + 2
            else:
                # Look for sentence boundary
                sentence_boundary = max(chunk_text.rfind('. '), chunk_text.rfind('.\n'))
                if sentence_boundary > len(chunk_text) - overlap:
                    end = start + sentence_boundary + 1
                else:
                    # Look for word boundary  
                    word_boundary = chunk_text.rfind(' ')
                    if word_boundary > len(chunk_text) - overlap:
                        end = start + word_boundary
            
            chunks.append(content[start:end])
            start = max(end - overlap, start + 1)  # Ensure progress
            
        return chunks
    
    def _summarize_chunk_for_query(self, chunk: str, original_query: str, chunk_index: int) -> Optional[str]:
        """Summarize a chunk in context of the original query."""
        summary, _ = self._summarize_and_extract_links_from_chunk(chunk, original_query, chunk_index)
        return summary
    
    def _summarize_and_extract_links_from_chunk(self, chunk: str, original_query: str, chunk_index: int) -> tuple[Optional[str], List[str]]:
        """Summarize a chunk and simultaneously extract relevant links in context of the original query."""
        prompt = f"""Analyze this content chunk in context of the user's query and provide both a summary and extract relevant URLs.

USER'S ORIGINAL QUERY: {original_query}

CONTENT CHUNK #{chunk_index}:
{chunk}

TASK: Provide both a summary AND extract relevant URLs that could help answer the query.

PART 1 - SUMMARY:
Create a concise summary (2-4 sentences) that extracts ONLY information relevant to answering the user's query.
If this chunk contains NO relevant information, respond with "IRRELEVANT"
Keep technical details and exact syntax when relevant.

PART 2 - LINKS:
EXTRACT ONLY URLs that are literally present in the content chunk above. 
CRITICAL: DO NOT generate, create, invent, or make up any URLs.
ONLY extract URLs that are actually written in the content chunk.

Look for:
- Complete URLs starting with http:// or https://
- Relative URLs starting with / 
- Focus on documentation, examples, guides, tutorials, installation instructions

Format your response as:
SUMMARY: [your summary here]
LINKS: [list URLs one per line that are literally present in the content, or "NONE" if no URLs found]"""

        try:
            response = self.llm.generate_response(prompt).strip()
            
            # Parse the response
            summary = None
            links = []
            
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    current_section = 'summary'
                    summary_text = line[8:].strip()  # Remove "SUMMARY:" prefix
                    if summary_text and summary_text.upper() != "IRRELEVANT":
                        summary = summary_text
                elif line.startswith('LINKS:'):
                    current_section = 'links'
                    links_text = line[6:].strip()  # Remove "LINKS:" prefix
                    if links_text and links_text.upper() != "NONE":
                        # This might be the first link on the same line
                        if links_text.startswith('http'):
                            links.append(links_text)
                elif current_section == 'summary' and line and not line.startswith('LINKS:'):
                    # Continue summary on multiple lines
                    if summary:
                        summary += " " + line
                    elif line.upper() != "IRRELEVANT":
                        summary = line
                elif current_section == 'links' and line and line.upper() != "NONE":
                    # Additional links on separate lines
                    if line.startswith('http') or line.startswith('/'):
                        links.append(line)
            
            # Final validation
            if summary and (summary.upper() == "IRRELEVANT" or len(summary) < 10):
                summary = None
                
            # Clean up links
            valid_links = []
            for link in links:
                link = link.strip().rstrip('.,;')
                if link and ('http' in link or link.startswith('/')):
                    valid_links.append(link)
            
            return summary, valid_links
            
        except Exception as e:
            self.logger.error(f"❌ Failed to summarize and extract links from chunk {chunk_index}: {e}")
            return None, []
    
    def _filter_relevant_summaries(self, summaries: List[str], original_query: str) -> List[str]:
        """Use LLM to filter out summaries that don't help answer the original query."""
        if len(summaries) <= 3:
            return summaries  # Keep all if we only have a few
            
        summaries_text = "\n\n".join([f"SUMMARY {i+1}: {summary}" for i, summary in enumerate(summaries)])
        
        prompt = f"""Filter these summaries to keep only those that help answer the user's query.

USER'S ORIGINAL QUERY: {original_query}

SUMMARIES:
{summaries_text}

TASK: List the numbers of summaries that contain information relevant to answering the user's query.

RULES:
1. Keep summaries with specific commands, examples, or instructions
2. Keep summaries with technical details needed to answer the query  
3. Drop summaries about unrelated topics, general information, or background
4. If multiple summaries cover the same topic, keep the most specific/detailed one

Respond with just the numbers (e.g., "1, 3, 5, 7") or "NONE" if no summaries are relevant:"""

        try:
            response = self.llm.generate_response(prompt).strip()
            
            if response.upper() == "NONE":
                return []
                
            # Parse the response to get summary indices
            relevant_indices = []
            for part in response.split(','):
                try:
                    index = int(part.strip()) - 1  # Convert to 0-based
                    if 0 <= index < len(summaries):
                        relevant_indices.append(index)
                except ValueError:
                    continue
                    
            return [summaries[i] for i in relevant_indices]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to filter summaries: {e}")
            return summaries[:3]  # Fallback: keep first 3
    
    def _concatenate_summaries(self, summaries: List[str], max_chars: int) -> str:
        """Concatenate summaries into a coherent context, staying within character limit."""
        if not summaries:
            return ""
            
        # Add headers and concatenate
        context_parts = []
        context_parts.append("=== RELEVANT INFORMATION ===")
        
        current_length = len(context_parts[0]) + 10  # Buffer for separators
        
        for i, summary in enumerate(summaries):
            section = f"\n{i+1}. {summary}"
            if current_length + len(section) > max_chars:
                context_parts.append(f"\n\n... ({len(summaries) - i} additional summaries truncated to stay within context limits)")
                break
            context_parts.append(section)
            current_length += len(section)
        
        return "".join(context_parts)

    async def _extract_links_in_batches(self, webpage_content: str, original_query: str, current_url: Optional[str]) -> List[str]:
        """Process webpage content in batches using LLM to extract relevant links."""
        BATCH_SIZE = 8000  # Characters per batch 
        OVERLAP = 1000     # Overlap between batches to avoid missing boundary links
        
        content_length = len(webpage_content)
        self.logger.info(f"📝 Processing {content_length:,} characters in batches of {BATCH_SIZE:,} chars")
        
        all_links = []
        batch_count = 0
        start = 0
        
        while start < content_length:
            batch_count += 1
            end = min(start + BATCH_SIZE, content_length)
            batch_content = webpage_content[start:end]
            
            self.logger.debug(f"🔄 Batch {batch_count}: chars {start:,}-{end:,}")
            
            # LLM extraction for this batch
            batch_links = await self._extract_links_from_batch(batch_content, original_query, batch_count)
            
            if batch_links:
                all_links.extend(batch_links)
                self.logger.info(f"✅ Batch {batch_count} found {len(batch_links)} links")
            
            # Move to next batch with overlap
            if end >= content_length:
                break
            start = max(start + BATCH_SIZE - OVERLAP, start + 1)
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(all_links))  # Preserves order, removes duplicates
        
        if unique_links:
            self.logger.info(f"🎯 Found {len(unique_links)} unique links from {batch_count} batches")
        
        return unique_links
    
    async def _extract_links_from_batch(self, batch_content: str, original_query: str, batch_num: int) -> List[str]:
        """Extract links from a single batch using LLM analysis."""
        prompt = f"""EXTRACT ONLY the URLs that are literally present in this content that could help answer: "{original_query}"

Content:
{batch_content}

CRITICAL INSTRUCTIONS:
- ONLY extract URLs that are actually written/present in the above content
- DO NOT generate, create, invent, or make up any URLs
- DO NOT suggest what URLs should exist - only extract what actually exists
- Look for complete URLs starting with http:// or https://
- Also look for relative URLs starting with / that could be relevant
- Focus on documentation, examples, guides, tutorials, installation instructions
- If no actual URLs are found in the content above, respond: NONE

EXAMPLE OF WHAT TO DO:
- If you see "Visit https://example.com/docs" → extract: https://example.com/docs
- If you see "See /examples/config.yaml" → extract: /examples/config.yaml

EXAMPLE OF WHAT NOT TO DO:
- Do NOT create URLs like https://github.com/user/repo/docs/ unless this exact URL appears in the content
- Do NOT invent paths like /docs/commands.md unless this exact path is written in the content

Return only URLs that are literally present in the content above, one per line, no explanations:"""

        try:
            response = self.llm.generate_response(prompt)
            
            if response.strip().upper() == "NONE":
                return []
            
            return self._parse_llm_links_response(response, None)  # Parse without current_url since it's batch-specific
            
        except Exception as e:
            self.logger.warning(f"❌ Batch {batch_num} extraction failed: {e}")
            return []

    def _parse_llm_links_response(self, response: str, current_url: Optional[str]) -> List[str]:
        """Parse LLM response to extract clean URLs, handling various formats."""
        import re
        from urllib.parse import urljoin
        
        extracted_links = []
        self.logger.info(f"🔗 Raw LLM response for link extraction:\n{response}")
        
        # Handle different response formats that LLM might return
        lines = response.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and "NONE" responses
            if not line or line.upper() == "NONE" or "explanation:" in line.lower():
                continue
                
            found_urls = []
            
            # Method 1: Extract markdown links [text](url)
            markdown_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            markdown_matches = re.findall(markdown_pattern, line)
            for text, url in markdown_matches:
                found_urls.append(url.strip())
                self.logger.info(f"🔗 Found markdown link: [{text}]({url})")
            
            # Method 2: Extract plain HTTP(S) URLs
            url_pattern = r'https?://[^\s<>"\'\']+'
            plain_urls = re.findall(url_pattern, line)
            for url in plain_urls:
                # Clean up trailing punctuation
                clean_url = url.rstrip('.,;"\'\')}])')
                found_urls.append(clean_url)
                self.logger.info(f"🔗 Found plain URL: {clean_url}")
            
            # Method 3: Handle relative URLs (starting with /)
            if line.startswith('/') and current_url:
                relative_url = line.split()[0].rstrip('.,;"\'\')}])')
                absolute_url = urljoin(current_url, relative_url)
                found_urls.append(absolute_url)
                self.logger.info(f"🔗 Found relative URL: {relative_url} → {absolute_url}")
            
            # Add valid URLs to extracted_links
            for url in found_urls:
                url = url.strip()
                if url and ('http' in url or url.startswith('/')):
                    # Make relative URLs absolute
                    if url.startswith('/') and current_url:
                        url = urljoin(current_url, url)
                    
                    # Final validation
                    if url.startswith('http') and '.' in url:
                        extracted_links.append(url)
                        self.logger.info(f"✅ Added valid URL: {url}")
                    else:
                        self.logger.warning(f"❌ Rejected invalid URL: {url}")
        
        self.logger.info(f"🎯 LLM link extraction summary: {len(extracted_links)} valid URLs found")
        for i, url in enumerate(extracted_links, 1):
            self.logger.info(f"   {i}. {url}")
            
        return extracted_links

    def _filter_unvisited_links(self, links: List[str], current_url: Optional[str], plan: Optional[Plan] = None) -> List[str]:
        """Filter links to remove already visited pages (ignoring query params and hash fragments)."""
        from urllib.parse import urlparse, urlunparse
        
        filtered_links = []
        
        # Get already visited URLs from the plan and normalize them
        visited_normalized_urls = set()
        if plan and hasattr(plan, 'read_urls'):
            for visited_url in plan.read_urls:
                normalized = self._normalize_url(visited_url)
                if normalized:
                    visited_normalized_urls.add(normalized)
            if visited_normalized_urls:
                self.logger.info(f"🔗 Plan tracking {len(visited_normalized_urls)} previously visited URLs")
        
        for link in links:
            try:
                # Normalize the link to compare with visited URLs (remove query params and fragments)
                normalized_link = self._normalize_url(link)
                if not normalized_link:
                    self.logger.debug(f"🔗 Skipping invalid URL: {link}")
                    continue
                
                # Skip already visited pages (same page, ignoring query/hash)
                if normalized_link in visited_normalized_urls:
                    self.logger.info(f"🔗 Skipping already visited: {link}")
                    continue
                
                # Parse the link for additional filtering
                parsed_link = urlparse(link)
                
                # Skip non-HTTP(S) links
                if parsed_link.scheme not in ['http', 'https']:
                    self.logger.debug(f"🔗 Skipping non-HTTP link: {link}")
                    continue
                
                # Skip common navigation/UI links (be more specific to avoid filtering docs)
                navigation_patterns = [
                    'login', 'signup', 'register', 'logout', '/login', '/signup', '/register',
                    '#start-of-content', 'javascript:', 'mailto:', 'tel:',
                    '/settings', '/profile', '/account',
                    # Remove 'features' - it's too broad and catches legitimate feature docs
                ]
                if any(skip_pattern in link.lower() for skip_pattern in navigation_patterns):
                    self.logger.debug(f"🔗 Skipping navigation/UI link: {link}")
                    continue
                
                # This link passes all filters
                filtered_links.append(link)
                self.logger.debug(f"🔗 ✅ Keeping unvisited link: {link}")
                
            except Exception as e:
                self.logger.warning(f"🔗 Error filtering link {link}: {e}")
                continue
        
        if filtered_links != links:
            filtered_out = len(links) - len(filtered_links)
            self.logger.info(f"🔗 URL filtering: {len(links)} found → {len(filtered_links)} unvisited ({filtered_out} already visited)")
        return filtered_links
    
    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL by removing query parameters and fragments for comparison."""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            # Keep scheme, netloc, and path; remove params, query, and fragment
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                '',  # params
                '',  # query
                ''   # fragment
            ))
            return normalized
        except Exception as e:
            self.logger.warning(f"🔗 Failed to normalize URL {url}: {e}")
            return None
    
    
    def _is_search_results_content(self, content: str) -> bool:
        """Use LLM to detect if content is primarily search results that should use fast URL extraction."""
        try:
            # For very short content, use simple heuristics
            if len(content) < 200:
                simple_indicators = ["Search Results for:", "URL:", "Snippet:", "using duckduckgo", "using bing", "using google"]
                return sum(1 for indicator in simple_indicators if indicator in content) >= 2
            
            # For longer content, use LLM assessment  
            detection_prompt = f"""Analyze this content to determine if it represents search engine results.

CONTENT SAMPLE:
{content[:800]}...

TASK:
Determine if this content appears to be search engine results (like from Google, DuckDuckGo, Bing, etc.) rather than actual webpage content.

Search results typically have:
- Multiple numbered entries with titles and URLs
- Snippets or descriptions under each result  
- Multiple different domains/sources
- Search engine attribution

Webpage content typically has:
- Continuous narrative or article text
- Single source/domain focus
- More detailed, in-depth information

Respond with EXACTLY ONE of:
- "SEARCH_RESULTS" - This appears to be search engine results
- "WEBPAGE_CONTENT" - This appears to be actual webpage content

Classification:"""

            classification = str(self.llm.generate_response(detection_prompt)).strip().upper()
            is_search_results = "SEARCH_RESULTS" in classification
            
            if is_search_results:
                self.logger.debug(f"🔍 LLM detected search results content")
            else:
                self.logger.debug(f"📄 LLM detected webpage content")
                
            return is_search_results
            
        except Exception as e:
            self.logger.warning(f"❌ LLM content classification failed: {e}")
            # Fallback to simple heuristics when LLM fails
            simple_indicators = ["Search Results for:", "URL:", "Snippet:", "1. ", "2. ", "3. "]
            return sum(1 for indicator in simple_indicators if indicator in content) >= 3
    
    def _extract_search_results_summary(self, knowledge_pieces: List[str], task: str) -> str:
        """Fast extraction of key information from search results without expensive LLM compression."""
        import re
        
        urls = []
        titles = []
        
        for piece in knowledge_pieces:
            # Extract URLs using regex
            url_pattern = r'URL: (https?://[^\s\n]+)'
            found_urls = re.findall(url_pattern, piece)
            urls.extend(found_urls)
            
            # Extract titles/snippets
            title_patterns = [
                r'^\d+\. ([^\n]+)',  # "1. Title"
                r'Title: ([^\n]+)',
                r'([^\n]+)\nURL:',   # Title above URL
            ]
            
            for pattern in title_patterns:
                matches = re.findall(pattern, piece, re.MULTILINE)
                titles.extend(matches)
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        unique_titles = list(dict.fromkeys(titles))
        
        # Create a concise summary focusing on actionable next steps
        summary_parts = []
        
        if unique_urls:
            summary_parts.append(f"Found {len(unique_urls)} relevant sources:")
            for i, (url, title) in enumerate(zip(unique_urls[:5], unique_titles[:5]), 1):
                clean_title = title.strip()[:80] + ("..." if len(title) > 80 else "")
                summary_parts.append(f"  {i}. {clean_title} - {url}")
            
            if len(unique_urls) > 5:
                summary_parts.append(f"  ... and {len(unique_urls) - 5} more sources")
        
        # Add next steps
        summary_parts.append("\nNext: Visit the most relevant documentation sources to find specific command syntax.")
        
        result = "\n".join(summary_parts)
        self.logger.debug(f"📋 Fast search results summary created: {len(result)} chars from {sum(len(p) for p in knowledge_pieces)} chars input")
        
        return result
    
    async def _has_sufficient_information_for_query(self, original_query: str, plan: Optional[Plan]) -> bool:
        """Use LLM to check if we already have sufficient information to answer the query without following more links."""
        try:
            if not plan or not hasattr(plan, 'evolving_answer') or not plan.evolving_answer:
                return False
            
            # Get current accumulated information
            current_answer = plan.evolving_answer
            
            # Use LLM to determine if current information is sufficient, without hard-coded templates
            assessment_prompt = f"""Evaluate whether the current information provides a complete answer to the user's question.

USER'S QUERY: {original_query}

CURRENT ACCUMULATED INFORMATION:
{current_answer}

EVALUATION TASK:
Based on what the user is specifically asking and the information we currently have, determine if we can provide a complete, helpful answer or if we need additional information.

Consider:
- Does the current information directly address the user's question?
- Is the information specific and actionable enough for the user's needs?
- Are there significant gaps that would make the answer incomplete or unhelpful?
- Would additional information likely improve the quality or completeness of the answer?

Respond with EXACTLY ONE of:
- "SUFFICIENT" - Current information adequately answers the user's query
- "INSUFFICIENT" - More information needed for a complete answer

Evaluation:"""

            assessment = str(self.llm.generate_response(assessment_prompt)).strip().upper()
            
            is_sufficient = "SUFFICIENT" in assessment
            
            if is_sufficient:
                self.logger.info(f"🎯 LLM sufficiency evaluation: Current information adequate to answer query")
            else:
                self.logger.debug(f"📋 LLM sufficiency evaluation: Additional information would be helpful")
                
            return is_sufficient
            
        except Exception as e:
            self.logger.warning(f"❌ Failed to assess information sufficiency: {e}")
            return False  # Default to continuing when in doubt
    
    def _should_continue_summarizing(self, summaries: List[str], original_query: str, total_chunks: int, processed: int) -> bool:
        """Determine if we should continue processing more chunks based on sufficiency or content relevance."""
        try:
            if not summaries:
                return True  # Continue if we don't have any summaries yet
                
            if processed < 3:
                return True  # Always process at least first 3 chunks
                
            # Combine current summaries for analysis
            current_info = "\n".join(summaries)
            
            # Check for early termination due to IRRELEVANT content (be conservative)
            # Only trigger irrelevance termination if we've processed a substantial portion (75%+)
            # and recent content is clearly irrelevant to avoid missing valuable information
            if processed >= max(6, total_chunks * 0.75):  # After processing at least 6 chunks OR 75% of content
                irrelevance_check = self._assess_content_relevance(summaries[-3:], original_query)  # Check last 3 summaries for stronger signal
                if irrelevance_check == "IRRELEVANT_REMAINING":
                    remaining_chunks = total_chunks - processed
                    self.logger.info(f"🚫 Conservative early termination: After {processed}/{total_chunks} chunks, recent content appears irrelevant to query, remaining {remaining_chunks} chunks likely unhelpful")
                    return False
            
            # Use LLM for SUFFICIENCY check after processing substantial content (be conservative)
            if processed >= total_chunks * 0.7:  # Only check sufficiency after processing 70% (was 50%)
                self.logger.debug(f"🤔 Conservative sufficiency check: Processed {processed}/{total_chunks} chunks, checking if we have enough information")
                
                sufficiency_prompt = f"""CONSERVATIVE SUFFICIENCY ASSESSMENT: Do we have enough information to fully answer the user's query?

USER'S QUERY: {original_query}

CURRENT ACCUMULATED INFORMATION (from {processed} processed chunks):
{current_info[:1200]}...

CONTEXT: We've processed {processed} out of {total_chunks} total content chunks. 

CONSERVATIVE ASSESSMENT GUIDELINES:
- For technical queries (CLI commands, installation, configuration): Prefer to continue unless you have SPECIFIC command syntax, examples, and parameters
- For GitHub repositories: Important technical details often appear in later sections (usage examples, installation guides)
- When in doubt, continue processing to ensure completeness

ASSESSMENT TASK:
Only declare information "sufficient" if you are confident the user can take immediate action based on current information.

Consider:
- Do we have specific, actionable information (exact commands, parameters, examples)?
- Are we missing crucial implementation details that might appear in remaining chunks?
- For technical documentation, do we have complete installation/usage instructions?
- Is the current information detailed enough for the user to successfully complete their task?

Be CONSERVATIVE - when in doubt, continue processing.

Respond with EXACTLY ONE of:
- "SUFFICIENT" - Only if we have complete, actionable information for the user's query
- "CONTINUE" - Default choice - more information would likely be valuable

Decision:"""

                try:
                    decision = str(self.llm.generate_response(sufficiency_prompt)).strip().upper()
                    
                    if "SUFFICIENT" in decision:
                        self.logger.info(f"🎯 Conservative LLM sufficiency assessment: Information sufficient to answer query after {processed}/{total_chunks} chunks")
                        return False  # Stop processing
                    else:
                        self.logger.debug(f"📋 Conservative LLM sufficiency assessment: Continue processing for completeness")
                        
                except Exception as e:
                    self.logger.warning(f"❌ LLM sufficiency assessment failed: {e}")
                    # Continue processing when LLM fails
            
            # Continue by default unless we've processed almost all content (be conservative)
            return processed < total_chunks * 0.95  # Only stop at 95% if no other criteria met (was 85%)
            
        except Exception as e:
            self.logger.warning(f"❌ Failed to assess summarization continuation: {e}")
            return True  # Default to continuing when in doubt
    
    def _assess_content_relevance(self, recent_summaries: List[str], original_query: str) -> str:
        """Use LLM to assess if recent content suggests remaining chunks will be irrelevant to the query."""
        try:
            if not recent_summaries:
                return "RELEVANT"
            
            # Combine recent summaries for analysis
            recent_content = " ".join(recent_summaries)
            
            # Skip assessment for very short content
            if len(recent_content.strip()) < 30:
                return "RELEVANT"
            
            # Use LLM to assess relevance trend with conservative bias toward continuing
            relevance_prompt = f"""CONSERVATIVE ASSESSMENT: Should we continue processing more content chunks to answer the user's query?

USER'S QUERY: {original_query}

RECENT CONTENT SUMMARIES (from latest processed chunks):
{recent_content}

IMPORTANT CONTEXT:
- We are analyzing technical documentation where specific commands/examples often appear later in the content
- GitHub repositories often have important technical details in later sections (installation, usage, examples)
- For technical queries, err on the side of continuing rather than stopping early

CONSERVATIVE ASSESSMENT TASK: 
Only recommend stopping if you are VERY CONFIDENT that the remaining content is extremely unlikely to contain ANY information that could help answer the user's query.

Consider:
- Could the remaining chunks contain technical examples, command syntax, or installation instructions?
- Is this the type of content where important details often appear later?
- Even if recent content seems off-topic, could it lead to or be followed by relevant sections?
- For technical documentation, is it worth continuing to ensure we don't miss crucial information?

Be CONSERVATIVE - when in doubt, continue processing.

Respond with EXACTLY ONE of:
- "RELEVANT" - Continue processing (default choice - be conservative)
- "IRRELEVANT_REMAINING" - Only if VERY confident remaining chunks won't help

Assessment:"""

            try:
                assessment = str(self.llm.generate_response(relevance_prompt)).strip().upper()
                
                if "IRRELEVANT_REMAINING" in assessment:
                    self.logger.info(f"🚫 LLM assessment: Recent content trend suggests remaining chunks unlikely to help answer query")
                    return "IRRELEVANT_REMAINING"
                else:
                    self.logger.debug(f"✅ LLM assessment: Content still appears relevant, continuing processing")
                    return "RELEVANT"
                    
            except Exception as e:
                self.logger.warning(f"❌ LLM relevance assessment failed: {e}")
                return "RELEVANT"  # Default to continuing when LLM fails
            
        except Exception as e:
            self.logger.warning(f"❌ Failed to assess content relevance: {e}")
            return "RELEVANT"  # Default to continuing when in doubt
    
    def _can_follow_more_links(self, plan: Optional[Plan], original_query: str) -> bool:
        """Check if we can follow more links based on comprehensive limits."""
        try:
            # Global per-query limits
            MAX_TOTAL_LINKS_PER_QUERY = 8  # Maximum total links to follow per query
            MAX_LINK_FOLLOWING_TIME = 60   # Maximum total time spent following links (seconds)
            
            if not plan:
                return True
            
            # Initialize link following tracking if not exists
            if not hasattr(plan, 'links_followed_count'):
                plan.links_followed_count = 0
            if not hasattr(plan, 'link_following_start_time'):
                plan.link_following_start_time = time.time()
            
            # Check total links limit
            if plan.links_followed_count >= MAX_TOTAL_LINKS_PER_QUERY:
                self.logger.info(f"🚫 Link following limit reached: {plan.links_followed_count}/{MAX_TOTAL_LINKS_PER_QUERY} links followed")
                return False
            
            # Check time limit
            elapsed_time = time.time() - plan.link_following_start_time
            if elapsed_time > MAX_LINK_FOLLOWING_TIME:
                self.logger.info(f"⏰ Link following time limit reached: {elapsed_time:.1f}s/{MAX_LINK_FOLLOWING_TIME}s")
                return False
            
            # Check if we have visited too many pages total (URLs from any source)
            if hasattr(plan, 'read_urls') and len(plan.read_urls) > 15:
                self.logger.info(f"📚 Too many pages visited ({len(plan.read_urls)}), limiting additional link following")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"❌ Error checking link following limits: {e}")
            return True  # Default to allowing when in doubt
    
    def _get_remaining_link_budget(self, plan: Optional[Plan]) -> int:
        """Get remaining link budget for this query."""
        MAX_TOTAL_LINKS_PER_QUERY = 8
        
        if not plan or not hasattr(plan, 'links_followed_count'):
            return MAX_TOTAL_LINKS_PER_QUERY
            
        remaining = MAX_TOTAL_LINKS_PER_QUERY - plan.links_followed_count
        return max(0, remaining)
    
    def _consume_link_budget(self, plan: Optional[Plan]) -> None:
        """Consume link budget when visiting a link."""
        if plan:
            if not hasattr(plan, 'links_followed_count'):
                plan.links_followed_count = 0
            plan.links_followed_count += 1
            
            # Log budget consumption
            remaining = self._get_remaining_link_budget(plan)
            self.logger.debug(f"📊 Link budget consumed: {plan.links_followed_count} used, {remaining} remaining")

    async def _visit_relevant_links(self, relevant_links: List[str], original_query: str, plan: Optional[Plan] = None, max_links: int = 2) -> Optional[str]:
        """Proactively visit relevant links with comprehensive limits to prevent endless following."""
        if not relevant_links:
            return None
            
        # Enforce comprehensive link following limits
        if not self._can_follow_more_links(plan, original_query):
            return None
        
        additional_info_parts = []
        visited_count = 0
        
        # Get remaining budget for this query
        remaining_budget = self._get_remaining_link_budget(plan)
        actual_max_links = min(max_links, remaining_budget)
        
        # Limit the number of links to visit
        links_to_visit = relevant_links[:actual_max_links]
        self.logger.info(f"🚀 Visiting {len(links_to_visit)} relevant links (budget: {remaining_budget})")
        
        start_time = time.time()
        MAX_LINK_VISITING_TIME = 30  # Maximum 30 seconds for all link visiting
        
        for i, link in enumerate(links_to_visit, 1):
            try:
                # Time limit check
                elapsed_time = time.time() - start_time
                if elapsed_time > MAX_LINK_VISITING_TIME:
                    remaining_links = len(links_to_visit) - i + 1
                    self.logger.info(f"⏰ Link visiting time limit reached ({MAX_LINK_VISITING_TIME}s), skipping {remaining_links} remaining links")
                    break
                
                # Budget check
                if visited_count >= actual_max_links:
                    break
                
                # Circular reference check
                if plan and hasattr(plan, 'read_urls') and link in plan.read_urls:
                    self.logger.debug(f"🔄 Skipping already visited link: {link}")
                    continue
                
                self.logger.info(f"🔗 Visiting relevant link {i}/{len(links_to_visit)}: {link}")
                
                # Update link following budget
                self._consume_link_budget(plan)
                
                # Create a tool call to read the webpage
                tool_call = {
                    "name": "read_webpage",
                    "parameters": {"url": link, "extract_links": False},  # Don't extract links to avoid recursion
                    "call_id": f"auto_link_{i}_{int(time.time())}"
                }
                
                # Execute the tool call
                result = await self.tool_registry.execute_tool(tool_call)
                
                if result.get("success"):
                    webpage_content = str(result.get("result", ""))
                    
                    # Mark this URL as visited in the plan
                    if plan:
                        plan.mark_url_as_read(link)
                    
                    # Extract key information from the webpage content
                    key_info = await self._extract_key_information(webpage_content, original_query, link)
                    
                    if key_info:
                        additional_info_parts.append(f"📄 From {link}:\n{key_info}")
                        visited_count += 1
                        self.logger.info(f"✅ Extracted useful information from {link}")
                    else:
                        self.logger.info(f"ℹ️  No relevant information found in {link}")
                else:
                    error = result.get("error", "Unknown error")
                    self.logger.warning(f"❌ Failed to visit {link}: {error}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error visiting link {link}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        if additional_info_parts:
            combined_info = "\n\n".join(additional_info_parts)
            self.logger.info(f"🎯 Gathered information from {len(additional_info_parts)} links in {total_time:.1f}s")
            return combined_info
        else:
            self.logger.info("ℹ️  No useful additional information found from visited links")
            return None

    async def _extract_key_information(self, webpage_content: str, original_query: str, source_url: str) -> Optional[str]:
        """Extract key information from webpage content that's relevant to the original query."""
        try:
            # Limit content size for processing
            max_content_length = 2000
            if len(webpage_content) > max_content_length:
                webpage_content = webpage_content[:max_content_length]
            
            # Use LLM to extract relevant information
            extraction_prompt = f"""From the following webpage content, extract ONLY information that is directly relevant to answering this query: "{original_query}"

Webpage content:
{webpage_content}

Extract the most relevant facts, commands, examples, or instructions that help answer the query. Be concise but complete.
If there is no relevant information, respond with "No relevant information found."

Relevant information:"""

            self.logger.debug(f"🧠 Extracting key information from {source_url} for query: {original_query}")
            response = self.llm.generate_response(extraction_prompt)
            
            # Check if the response indicates no relevant information
            if response.strip().lower() in ["no relevant information found.", "no relevant information found", "none", "n/a"]:
                return None
            
            # Clean and validate the response
            cleaned_response = response.strip()
            if len(cleaned_response) < 20:  # Too short to be useful
                return None
                
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract key information from {source_url}: {e}")
            return None

    async def _update_evolving_answer(self, plan: Plan, new_information: str, is_final: bool = False) -> None:
        """Update the evolving answer with new information, prioritizing webpage content over search results."""
        if not new_information or not new_information.strip():
            return

        # Add to knowledge pieces for tracking
        plan.knowledge_pieces.append(new_information)

        if is_final:
            # For final answer, just use it directly
            plan.evolving_answer = new_information
            return

        # Analyze the type of information we're adding
        info_type = self._classify_information_type(new_information)
        self.logger.info(f"📊 Classifying new information as: {info_type}")

        # Build up the evolving answer incrementally with intelligent prioritization
        if not plan.evolving_answer:
            # First piece of information - start the answer, but don't rely on search-only results
            if info_type == "search_results":
                self.logger.info("📊 First information is search results - starting basic structure")
                plan.evolving_answer = f"Research in progress for '{plan.description}'. Found relevant sources that need to be read for complete information."
            else:
                plan.evolving_answer = f"Based on the research for '{plan.description}':\n\n{new_information}"
        else:
            # We have existing answer - use intelligent integration
            try:
                await self._intelligent_integration(plan, new_information, info_type)
            except Exception as e:
                self.logger.warning(f"Failed to integrate new information intelligently: {e}")
                # Fallback: simple append
                plan.evolving_answer += f"\n\nAdditionally:\n{new_information}"

    def _classify_information_type(self, information: str) -> str:
        """Classify the type of information to determine how to integrate it."""
        info_lower = information.lower()
        
        # Check for actual webpage content (high value)
        if "webpage content:" in info_lower:
            return "webpage_content"
        
        # Check for search results (low value until pages are read)
        if "search results for:" in info_lower or "discovered urls" in info_lower:
            return "search_results"
        
        # Check for file content
        if "file:" in info_lower or "directory:" in info_lower:
            return "file_content"
        
        # Check for memory/database content
        if "stored in memory" in info_lower or "search memories" in info_lower:
            return "memory_content"
        
        # Default to general information
        return "general_information"

    async def _intelligent_integration(self, plan: Plan, new_information: str, info_type: str) -> None:
        """Intelligently integrate new information based on its type and value."""
        
        # Different integration strategies based on information type
        if info_type == "webpage_content":
            # Webpage content is high-value - prioritize it
            integration_prompt = f"""Task: {plan.description}

Current Answer (may be incomplete):
{plan.evolving_answer}

NEW HIGH-VALUE WEBPAGE CONTENT (prioritize this information):
{new_information}

The webpage content contains the actual details the user needs. Please integrate this information prominently into the answer, replacing any placeholder or incomplete information from search results. Focus on extracting specific facts like version numbers, dates, features, etc.

Updated Comprehensive Answer:"""

        elif info_type == "search_results":
            # Search results are low-value - don't let them dominate the answer
            if len(plan.knowledge_pieces) == 1:  # This is our first information
                self.logger.info("📊 Keeping answer minimal until we get actual content")
                return  # Don't update the answer with just search results
            else:
                # We have other information - just note that we found more sources
                plan.evolving_answer += f"\n\n[Additional sources identified - content pending]"
                return

        else:
            # General integration for other types
            integration_prompt = f"""Task: {plan.description}

Current Answer:
{plan.evolving_answer}

New Information:
{new_information}

Please integrate this new information into the existing answer to create a more comprehensive response.
Keep the structure clear and avoid redundancy. Focus on building upon what we already know.

Updated Answer:"""

        try:
            # Generate integration context for debug filename
            context_desc = f"integration_{info_type}_{plan.description[:20].replace(' ', '_')}"
            
            updated_answer, debug_filename = self.llm.generate_response_with_debug(integration_prompt, context_desc)
            plan.evolving_answer = updated_answer.strip()
            
            # Log integration debug info
            if debug_filename:
                self.logger.info(f"✅ Successfully integrated {info_type} information (debug: {debug_filename})")
            else:
                self.logger.info(f"✅ Successfully integrated {info_type} information")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate {info_type}: {e}")
            raise

    def _synthesize_final_answer(self, task: str, observations: List[str], evolving_answer: str = "") -> str:
        """Synthesize a final answer from observations and evolving answer."""
        if evolving_answer:
            return evolving_answer

        if not observations:
            return f"Could not complete task: {task}"

        successful_observations = [obs for obs in observations if "succeeded" in obs]

        if successful_observations:
            return "Task completed. Key findings:\n" + "\n".join(successful_observations[-3:])
        else:
            return "Task attempted but encountered issues:\n" + "\n".join(observations[-3:])

    def _create_goal(self, task_description: str) -> Goal:
        """Create a goal from task description."""
        # Simple goal creation - could be enhanced with LLM reasoning
        success_criteria = []

        task_lower = task_description.lower()
        if any(word in task_lower for word in ["find", "search", "information", "about"]):
            success_criteria = ["Find relevant information", "Provide clear explanation"]
        else:
            success_criteria = ["Complete the requested task", "Provide useful result"]

        return Goal(
            description=task_description,
            success_criteria=success_criteria,
        )

    async def _parse_reasoning_response(self, response: str, task: str = "", observations: List[str] = None) -> Dict[str, Any]:
        """Parse LLM reasoning response with enhanced error handling and recovery."""
        try:
            import json
            import re

            self.logger.debug(f"🔍 Parsing LLM response: {response[:200]}...")

            # Clean the response first
            response = response.strip()
            
            # Try to extract JSON by counting braces for proper nesting
            json_candidate = self._extract_json_properly(response)
            
            if json_candidate:
                self.logger.debug(f"🎯 Found JSON candidate: {json_candidate[:200]}...")

                parsed_result = json.loads(json_candidate)
                
                # Validate required fields and provide defaults
                validated_result = self._validate_reasoning_result(parsed_result)
                
                # Validate completion criteria before accepting
                if validated_result.get("complete", False):
                    if not self._validate_completion_criteria(validated_result, observations or [], task):
                        self.logger.warning("🚨 Completion criteria not met - forcing continuation")
                        validated_result["complete"] = False
                        validated_result["reasoning"] = f"Completion blocked: {validated_result.get('reasoning', '')} (Need actual content, not just search results)"
                
                self.logger.info("✅ Successfully parsed JSON reasoning response")
                self.logger.debug(f"   Parsed result: {validated_result}")
                return validated_result
            else:
                # Enhanced fallback parsing - try to extract meaningful content
                return self._intelligent_fallback_parsing(response, task, observations)

        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON parsing failed: {e}")
            self.logger.error(f"   Problematic JSON: {json_candidate[:500] if 'json_candidate' in locals() else 'No candidate found'}")
            
            # Try to recover from common JSON errors
            recovered_result = self._attempt_json_recovery(json_candidate if 'json_candidate' in locals() else response)
            if recovered_result:
                self.logger.info("🔧 Successfully recovered from JSON parsing error")
                return recovered_result
            
            # Final fallback
            self.logger.warning("⚠️  Using intelligent fallback after JSON recovery failed")
            return self._intelligent_fallback_parsing(response, task, observations)
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error parsing reasoning response: {e}")
            self.logger.error(f"   Response was: {response[:500]}...")
            return self._intelligent_fallback_parsing(response, task, observations)

    def _validate_reasoning_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix common issues in reasoning results."""
        validated = {
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "complete": bool(result.get("complete", False)),
            "action": str(result.get("action", "")).strip(),
            "parameters": result.get("parameters", {}),
        }
        
        # Only add final_answer if complete is True
        if validated["complete"]:
            validated["final_answer"] = result.get("final_answer", "Task completed")
        
        # Ensure parameters is a dict
        if not isinstance(validated["parameters"], dict):
            self.logger.warning("⚠️  Parameters not a dict, converting to empty dict")
            validated["parameters"] = {}
        
        return validated

    def _validate_completion_criteria(self, reasoning_result: Dict, observations: List[str], task: str = "") -> bool:
        """Validate that completion criteria are actually met."""
        
        if not reasoning_result.get("complete", False):
            return True  # Not claiming completion, so validation passes
        
        # Check recent observations for continuation indicators
        recent_obs = " ".join(observations[-2:]) if observations else ""
        
        continuation_indicators = [
            "REQUIRED NEXT STEP", "CRITICAL: Task is NOT complete", 
            "MUST do", "need to", "follow up with", "before completing",
            "DISCOVERED URLS", "read_webpage on", "extract information from"
        ]
        
        if any(indicator in recent_obs for indicator in continuation_indicators):
            self.logger.warning(f"🚨 Completion blocked - observations indicate task not complete: found '{[i for i in continuation_indicators if i in recent_obs]}'")
            return False
        
        # Check if we have actual content vs just search results
        if self._only_has_search_results(recent_obs):
            self.logger.warning("🚨 Completion blocked - only search results available, no actual content")
            return False
        
        # For CLI command tasks, ensure we have actual command syntax
        if any(cmd_word in task.lower() for cmd_word in ["command", "cli", "kubectl", "syntax", "cmd"]):
            final_answer = reasoning_result.get("final_answer", "")
            reasoning = reasoning_result.get("reasoning", "")
            combined_text = f"{final_answer} {reasoning}".lower()
            
            # Check if we have actual command indicators, not just reasoning about commands
            has_command_syntax = any(indicator in combined_text for indicator in [
                "kubectl", "--", "create", "apply", "get", "$", "command:", "run"
            ])
            
            if not has_command_syntax:
                self.logger.warning("🚨 Completion blocked - CLI task but no actual command syntax found")
                return False
        
        return True

    def _only_has_search_results(self, recent_obs: str) -> bool:
        """Check if we only have search results without actual content."""
        
        has_search_results = any(indicator in recent_obs for indicator in [
            "Search Results for:", "Found ", " URLs", "DuckDuckGo", "search succeeded"
        ])
        
        has_actual_content = any(indicator in recent_obs for indicator in [
            "webpage content:", "Page content:", "Found content:", 
            "AUTOMATICALLY VISITED RELEVANT LINKS", "Content:", "URL: http",
            "Title:", "Page URL:", "Main content:"
        ])
        
        return has_search_results and not has_actual_content

    def _intelligent_fallback_parsing(self, response: str, task: str = "", observations: List[str] = None) -> Dict[str, Any]:
        """Conservative fallback parsing when JSON extraction fails."""
        observations = observations or []
        response = response.strip()
        
        self.logger.warning(f"🔧 Using intelligent fallback parsing for response: {response[:150]}...")
        
        # Check if we have search results that need to be followed up
        recent_obs = " ".join(observations[-2:]) if observations else ""
        
        if "DISCOVERED URLS" in recent_obs or "https://" in recent_obs:
            # Extract the most relevant URL and continue
            url = self._extract_primary_url_from_observation(recent_obs)
            if url:
                return {
                    "reasoning": "Recovered from JSON parsing error - continuing with webpage reading",
                    "complete": False,
                    "action": "read_webpage",
                    "parameters": {"url": url, "extract_links": True}
                }
        
        # If we have some knowledge but parsing failed, try to synthesize
        if len(observations) > 0 and any("succeeded" in obs for obs in observations):
            return {
                "reasoning": "JSON parsing failed but have some information - attempting to synthesize available data",
                "complete": False,
                "action": "synthesize_information", 
                "parameters": {"query": task}
            }
        
        # Conservative fallback - don't assume completion
        return {
            "reasoning": f"Fallback parsing detected unparseable response. Raw response: {response[:200]}...",
            "complete": False,  # 🚨 NEVER assume completion on parsing failure
            "action": "",
            "parameters": {}
        }

    def _extract_primary_url_from_observation(self, observation: str) -> Optional[str]:
        """Extract the most relevant URL from an observation."""
        import re
        
        # Look for URLs in the observation
        url_pattern = r'https?://[^\s\n<>"]+'
        urls = re.findall(url_pattern, observation)
        
        if not urls:
            return None
        
        # Prioritize certain URL types
        priority_patterns = [
            r'github\.com', r'docs\.', r'documentation', r'pkg\.go\.dev',
            r'\.org', r'\.io', r'official'
        ]
        
        for pattern in priority_patterns:
            for url in urls:
                if re.search(pattern, url, re.IGNORECASE):
                    return url.rstrip('.,;')
        
        # Return first URL if no priority match
        return urls[0].rstrip('.,;')

    async def _extract_discovered_urls_from_search_result(self, search_result: str) -> List[str]:
        """Extract clean URLs from DISCOVERED URLS section of search results."""
        try:
            urls = []
            lines = search_result.split('\n')
            in_discovered_section = False
            
            for line in lines:
                line = line.strip()
                
                if "DISCOVERED URLS" in line:
                    in_discovered_section = True
                    continue
                elif in_discovered_section:
                    # Look for numbered URLs like "1. https://example.com"
                    if line and (line[0].isdigit() or line.startswith('http')):
                        # Extract clean URL
                        if 'http' in line:
                            # Find the URL part, handling various formats
                            import re
                            url_match = re.search(r'https?://[^\s\'"<>]+', line)
                            if url_match:
                                clean_url = url_match.group(0)
                                # Remove trailing punctuation and quotes
                                clean_url = clean_url.rstrip('.,;\'")]}')
                                urls.append(clean_url)
                    elif not line or line.startswith('=') or 'END' in line.upper():
                        # End of DISCOVERED URLS section
                        break
            
            self.logger.info(f"🔍 Extracted {len(urls)} clean URLs from DISCOVERED URLS section")
            return urls
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract discovered URLs: {e}")
            return []

    async def _visit_discovered_urls(self, discovered_urls: List[str], original_query: str, plan: Optional[Plan] = None, max_urls: int = 2) -> Optional[str]:
        """Visit discovered URLs and gather information, prioritizing most relevant ones."""
        if not discovered_urls:
            return None
        
        # Pre-filter and rank URLs by relevance
        relevant_urls = self._rank_discovered_urls_by_relevance(discovered_urls, original_query)
        urls_to_visit = relevant_urls[:max_urls]  # Visit top 2 most relevant
        
        if not urls_to_visit:
            self.logger.info("🔍 No URLs passed relevance filtering")
            return None
        
        self.logger.info(f"🚀 Visiting top {len(urls_to_visit)} discovered URLs for more information")
        
        additional_info_parts = []
        visited_count = 0
        
        for i, url in enumerate(urls_to_visit, 1):
            try:
                self.logger.info(f"🔍 Visiting discovered URL {i}/{len(urls_to_visit)}: {url}")
                
                # Create a tool call to read the webpage
                tool_call = {
                    "name": "read_webpage",
                    "parameters": {"url": url, "extract_links": True},  # Allow link extraction for follow-up
                    "call_id": f"discovered_url_{i}_{datetime.now().timestamp()}"
                }
                
                # Execute the tool call
                result = await self.tool_registry.execute_tool(tool_call)
                
                if result.get("success"):
                    webpage_content = str(result.get("result", ""))
                    
                    # Mark this URL as visited in the plan
                    if plan:
                        plan.mark_url_as_read(url)
                    
                    # Extract key information from the webpage content
                    key_info = await self._extract_key_information(webpage_content, original_query, url)
                    
                    if key_info:
                        additional_info_parts.append(f"📄 From {url}:\n{key_info}")
                        visited_count += 1
                        self.logger.info(f"✅ Extracted useful information from discovered URL {url}")
                    else:
                        self.logger.info(f"ℹ️  No relevant information found in discovered URL {url}")
                else:
                    error = result.get("error", "Unknown error")
                    self.logger.warning(f"❌ Failed to visit discovered URL {url}: {error}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error visiting discovered URL {url}: {e}")
                continue
        
        if additional_info_parts:
            combined_info = "\n\n".join(additional_info_parts)
            self.logger.info(f"🎯 Successfully gathered information from {len(additional_info_parts)} discovered URLs")
            return combined_info
        else:
            self.logger.info("ℹ️  No useful information found from visited discovered URLs")
            return None

    def _rank_discovered_urls_by_relevance(self, urls: List[str], original_query: str) -> List[str]:
        """Rank discovered URLs by relevance to the query."""
        query_lower = original_query.lower()
        query_terms = set(word for word in query_lower.split() if len(word) >= 3)
        
        ranked_urls = []
        
        for url in urls:
            url_lower = url.lower()
            score = 0
            
            # Higher priority for documentation, CLI guides, examples
            high_priority_patterns = [
                '/docs/', '/documentation/', '/guide/', '/cli/', '/command',
                '/examples/', '/tutorial/', '/howto/', '/getting-started/',
                'docs.redhat.com', 'kubevirt.io/user-guide'
            ]
            
            for pattern in high_priority_patterns:
                if pattern in url_lower:
                    score += 10
            
            # Medium priority for official repos and tools
            medium_priority_patterns = [
                'github.com/', '/forklift', '/kubectl-mtv', '/migration',
                'redhat.com', 'kubernetes.io'
            ]
            
            for pattern in medium_priority_patterns:
                if pattern in url_lower:
                    score += 5
            
            # Score based on query term matches
            for term in query_terms:
                if term in url_lower:
                    score += 3
            
            # Penalize generic or marketing URLs
            penalty_patterns = ['/blog/', '/news/', '/about', '/contact', '/pricing']
            for pattern in penalty_patterns:
                if pattern in url_lower:
                    score -= 5
            
            ranked_urls.append((score, url))
        
        # Sort by score descending and return URLs
        ranked_urls.sort(key=lambda x: x[0], reverse=True)
        result = [url for score, url in ranked_urls if score > 0]
        
        self.logger.info(f"🔍 Ranked {len(urls)} URLs → {len(result)} relevant URLs")
        return result

    def _fallback_reasoning_parsing(self, response: str) -> Dict[str, Any]:
        """DEPRECATED: Use _intelligent_fallback_parsing instead."""
        self.logger.warning("⚠️  Using deprecated fallback parsing - should use intelligent fallback")
        return self._intelligent_fallback_parsing(response)
        
        # Look for tool mentions
        available_tools = [tool['name'] for tool in self.tool_registry.get_tools_for_llm()]
        mentioned_tools = [tool for tool in available_tools if tool in response.lower()]
        
        if mentioned_tools:
            suggested_tool = mentioned_tools[0]  # Use first mentioned tool
            self.logger.info(f"🎯 Detected tool mention '{suggested_tool}', suggesting action")
            return {
                "reasoning": f"Fallback parsing detected tool mention: {response[:100]}...",
                "complete": False,
                "action": suggested_tool,
                "parameters": {},
                "final_answer": ""
            }
        
        # Ultimate fallback - suggest synthesis or completion
        if len(response) > 50:  # If we have substantial content, try synthesis
            return {
                "reasoning": f"Fallback parsing with substantial content: {response[:100]}...",
                "complete": False,
                "action": "synthesize_information",
                "parameters": {},
                "final_answer": ""
            }
        else:
            # Very short response, likely an error - complete with what we have
            return {
                "reasoning": f"Fallback parsing with minimal content: {response}",
                "complete": True,
                "action": "",
                "parameters": {},
                "final_answer": f"Unable to parse response properly. Raw response: {response}"
            }

    def _attempt_json_recovery(self, json_text: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover from common JSON parsing errors."""
        import json
        import re
        
        if not json_text:
            return None
            
        # Common fixes
        fixes = [
            # Fix unescaped quotes in strings
            lambda x: re.sub(r'("reasoning":\s*"[^"]*)"([^"]*"[^"]*")', r'\1\"\2', x),
            # Fix trailing commas
            lambda x: re.sub(r',(\s*[}\]])', r'\1', x),
            # Fix missing quotes around keys
            lambda x: re.sub(r'(\w+)(\s*:)', r'"\1"\2', x),
            # Fix single quotes to double quotes
            lambda x: x.replace("'", '"'),
        ]
        
        for fix in fixes:
            try:
                fixed_json = fix(json_text)
                result = json.loads(fixed_json)
                self.logger.info(f"🔧 JSON recovery successful with fix")
                return self._validate_reasoning_result(result)
            except:
                continue
        
        return None

    def _extract_json_properly(self, response: str) -> Optional[str]:
        """Extract JSON by counting braces to handle nested structures properly."""
        # Find the first opening brace
        start = response.find('{')
        if start == -1:
            return None
            
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response[start:], start):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                if brace_count == 0:
                    # Found the matching closing brace
                    return response[start:i+1]
        
        # If we get here, braces weren't balanced
        return None

    def _get_accumulated_information(self) -> str:
        """Get all accumulated information for synthesis."""
        if not self.current_plan:
            return ""

        information_pieces = []

        # Add evolving answer if available
        if self.current_plan.evolving_answer:
            information_pieces.append(f"Current accumulated answer:\n{self.current_plan.evolving_answer}")

        # Add knowledge pieces with intelligent compression for large volumes
        if self.current_plan.knowledge_pieces:
            all_knowledge = "\n---\n".join(self.current_plan.knowledge_pieces)
            
            # Use intelligent compression if we have substantial knowledge accumulated
            if len(all_knowledge) > 2000 and len(self.current_plan.knowledge_pieces) > 3:
                compressed_knowledge = self._intelligent_context_compression(
                    all_knowledge, 
                    "synthesize accumulated knowledge", 
                    max_context_chars=1500
                )
                information_pieces.append("Accumulated Knowledge (compressed):")
                information_pieces.append(compressed_knowledge)
                self.logger.info(f"📊 Compressed {len(self.current_plan.knowledge_pieces)} knowledge pieces from {len(all_knowledge):,} to {len(compressed_knowledge):,} chars")
            else:
                # For smaller volumes, list individually  
                information_pieces.append("Individual knowledge pieces:")
                for i, piece in enumerate(self.current_plan.knowledge_pieces, 1):
                    information_pieces.append(f"{i}. {piece}")

        return "\n\n".join(information_pieces) if information_pieces else "No accumulated information available."
