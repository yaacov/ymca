"""Web search functionality with multiple search engines and result parsing."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import validators
from bs4 import BeautifulSoup, Tag

from ..llm.llm import LLM
from .models import SearchResult
from .selenium_manager import SeleniumManager


class SearchEngine:
    """Handles web search operations across multiple search engines."""

    def __init__(self, llm: LLM, selenium_manager: Optional[SeleniumManager] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize search engine.

        Args:
            llm: LLM instance for intelligent processing
            selenium_manager: Selenium manager for web requests
            logger: Logger instance
        """
        self.llm = llm
        self.selenium_manager = selenium_manager or SeleniumManager(logger)
        self.logger = logger or logging.getLogger("ymca.web.search")

        # Search engine endpoints
        self.search_engines = {"duckduckgo": "https://duckduckgo.com/html/?q={query}", "bing": "https://www.bing.com/search?q={query}", "google": "https://www.google.com/search?q={query}"}

    async def search_web(self, query: str, engine: str = "duckduckgo", max_results: int = 10) -> List[SearchResult]:
        """
        Search the web and return results with automatic fallback to other engines.

        Args:
            query: Search query
            engine: Search engine to use (duckduckgo, bing, google)
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        if engine not in self.search_engines:
            raise ValueError(f"Unsupported search engine: {engine}")

        # Try primary search engine first
        results = await self._search_single_engine(query, engine, max_results)

        # If no results, try fallback engines
        if not results:
            self.logger.info(f"No results from {engine}, trying fallback engines")
            fallback_engines = [eng for eng in self.search_engines.keys() if eng != engine]

            for fallback_engine in fallback_engines:
                self.logger.info(f"Trying fallback engine: {fallback_engine}")
                results = await self._search_single_engine(query, fallback_engine, max_results)
                if results:
                    self.logger.info(f"Found results using fallback engine: {fallback_engine}")
                    break

        # Use LLM to score relevance
        if results:
            results = await self._score_search_results(query, results)

        self.logger.info(f"Found {len(results)} search results")
        return results[:max_results]

    async def _search_single_engine(self, query: str, engine: str, max_results: int) -> List[SearchResult]:
        """Search using a single search engine with unified parsing."""
        try:
            search_url = self.search_engines[engine].format(query=query.replace(" ", "+"))
            self.logger.info(f"Searching '{query}' using {engine}")

            # Use Selenium for web scraping
            page_source = await self.selenium_manager.request_page(search_url)

            if not page_source:
                self.logger.warning(f"Failed to get page content for {engine}")
                return []

            soup = BeautifulSoup(page_source.encode("utf-8"), "html.parser")

            # Use unified parsing for all search engines
            results = await self._parse_search_results_unified(soup, engine, max_results)

            return results

        except Exception as e:
            self.logger.error(f"Search failed for {engine}: {e}")
            return []

    async def _parse_search_results_unified(self, soup: BeautifulSoup, engine: str, max_results: int) -> List[SearchResult]:
        """
        Unified search results parser using simple heuristics that work across all search engines.

        The approach uses simple patterns common to all search engines:
        1. Find all links on the page
        2. Score them based on simple heuristics
        3. Return the highest scoring ones
        """
        results: List[SearchResult] = []

        # Simple heuristic: find all links and score them
        all_links = soup.find_all("a", href=True)
        candidates = []

        for link in all_links:
            if not isinstance(link, Tag):
                continue

            href = str(link.get("href", ""))
            if not href or href.startswith("#"):
                continue

            # Get link text (title)
            title = link.get_text().strip()
            if len(title) < 3:
                continue

            # Get context from surrounding elements
            context = self._extract_link_context(link)

            # Clean up URL
            href = self._clean_search_url(href, engine)

            # Skip if not a valid URL after cleanup
            if not validators.url(href):
                continue

            candidates.append({"url": href, "title": title, "context": context, "link_element": link})

        # Score and filter candidates using simple heuristics
        scored_results = self._score_search_candidates_simple(candidates, engine)

        # Take top results
        results = scored_results[:max_results]

        self.logger.info(f"Extracted {len(results)} search results from {engine} using unified parsing")
        return results

    def _clean_search_url(self, href: str, engine: str) -> str:
        """Clean up search result URLs."""
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            # Relative URL - construct absolute URL based on engine
            if engine.lower() == "duckduckgo":
                href = "https://duckduckgo.com" + href
            elif engine.lower() == "bing":
                href = "https://www.bing.com" + href
            elif engine.lower() == "google":
                href = "https://www.google.com" + href

        return href

    def _extract_link_context(self, link: Tag) -> str:
        """Extract context text around a link for better understanding."""
        context_text = ""

        # Get text from the link itself
        link_text = link.get_text().strip()

        # Strategy 1: Look for search engine specific patterns
        context_text = self._extract_search_engine_snippet(link)
        if context_text and len(context_text.strip()) > 20:
            return context_text[:300]

        # Strategy 2: Get text from parent element
        parent = link.parent
        if parent and isinstance(parent, Tag):
            parent_text = parent.get_text().strip()
            # Remove the link text to avoid duplication
            if link_text in parent_text:
                parent_text = parent_text.replace(link_text, "").strip()
            context_text = parent_text

        # Strategy 3: Look for nearby descriptive text elements
        if link.parent:
            # Look for common description containers
            description_containers = link.parent.find_all(['p', 'span', 'div'], 
                                                         class_=lambda x: x and any(word in str(x).lower() 
                                                                                   for word in ['desc', 'snippet', 'summary', 'abstract']))
            for container in description_containers:
                desc_text = container.get_text().strip()
                if len(desc_text) > 20 and desc_text not in context_text:
                    context_text += " " + desc_text

        # Strategy 4: Look for siblings with substantial text
        if link.parent and len(context_text.strip()) < 20:
            for sibling in link.parent.find_next_siblings():
                if isinstance(sibling, Tag):
                    text = sibling.get_text().strip()
                    if len(text) > 20 and text not in context_text:
                        context_text += " " + text
                        break

        # Limit context length and clean up
        context_text = re.sub(r"\s+", " ", context_text).strip()[:300]
        return context_text

    def _extract_search_engine_snippet(self, link: Tag) -> str:
        """Extract snippet text using search engine specific patterns."""
        snippet = ""
        
        # DuckDuckGo patterns
        if link.parent:
            # Look for result snippet in DuckDuckGo structure
            result_container = link.parent
            for _ in range(3):  # Go up max 3 levels
                if result_container and isinstance(result_container, Tag):
                    # Look for snippet text in common DuckDuckGo classes
                    snippet_elem = result_container.find(class_=lambda x: x and any(word in str(x).lower() 
                                                                                  for word in ['snippet', 'result__snippet', 'abstract']))
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()
                        if len(snippet) > 20:
                            break
                    result_container = result_container.parent
                else:
                    break

        # Google patterns  
        if not snippet and link.parent:
            # Look for Google result descriptions
            result_container = link.parent
            for _ in range(3):
                if result_container and isinstance(result_container, Tag):
                    desc_elem = result_container.find('span', class_=lambda x: x and 'st' in str(x).lower())
                    if not desc_elem:
                        desc_elem = result_container.find('div', class_=lambda x: x and any(word in str(x).lower() 
                                                                                          for word in ['vvjwjb', 'f', 's']))
                    if desc_elem:
                        snippet = desc_elem.get_text().strip()
                        if len(snippet) > 20:
                            break
                    result_container = result_container.parent
                else:
                    break

        # Bing patterns
        if not snippet and link.parent:
            # Look for Bing result descriptions
            result_container = link.parent
            for _ in range(3):
                if result_container and isinstance(result_container, Tag):
                    desc_elem = result_container.find(class_=lambda x: x and any(word in str(x).lower() 
                                                                               for word in ['b_caption', 'b_snippet']))
                    if desc_elem:
                        snippet = desc_elem.get_text().strip()
                        if len(snippet) > 20:
                            break
                    result_container = result_container.parent
                else:
                    break

        return snippet

    def _score_search_candidates_simple(self, candidates: List[Dict[str, Any]], engine: str) -> List[SearchResult]:
        """Score search result candidates using simple heuristics."""
        scored_results = []

        engine_domains = {"duckduckgo": "duckduckgo.com", "bing": "bing.com", "google": "google.com"}

        engine_domain = engine_domains.get(engine.lower(), "")

        for candidate in candidates:
            url = candidate["url"]
            title = candidate["title"]
            context = candidate.get("context", "")

            # Skip internal search engine links
            if engine_domain and engine_domain in url and not self._is_external_redirect(url, engine):
                continue

            # Skip obviously non-result links
            if self._is_navigation_link(title, url):
                continue

            # Calculate simple score
            score = self._calculate_simple_score(title, url, context)

            if score > 0:
                scored_results.append(SearchResult(title=title, url=url, description=context, relevance_score=score / 100.0))  # Normalize to 0-1

        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)

        return scored_results

    def _is_external_redirect(self, url: str, engine: str) -> bool:
        """Check if URL is an external redirect (actual search result)."""
        # These are patterns that indicate the URL is redirecting to external content
        redirect_patterns = [
            "/url?",  # Google redirects
            "/l/?u=",  # DuckDuckGo redirects
            "linkId=",  # Bing redirects
            "uddg=",  # DuckDuckGo
            "sa=D&url=",  # Google
        ]

        return any(pattern in url for pattern in redirect_patterns)

    def _is_navigation_link(self, title: str, url: str) -> bool:
        """Check if this is likely a navigation/utility link rather than a search result."""
        title_lower = title.lower()
        url_lower = url.lower()

        # Skip common navigation terms
        nav_terms = [
            "privacy",
            "terms",
            "help",
            "about",
            "contact",
            "support",
            "login",
            "sign up",
            "sign in",
            "settings",
            "preferences",
            "cookie",
            "legal",
            "advertise",
            "business",
            "careers",
            "images",
            "videos",
            "news",
            "maps",
            "more",
            "all",
            "previous",
            "next",
            "page",
            "advanced search",
        ]

        if any(term in title_lower for term in nav_terms):
            return True

        # Skip very short titles that are likely navigation
        if len(title) < 10 and not any(term in title_lower for term in ["kubectl", "mtv", "k8s"]):
            return True

        # Skip URLs that look like navigation
        nav_url_patterns = ["/about", "/help", "/support", "/contact", "/privacy", "/terms", "/login", "/signin", "/signup", "/settings", "/preferences"]

        if any(pattern in url_lower for pattern in nav_url_patterns):
            return True

        return False

    def _calculate_simple_score(self, title: str, url: str, context: str) -> float:
        """Calculate a simple relevance score for a search result candidate."""
        score = 0.0
        title_lower = title.lower()
        context_lower = context.lower()
        url_lower = url.lower()

        # Base score for having reasonable title length
        score += min(len(title), 100) * 0.2

        # Bonus for external domains (likely actual results)
        external_domains = ["github.com", "gitlab.com", "stackoverflow.com", "docs.", "kubernetes.io", "redhat.com", "openshift.com", "kubevirt.io"]

        if any(domain in url_lower for domain in external_domains):
            score += 30

        # Bonus for relevant keywords in title
        relevant_terms = ["kubectl", "mtv", "migration", "virtualization", "kubevirt", "openshift", "kubernetes"]
        for term in relevant_terms:
            if term in title_lower:
                score += 15
            if term in context_lower:
                score += 8
            if term in url_lower:
                score += 10

        # Special bonus for exact matches
        if "kubectl-mtv" in title_lower or "kubectl-mtv" in context_lower:
            score += 50

        # Bonus for GitHub/documentation links
        if "github.com" in url_lower:
            score += 25
        if "docs." in url_lower or "documentation" in url_lower:
            score += 20

        # Penalty for very long URLs (often redirects/tracking)
        if len(url) > 150:
            score -= 10

        # Penalty for URLs with lots of parameters (likely tracking/ads)
        if url.count("&") > 3:
            score -= 15

        return max(0, score)  # Ensure non-negative

    async def _score_search_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Use LLM to score search result relevance."""
        try:
            # Create prompt for relevance scoring
            results_text = "\n".join([f"{i+1}. {result.title}\n   {result.description}\n   URL: {result.url}" for i, result in enumerate(results)])

            prompt = f"""Given the search query: "{query}"

Rate the relevance of each search result on a scale of 0.0 to 1.0, where 1.0 is highly relevant and 0.0 is not relevant.

Search Results:
{results_text}

Respond with only a JSON array of scores in the same order as the results, like: [0.9, 0.7, 0.3, ...]"""

            response = self.llm.generate_response(prompt)

            # Better error handling for JSON parsing
            if not response or not response.strip():
                self.logger.warning("LLM returned empty response for scoring, using original order")
                return results

            # Try to extract JSON from response
            json_response = response.strip()

            # Look for JSON array pattern in the response
            json_match = re.search(r"\[[\d.,\s]+\]", json_response)
            if json_match:
                json_response = json_match.group()

            try:
                scores = json.loads(json_response)

                # Ensure it's a list and has valid scores
                if not isinstance(scores, list) or len(scores) != len(results):
                    self.logger.warning("LLM scoring response has incorrect format, using original order")
                    return results

                # Update results with scores
                for i, score in enumerate(scores):
                    if i < len(results) and isinstance(score, (int, float)):
                        results[i].relevance_score = float(score)

                # Sort by relevance score
                results.sort(key=lambda x: x.relevance_score or 0, reverse=True)

            except (json.JSONDecodeError, ValueError) as json_error:
                self.logger.warning(f"Failed to parse LLM scoring response: {json_error}, response: {json_response[:100]}...")

        except Exception as e:
            self.logger.warning(f"Failed to score search results with LLM: {e}")

        return results

    async def find_relevant_links(self, search_results: List[SearchResult], criteria: str, max_links: int = 5) -> List[SearchResult]:
        """
        Use LLM to find the most relevant links from search results.

        Args:
            search_results: List of search results
            criteria: Criteria for relevance (e.g., "technical documentation", "news articles")
            max_links: Maximum number of links to return

        Returns:
            Filtered and sorted list of most relevant search results
        """
        if not search_results:
            return search_results[:max_links]

        try:
            results_text = "\n".join([f"{i+1}. {result.title}\n   {result.description}\n   URL: {result.url}" for i, result in enumerate(search_results)])

            prompt = f"""Given the following search results, identify the {max_links} most relevant links based on this criteria: "{criteria}"

Search Results:
{results_text}

Respond with only a JSON array containing the numbers of the most relevant results in order of relevance (e.g., [3, 1, 7, 2, 5]):"""

            response = self.llm.generate_response(prompt)

            # Better error handling for JSON parsing
            if not response or not response.strip():
                self.logger.warning("LLM returned empty response for relevance filtering, using original order")
                return search_results[:max_links]

            # Try to extract JSON from response
            json_response = response.strip()

            # Look for JSON array pattern in the response
            json_match = re.search(r"\[[\d,\s]+\]", json_response)
            if json_match:
                json_response = json_match.group()

            try:
                indices = json.loads(json_response)

                # Ensure it's a list
                if not isinstance(indices, list):
                    self.logger.warning("LLM relevance response is not a list, using original order")
                    return search_results[:max_links]

                # Filter results based on LLM selection
                relevant_results = []
                for idx in indices:
                    if isinstance(idx, int) and 1 <= idx <= len(search_results):
                        relevant_results.append(search_results[idx - 1])

                self.logger.info(f"LLM selected {len(relevant_results)} relevant links")
                return relevant_results[:max_links]

            except (json.JSONDecodeError, ValueError) as json_error:
                self.logger.warning(f"Failed to parse LLM relevance response: {json_error}, response: {json_response[:100]}...")
                return search_results[:max_links]

        except Exception as e:
            self.logger.warning(f"Failed to find relevant links with LLM: {e}")
            return search_results[:max_links]
