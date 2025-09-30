"""Web browser simulation with state management, search, and content extraction."""

import logging
import time
from typing import Any, Dict, List, Optional

import validators
from asyncio_throttle import Throttler
from bs4 import BeautifulSoup

from ..llm.llm import LLM
from .content_extractor import ContentExtractor
from .models import SearchResult, WebPage
from .search_engine import SearchEngine
from .selenium_manager import SeleniumManager
from .web_archive import WebArchive


class WebBrowser:
    """Web browser simulation with intelligent content extraction and LLM integration."""

    def __init__(self, llm: LLM, max_requests_per_second: float = 0.5, max_requests_per_minute: int = 20, archive_dir: str = "data/web_archive", cache_max_age_hours: int = 24, logger: Optional[logging.Logger] = None):
        """
        Initialize web browser.

        Args:
            llm: LLM instance for intelligent processing
            max_requests_per_second: Rate limit for requests per second
            max_requests_per_minute: Rate limit for requests per minute
            archive_dir: Directory for web page archive
            cache_max_age_hours: Maximum age in hours for cached pages
            logger: Logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger("ymca.web")

        # Request throttling
        self.throttler_per_second = Throttler(rate_limit=int(max_requests_per_second * 10), period=10.0)
        self.throttler_per_minute = Throttler(rate_limit=max_requests_per_minute, period=60.0)

        # Browser state
        self.session_state: Dict[str, Any] = {}
        self.cookies: Dict[str, str] = {}
        self.visited_urls: List[str] = []

        # Initialize components
        self.selenium_manager = SeleniumManager(logger=self.logger)
        self.content_extractor = ContentExtractor(llm=self.llm, logger=self.logger)
        self.search_engine = SearchEngine(llm=self.llm, selenium_manager=self.selenium_manager, logger=self.logger)
        self.web_archive = WebArchive(archive_dir=archive_dir, max_age_hours=cache_max_age_hours, logger=self.logger)

        # Periodic cache cleanup
        self.web_archive.clear_expired()

        self.logger.info("Web browser initialized with modular components and web archive")

    def _rotate_user_agent(self) -> None:
        """Rotate user agent to avoid detection."""
        self.selenium_manager.rotate_user_agent()

    async def _selenium_request(self, url: str, timeout: int = 30) -> Optional[str]:
        """Make a request using selenium for more robust scraping."""
        page_source = await self.selenium_manager.request_page(url, timeout)
        if page_source:
            self.visited_urls.append(url)
        return page_source


    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML with enhanced content detection."""
        return self.content_extractor.extract_text_content(soup)

    def _extract_page_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page metadata."""
        return self.content_extractor.extract_page_metadata(soup)

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
        return await self.search_engine.search_web(query, engine, max_results)

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
        return await self.search_engine.find_relevant_links(search_results, criteria, max_links)

    async def read_webpage(self, url: str, extract_links: bool = True, force_refresh: bool = False) -> Optional[WebPage]:
        """
        Read and extract content from a web page, using archive when possible.

        Args:
            url: URL to read
            extract_links: Legacy parameter - link extraction now uses LLM-based content analysis
            force_refresh: Force fresh fetch, bypassing cache

        Returns:
            WebPage object with extracted content or None if failed
        """
        if not validators.url(url):
            self.logger.error(f"Invalid URL: {url}")
            return None

        # Check archive first unless force refresh is requested
        if not force_refresh:
            cached_page = self.web_archive.get_cached_page(url)
            if cached_page:
                return cached_page

        self.logger.info(f"Fetching webpage: {url}")

        page_source = await self._selenium_request(url)
        if not page_source:
            return None

        soup = BeautifulSoup(page_source, "html.parser")

        # Extract content
        title = self._extract_page_metadata(soup).get("title", "")
        content = self._extract_text_content(soup)
        # Only use LLM-based link extraction from content, not mechanical HTML parsing
        links = []  # Disable mechanical link extraction - use LLM analysis instead
        metadata = self._extract_page_metadata(soup)

        # Use hybrid extraction: static analysis + LLM for best results
        raw_links = []
        if content:
            content, raw_links = await self.content_extractor.hybrid_content_extraction(content, url, soup)

        webpage = WebPage.create_now(url=url, title=title, content=content, links=raw_links, metadata=metadata)

        # Cache the successfully fetched page
        self.web_archive.cache_page(webpage)

        self.logger.info(f"Extracted {len(content)} chars + {len(raw_links)} raw links from {url} (using hybrid content+link extraction)")
        return webpage

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get web archive statistics."""
        return self.web_archive.get_cache_stats()

    def clear_archive_expired(self) -> int:
        """Clear expired pages from archive."""
        return self.web_archive.clear_expired()

    def remove_from_archive(self, url: str) -> bool:
        """Remove a specific URL from archive."""
        return self.web_archive.remove_url(url)

    async def smart_web_search(self, query: str, criteria: Optional[str] = None, max_results: int = 5, read_pages: bool = True) -> Dict[str, Any]:
        """
        Perform an intelligent web search with content extraction.

        Args:
            query: Search query
            criteria: Criteria for link relevance (optional)
            max_results: Maximum number of results to process
            read_pages: Whether to read the content of found pages

        Returns:
            Dictionary containing search results and extracted content
        """
        self.logger.info(f"Starting smart web search for: {query}")

        # Perform web search
        search_results = await self.search_web(query, max_results=max_results * 2)

        if not search_results:
            return {"query": query, "results": [], "pages": []}

        # Filter for relevant links if criteria provided
        if criteria:
            search_results = await self.find_relevant_links(search_results, criteria, max_results)
        else:
            search_results = search_results[:max_results]

        # Read page contents if requested
        pages = []
        if read_pages:
            for result in search_results:
                page = await self.read_webpage(result.url)
                if page:
                    pages.append(page)

        return {"query": query, "criteria": criteria, "results": search_results, "pages": pages, "search_timestamp": time.time()}

    def get_session_state(self) -> Dict[str, Any]:
        """Get current browser session state."""
        return {"visited_urls": self.visited_urls.copy(), "cookies": self.cookies.copy(), "session_state": self.session_state.copy(), "current_user_agent": getattr(self.selenium_manager, "current_user_agent", "")}

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Set browser session state."""
        self.visited_urls = state.get("visited_urls", [])
        self.cookies = state.get("cookies", {})
        self.session_state = state.get("session_state", {})

    def clear_session(self) -> None:
        """Clear browser session state."""
        self.visited_urls.clear()
        self.cookies.clear()
        self.session_state.clear()
        self.selenium_manager.close_driver()
        self.logger.info("Browser session cleared")

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        try:
            self.selenium_manager.close_driver()
        except Exception:
            pass  # Ignore errors during cleanup
