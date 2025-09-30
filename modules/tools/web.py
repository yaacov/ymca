"""Web-related tools for information gathering and browsing."""

from typing import Any, Callable, Dict, List, Tuple, cast


def create_web_tools(web_browser: Any) -> List[Tuple[Dict[str, Any], Callable[..., Any]]]:
    """Create comprehensive web browser tools for search, content extraction, and session management."""
    tools = []

    # Basic web search tool - just search results without content extraction
    search_web_tool = {
        "name": "search_web",
        "description": (
            "Perform basic web search and get search results with titles, URLs, "
            "and snippets. Use this when you only need search results without "
            "full content extraction, or want to see many results quickly. "
            "Does not read page content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "engine": {"type": "string", "description": "Search engine to use", "enum": ["duckduckgo", "bing", "google"], "default": "duckduckgo"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
            },
            "required": ["query"],
        },
        "category": "web",
        "enabled": True,
    }

    def _extract_clean_url(raw_url: str) -> str:
        """Extract clean URL from search engine redirect URLs."""
        import urllib.parse
        
        # Handle DuckDuckGo redirect URLs
        if "duckduckgo.com/l/?uddg=" in raw_url:
            try:
                # Extract the uddg parameter which contains the encoded target URL
                parsed = urllib.parse.urlparse(raw_url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'uddg' in params:
                    encoded_url = params['uddg'][0]
                    # URL decode it
                    clean_url = urllib.parse.unquote(encoded_url)
                    return clean_url
            except Exception:
                pass
        
        # Handle other redirect patterns if needed
        # For now, just return the original URL if no extraction is possible
        return raw_url

    async def search_web_handler(query: str, engine: str = "duckduckgo", max_results: int = 10) -> str:
        results = await web_browser.search_web(query, engine, max_results)
        if not results:
            return f"No search results found for: {query}"

        formatted = [f"Search Results for: {query} (using {engine})"]
        clean_urls = []
        
        for i, result in enumerate(results[:max_results], 1):
            # Extract clean URL
            clean_url = _extract_clean_url(result.url)
            clean_urls.append(clean_url)
            
            formatted.append(f"{i}. {result.title}")
            formatted.append(f"URL: {clean_url}")
            formatted.append(f"Snippet: {result.description}\n")

        # Add a summary of discovered URLs for easy reference
        formatted.append("DISCOVERED URLS (ready for read_webpage):")
        for i, url in enumerate(clean_urls, 1):
            formatted.append(f"{i}. {url}")

        return "\n".join(formatted)

    # Find relevant links tool
    find_relevant_links_tool = {
        "name": "find_relevant_links",
        "description": (
            "Use LLM to intelligently filter and rank search results based on "
            "relevance criteria. Use this after getting search results when you "
            "need to identify the most relevant sources for specific needs "
            "(e.g., 'official documentation', 'recent news', 'tutorials')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_results_text": {"type": "string", "description": "Text containing search results (from search_web tool)"},
                "criteria": {"type": "string", "description": "Criteria for relevance filtering (e.g., 'technical documentation', 'news articles')"},
                "max_links": {"type": "integer", "description": "Maximum number of relevant links to return", "default": 5},
            },
            "required": ["search_results_text", "criteria"],
        },
        "category": "web",
        "enabled": True,
    }

    async def find_relevant_links_handler(search_results_text: str, criteria: str, max_links: int = 5) -> str:
        """Extract and rank URLs from search results based on relevance criteria."""
        import re
        
        # Extract URLs from search results
        urls = []
        lines = search_results_text.split('\n')
        
        for line in lines:
            if line.strip().startswith('URL:'):
                url = line.replace('URL:', '').strip()
                if url and url.startswith('http'):
                    urls.append(url)
        
        if not urls:
            return f"No URLs found in search results to filter with criteria: {criteria}"
        
        # For now, implement simple relevance scoring based on criteria keywords
        criteria_keywords = criteria.lower().split()
        scored_urls = []
        
        for url in urls:
            score = 0
            url_lower = url.lower()
            
            # Score based on URL content matching criteria
            for keyword in criteria_keywords:
                if keyword in url_lower:
                    score += 2  # URL contains criteria keyword
            
            # Common relevance patterns
            if 'github.com' in url_lower and any(k in criteria.lower() for k in ['documentation', 'official', 'source']):
                score += 3
            elif 'docs.' in url_lower or '/docs' in url_lower:
                score += 2
            elif 'releases' in url_lower and 'version' in criteria.lower():
                score += 3
            
            scored_urls.append((score, url))
        
        # Sort by score (descending) and take top max_links
        scored_urls.sort(key=lambda x: x[0], reverse=True)
        top_urls = [url for score, url in scored_urls[:max_links] if score > 0]
        
        if not top_urls:
            # If no high-scoring URLs, return the top original URLs
            top_urls = urls[:max_links]
        
        result = [f"Relevant links filtered by criteria '{criteria}':"]
        for i, url in enumerate(top_urls, 1):
            result.append(f"{i}. {url}")
        
        result.append(f"\nRecommendation: Use read_webpage on these URLs to get detailed content.")
        
        return "\n".join(result)

    # Read webpage tool - extract content from a specific URL
    read_webpage_tool = {
        "name": "read_webpage",
        "description": (
            "Read and extract clean, readable content from a specific webpage URL. "
            "Use this when you have a specific URL and want to get its full text "
            "content, or after getting search results to read specific pages. "
            "Handles JavaScript-rendered content and extracts main text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read"},
                "extract_links": {"type": "boolean", "description": "Legacy parameter - link extraction now uses LLM-based analysis of content", "default": True},
                "force_refresh": {"type": "boolean", "description": "Force fresh fetch, bypassing cache", "default": False},
            },
            "required": ["url"],
        },
        "category": "web",
        "enabled": True,
    }

    async def read_webpage_handler(url: str, extract_links: bool = True, force_refresh: bool = False) -> str:
        page = await web_browser.read_webpage(url, extract_links=extract_links, force_refresh=force_refresh)
        if not page:
            return f"Failed to read webpage: {url}"

        formatted = [f"Webpage Content: {page.title}"]
        formatted.append(f"URL: {page.url}")

        if page.metadata.get("description"):
            formatted.append(f"Description: {page.metadata['description']}")

        formatted.append(f"\nContent ({len(page.content)} characters):")
        # Provide full content since we now have intelligent batching that can handle large content
        formatted.append(page.content)

        # Don't pre-extract and list links - let LLM analyze full content for relevant links with context
        # This allows LLM to see "Check our documentation at README-usage.md" and extract it intelligently
        
        # Note: We disable mechanical link extraction since we use LLM-based intelligent extraction

        return "\n".join(formatted)

    # Archive management tool
    archive_stats_tool = {
        "name": "web_archive_stats",
        "description": (
            "Get statistics about the web page archive, including number of cached pages, "
            "storage size, and freshness information. Useful for monitoring cache performance."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "category": "web",
        "enabled": True,
    }

    def archive_stats_handler() -> str:
        stats = web_browser.get_archive_stats()
        
        formatted = ["📊 Web Archive Statistics:"]
        formatted.append(f"📁 Cache directory: {stats['cache_dir']}")
        formatted.append(f"📄 Total cached pages: {stats['total_cached_pages']}")
        formatted.append(f"✅ Fresh pages: {stats['fresh_pages']}")
        formatted.append(f"⏰ Expired pages: {stats['expired_pages']}")
        formatted.append(f"💾 Total cache size: {stats['total_size_mb']} MB")
        
        if stats['total_cached_pages'] > 0:
            cache_hit_potential = (stats['fresh_pages'] / stats['total_cached_pages']) * 100
            formatted.append(f"🎯 Cache efficiency: {cache_hit_potential:.1f}%")
            
        if stats['expired_pages'] > 0:
            formatted.append("\n💡 Tip: Expired pages will be automatically cleaned up on next archive access")
            
        return "\n".join(formatted)

    # Register all tools
    tools.extend(
        [
            (search_web_tool, cast(Callable[..., Any], search_web_handler)),
            (find_relevant_links_tool, cast(Callable[..., Any], find_relevant_links_handler)),
            (read_webpage_tool, cast(Callable[..., Any], read_webpage_handler)),
            (archive_stats_tool, cast(Callable[..., Any], archive_stats_handler)),
        ]
    )

    return tools
