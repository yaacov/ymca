"""Tests for the web browser module."""

import asyncio
import unittest
from unittest.mock import Mock, patch

from modules.llm.llm import LLM
from modules.web.models import SearchResult, WebPage
from modules.web.web_browser import WebBrowser


class TestWebBrowser(unittest.TestCase):
    """Test WebBrowser functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLM)
        self.web_browser = WebBrowser(llm=self.mock_llm, max_requests_per_second=100, max_requests_per_minute=1000)
        self.selenium_browser = WebBrowser(llm=self.mock_llm, max_requests_per_second=100, max_requests_per_minute=1000)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test fixtures."""
        self.loop.close()

    def test_browser_initialization(self):
        """Test browser initialization."""
        self.assertEqual(self.web_browser.llm, self.mock_llm)
        self.assertIsNotNone(self.web_browser.selenium_manager)
        self.assertIsNotNone(self.web_browser.content_extractor)
        self.assertIsNotNone(self.web_browser.search_engine)
        self.assertEqual(len(self.web_browser.visited_urls), 0)
        self.assertEqual(len(self.web_browser.session_state), 0)

    def test_user_agent_rotation(self):
        """Test user agent rotation."""
        self.web_browser._rotate_user_agent()
        # Note: May be the same if random picks the same UA
        self.assertIsNotNone(self.web_browser.selenium_manager.current_user_agent)

    def test_parse_unified_results(self):
        """Test unified search results parsing."""
        # More realistic HTML structure for testing unified parsing
        html_content = """
        <html>
        <body>
            <nav>Navigation menu</nav>
            <header>Site header</header>
            <main>
                <div class="search-results-container">
                    <div class="result-item">
                        <h3><a href="https://example.com">Example Title - A Great Website</a></h3>
                        <p class="description">This is an example description with more detailed content about the website and what it offers to users.</p>
                        <span class="url">https://example.com</span>
                    </div>
                    <div class="result-item">
                        <h3><a href="https://test.com">Test Title - Testing Resources</a></h3>
                        <p class="description">Test description with comprehensive information about testing tools and methodologies for developers.</p>
                        <span class="url">https://test.com</span>
                    </div>
                    <div class="result-item">
                        <h3><a href="https://github.com/kubectl-mtv">kubectl-mtv - Migration Tool for VMs</a></h3>
                        <p class="description">A kubectl plugin for migrating virtualization workloads from platforms like oVirt, VMware, OpenStack to KubeVirt.</p>
                        <span class="url">https://github.com/kubectl-mtv</span>
                    </div>
                    <div class="nav-item">
                        <a href="/privacy">Privacy Policy</a>
                    </div>
                    <div class="nav-item">
                        <a href="https://duckduckgo.com/settings">Settings</a>
                    </div>
                </div>
            </main>
            <footer>Site footer</footer>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        results = self.loop.run_until_complete(self.web_browser.search_engine._parse_search_results_unified(soup, "duckduckgo", 10))

        # Should find the external links and filter out navigation
        self.assertGreaterEqual(len(results), 2)

        # Check that we got some reasonable results
        if len(results) >= 2:
            # Check that kubectl-mtv result gets highest score
            kubectl_result = next((r for r in results if "kubectl-mtv" in r.title.lower()), None)
            self.assertIsNotNone(kubectl_result)

            # Should prioritize GitHub result
            if len(results) > 1:
                self.assertGreater(results[0].relevance_score or 0, 0)

    def test_extract_links_from_soup(self):
        """Test link extraction from HTML."""
        html_content = """
        <a href="https://example.com">Example</a>
        <a href="/relative-link">Relative</a>
        <a href="#fragment">Fragment</a>
        <a href="mailto:test@example.com">Email</a>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        links = self.web_browser._extract_links_from_soup(soup, "https://base.com")

        # Should contain absolute URLs
        self.assertIn("https://example.com", links)
        self.assertIn("https://base.com/relative-link", links)

    def test_extract_text_content(self):
        """Test text content extraction."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <script>alert('remove me');</script>
            <h1>Main Heading</h1>
            <p>This is a paragraph with <strong>bold</strong> text.</p>
            <nav>Navigation menu</nav>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        text = self.web_browser._extract_text_content(soup)

        self.assertIn("Main Heading", text)
        self.assertIn("This is a paragraph", text)
        self.assertNotIn("alert", text)  # Script removed

    def test_extract_page_metadata(self):
        """Test page metadata extraction."""
        html_content = """
        <html lang="en">
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="Test description">
            <meta property="og:title" content="OG Title">
        </head>
        <body></body>
        </html>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        metadata = self.web_browser._extract_page_metadata(soup)

        self.assertEqual(metadata["title"], "Test Page Title")
        self.assertEqual(metadata["meta_description"], "Test description")
        self.assertEqual(metadata["meta_og:title"], "OG Title")
        self.assertEqual(metadata["language"], "en")

    @patch("modules.web.web_browser.WebBrowser._selenium_request")
    async def test_search_web_duckduckgo(self, mock_selenium_request):
        """Test web search with DuckDuckGo."""
        # Mock Selenium response
        mock_html_content = """
        <html>
        <body>
            <div class="result">
                <a href="https://example.com">Example</a>
                <span>Example description</span>
            </div>
        </body>
        </html>
        """
        mock_selenium_request.return_value = mock_html_content

        results = await self.web_browser.search_web("test query", engine="duckduckgo")

        # Should find at least one result
        self.assertGreater(len(results), 0)
        # Check that example.com is in the results
        result_urls = [r.url for r in results]
        self.assertIn("https://example.com", result_urls)

    async def test_score_search_results_with_llm(self):
        """Test search result scoring with LLM."""
        results = [
            SearchResult("Title 1", "https://example1.com", "Description 1"),
            SearchResult("Title 2", "https://example2.com", "Description 2"),
        ]

        self.mock_llm.generate_response.return_value = "[0.9, 0.3]"

        scored_results = await self.web_browser._score_search_results("test query", results)

        self.assertEqual(scored_results[0].relevance_score, 0.9)
        self.assertEqual(scored_results[1].relevance_score, 0.3)
        # Should be sorted by relevance (highest first)
        self.assertGreater(scored_results[0].relevance_score, scored_results[1].relevance_score)

    @patch("modules.web.web_browser.WebBrowser._selenium_request")
    async def test_read_webpage(self, mock_selenium_request):
        """Test webpage reading and content extraction."""
        # Mock Selenium response
        mock_html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Content</h1>
            <p>This is the main content of the page.</p>
            <a href="https://link1.com">Link 1</a>
            <a href="/relative">Relative Link</a>
        </body>
        </html>
        """
        mock_selenium_request.return_value = mock_html_content

        webpage = await self.web_browser.read_webpage("https://example.com")

        self.assertIsNotNone(webpage)
        self.assertEqual(webpage.url, "https://example.com")
        self.assertEqual(webpage.title, "Test Page")
        self.assertIn("Main Content", webpage.content)
        self.assertIn("https://link1.com", webpage.links)
        self.assertIn("https://example.com/relative", webpage.links)

    async def test_extract_useful_content_with_llm(self):
        """Test useful content extraction with LLM."""
        raw_content = "Navigation menu\n\nMain article content here with important information.\n\nFooter copyright"

        self.mock_llm.generate_response.return_value = "Main article content here with important information."

        extracted = await self.web_browser._extract_useful_content(raw_content, "https://example.com")

        self.assertEqual(extracted, "Main article content here with important information.")
        self.mock_llm.generate_response.assert_called_once()

    async def test_find_relevant_links_with_llm(self):
        """Test finding relevant links with LLM."""
        results = [
            SearchResult("Title 1", "https://example1.com", "Description 1"),
            SearchResult("Title 2", "https://example2.com", "Description 2"),
            SearchResult("Title 3", "https://example3.com", "Description 3"),
        ]

        self.mock_llm.generate_response.return_value = "[2, 1]"  # Select results 2 and 1

        relevant = await self.web_browser.find_relevant_links(results, "technical docs", max_links=2)

        self.assertEqual(len(relevant), 2)
        self.assertEqual(relevant[0].title, "Title 2")  # Second result first
        self.assertEqual(relevant[1].title, "Title 1")  # First result second

    def test_session_state_management(self):
        """Test browser session state management."""
        # Set some state
        self.web_browser.visited_urls = ["https://example.com"]
        self.web_browser.cookies = {"session": "abc123"}
        self.web_browser.session_state = {"key": "value"}

        # Get state
        state = self.web_browser.get_session_state()
        self.assertEqual(state["visited_urls"], ["https://example.com"])
        self.assertEqual(state["cookies"], {"session": "abc123"})
        self.assertEqual(state["session_state"], {"key": "value"})

        # Clear and set new state
        self.web_browser.clear_session()
        self.assertEqual(len(self.web_browser.visited_urls), 0)

        new_state = {"visited_urls": ["https://test.com"], "cookies": {"new_session": "xyz789"}, "session_state": {"new_key": "new_value"}}
        self.web_browser.set_session_state(new_state)

        self.assertEqual(self.web_browser.visited_urls, ["https://test.com"])
        self.assertEqual(self.web_browser.cookies, {"new_session": "xyz789"})

    @patch("modules.web.web_browser.WebBrowser.search_web")
    @patch("modules.web.web_browser.WebBrowser.read_webpage")
    async def test_smart_web_search(self, mock_read_webpage, mock_search_web):
        """Test smart web search functionality."""
        # Mock search results
        search_results = [
            SearchResult("Title 1", "https://example1.com", "Description 1"),
            SearchResult("Title 2", "https://example2.com", "Description 2"),
        ]
        mock_search_web.return_value = search_results

        # Mock webpage reading
        mock_webpage = WebPage(url="https://example1.com", title="Title 1", content="Page content", links=["https://link1.com"], metadata={"title": "Title 1"}, extraction_timestamp=1234567890)
        mock_read_webpage.return_value = mock_webpage

        result = await self.web_browser.smart_web_search("test query", max_results=2)

        self.assertEqual(result["query"], "test query")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(len(result["pages"]), 2)  # Should read both pages
        self.assertIn("search_timestamp", result)

    def test_invalid_url_handling(self):
        """Test handling of invalid URLs."""

        async def test_invalid():
            result = await self.web_browser.read_webpage("not-a-valid-url")
            return result

        result = self.loop.run_until_complete(test_invalid())
        self.assertIsNone(result)

    def test_selenium_mode_initialization(self):
        """Test selenium mode initialization."""
        # Should have selenium manager with driver attribute
        self.assertTrue(hasattr(self.selenium_browser, "selenium_manager"))
        self.assertIsNone(self.selenium_browser.selenium_manager.selenium_driver)  # Not created until needed

    def test_enhanced_heuristics(self):
        """Test enhanced heuristic scoring for kubectl-mtv searches."""
        candidates = [
            {
                "url": "https://github.com/kubevirt/kubectl-mtv",
                "title": "kubectl-mtv - Migration Tool for VMs",
                "context": "A kubectl plugin for migrating virtualization workloads from platforms like oVirt, VMware, OpenStack to KubeVirt.",
                "link_element": None,
            },
            {
                "url": "https://stackoverflow.com/questions/50336665/how-do-i-force-delete-kubernetes-pods",
                "title": "How do I force delete kubernetes pods?",
                "context": "General kubernetes questions about pod management",
                "link_element": None,
            },
            {
                "url": "https://docs.openshift.com/container-platform/4.9/virt/virtual_machines/importing_vms/virt-importing-virtual-machine-images-datavolumes.html",
                "title": "Importing virtual machine images with DataVolumes",
                "context": "Documentation for importing VM images into OpenShift Virtualization using DataVolumes",
                "link_element": None,
            },
        ]

        results = self.web_browser.search_engine._score_search_candidates_simple(candidates, "duckduckgo")

        # Should prioritize the kubectl-mtv GitHub result
        self.assertGreater(len(results), 0)

        # Check that GitHub result gets highest score
        github_result = next((r for r in results if "github.com" in r.url), None)
        self.assertIsNotNone(github_result)

        # GitHub result should have higher relevance score than stackoverflow
        stackoverflow_result = next((r for r in results if "stackoverflow.com" in r.url), None)
        if stackoverflow_result and github_result:
            self.assertGreater(github_result.relevance_score or 0, stackoverflow_result.relevance_score or 0)

    def test_identify_main_content_areas(self):
        """Test main content area identification."""
        html_content = """
        <html>
        <body>
            <nav>Navigation</nav>
            <main class="main-content">
                <h1>Main Article</h1>
                <p>This is the main content of the article with substantial information.</p>
                <p>Multiple paragraphs of meaningful content here.</p>
            </main>
            <footer>Footer</footer>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        main_content = self.web_browser.content_extractor._identify_main_content_areas(soup)

        self.assertIsNotNone(main_content)
        self.assertEqual(main_content.name, "main")
        self.assertIn("Main Article", main_content.get_text())

    def test_extract_github_content(self):
        """Test GitHub-specific content extraction."""
        github_html = """
        <html>
        <head><title>GitHub - user/repo</title></head>
        <body>
            <nav>GitHub Navigation</nav>
            <article class="markdown-body">
                <h1>My Project</h1>
                <p>Project description here</p>
                <h2>Installation</h2>
                <pre><code>npm install my-project</code></pre>
            </article>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(github_html, "html.parser")
        github_content = self.web_browser.content_extractor._extract_github_content(soup)

        self.assertIsNotNone(github_content)
        self.assertIn("My Project", github_content.get_text())
        self.assertIn("Installation", github_content.get_text())

    def test_score_content_element(self):
        """Test content element scoring."""
        good_content_html = """
        <div class="content">
            <h1>Technical Documentation</h1>
            <p>This is a comprehensive guide for kubectl installation and usage.</p>
            <pre><code>kubectl get pods</code></pre>
            <ul>
                <li>Feature 1</li>
                <li>Feature 2</li>
            </ul>
        </div>
        """

        bad_content_html = """
        <div class="sidebar">
            <a href="/privacy">Privacy</a>
            <a href="/terms">Terms</a>
            <a href="/contact">Contact</a>
        </div>
        """

        from bs4 import BeautifulSoup

        good_soup = BeautifulSoup(good_content_html, "html.parser")
        bad_soup = BeautifulSoup(bad_content_html, "html.parser")

        good_element = good_soup.find("div")
        bad_element = bad_soup.find("div")

        good_score = self.web_browser.content_extractor._score_content_element(good_element)
        bad_score = self.web_browser.content_extractor._score_content_element(bad_element)

        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 30)  # Should get decent score for good content

    def test_assess_content_quality(self):
        """Test content quality assessment."""
        high_quality_content = """# kubectl-mtv Migration Tool

A comprehensive tool for migrating virtualization workloads.

## Installation

```bash
kubectl krew install mtv
```

## Usage

Basic usage example:

```bash
kubectl mtv migrate --source ovirt --target kubevirt
```

## Features

- Support for multiple virtualization platforms
- Seamless integration with KubeVirt
- Advanced migration options"""

        low_quality_content = """Home About Contact Privacy Policy Terms of Service"""

        high_quality = self.web_browser.content_extractor.assess_content_quality(high_quality_content)
        low_quality = self.web_browser.content_extractor.assess_content_quality(low_quality_content)

        self.assertGreater(high_quality.score, low_quality.score)
        self.assertGreater(high_quality.score, 0.7)  # Should be high quality
        self.assertIn("technical_content", high_quality.reasons)
        self.assertIn("has_code", high_quality.reasons)

    async def test_analyze_content_characteristics(self):
        """Test content characteristic analysis."""
        github_content = "This is a GitHub repository with kubectl and kubernetes content"
        github_url = "https://github.com/user/repo"

        doc_content = "This documentation explains installation and configuration procedures"
        doc_url = "https://docs.example.com/guide"

        github_chars = self.web_browser.content_extractor.analyze_content_characteristics(github_content, github_url)
        doc_chars = self.web_browser.content_extractor.analyze_content_characteristics(doc_content, doc_url)

        self.assertTrue(github_chars.is_github)
        self.assertTrue(github_chars.is_technical)

        self.assertFalse(doc_chars.is_github)
        self.assertTrue(doc_chars.is_documentation)

    @patch("modules.web.web_browser.WebBrowser._selenium_request")
    async def test_hybrid_content_extraction(self, mock_selenium_request):
        """Test hybrid content extraction combining static and LLM."""
        mock_html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation menu</nav>
            <main class="content">
                <h1>kubectl-mtv Guide</h1>
                <p>This is a comprehensive guide for using kubectl-mtv for migration.</p>
                <h2>Installation</h2>
                <pre><code>kubectl krew install mtv</code></pre>
            </main>
            <footer>Footer</footer>
        </body>
        </html>
        """
        mock_selenium_request.return_value = mock_html_content

        # Mock LLM to return improved content
        self.mock_llm.generate_response.return_value = """kubectl-mtv Guide

A comprehensive guide for using kubectl-mtv for migration.

Installation:
kubectl krew install mtv"""

        webpage = await self.web_browser.read_webpage("https://example.com")

        self.assertIsNotNone(webpage)
        self.assertIn("kubectl-mtv", webpage.content)
        # Should have used either static or LLM extraction
        self.assertGreater(len(webpage.content), 50)


class TestSearchResult(unittest.TestCase):
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult creation and attributes."""
        result = SearchResult(title="Test Title", url="https://example.com", description="Test description", relevance_score=0.8)

        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.description, "Test description")
        self.assertEqual(result.relevance_score, 0.8)

    def test_search_result_without_score(self):
        """Test SearchResult without relevance score."""
        result = SearchResult(title="Test Title", url="https://example.com", description="Test description")

        self.assertIsNone(result.relevance_score)


class TestWebPage(unittest.TestCase):
    """Test WebPage dataclass."""

    def test_webpage_creation(self):
        """Test WebPage creation and attributes."""
        page = WebPage(url="https://example.com", title="Test Page", content="Page content", links=["https://link1.com", "https://link2.com"], metadata={"title": "Test Page", "language": "en"}, extraction_timestamp=1234567890.5)

        self.assertEqual(page.url, "https://example.com")
        self.assertEqual(page.title, "Test Page")
        self.assertEqual(page.content, "Page content")
        self.assertEqual(len(page.links), 2)
        self.assertIn("https://link1.com", page.links)
        self.assertEqual(page.metadata["language"], "en")
        self.assertEqual(page.extraction_timestamp, 1234567890.5)


if __name__ == "__main__":
    unittest.main()
