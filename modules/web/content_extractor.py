"""Content extraction functionality for web pages."""

import logging
import re
from typing import Any, Dict, List, Optional
import html2text
from bs4 import BeautifulSoup, Tag

from ..llm.llm import LLM
from .models import ContentCharacteristics, ContentQuality


class ContentExtractor:
    """Handles extraction of content from web pages using various strategies."""

    def __init__(self, llm: LLM, logger: Optional[logging.Logger] = None):
        """
        Initialize content extractor.

        Args:
            llm: LLM instance for intelligent processing
            logger: Logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger("ymca.web.content")

        # HTML to text converter
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.h.ignore_images = True
        self.h.body_width = 0  # Don't wrap text

    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML with enhanced content detection."""
        # Enhanced content detection - identify main content areas
        main_content = self._identify_main_content_areas(soup)

        if main_content:
            # Use identified main content
            text_content = self.h.handle(str(main_content))
        else:
            # Fallback to original method
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Get text using html2text for better formatting
            html_content = str(soup)
            text_content = self.h.handle(html_content)

        # Clean up extra whitespace
        text_content = re.sub(r"\n\s*\n", "\n\n", text_content)
        text_content = text_content.strip()

        return text_content

    def _identify_main_content_areas(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Identify main content areas using semantic HTML and heuristics."""
        # Strategy 1: Look for semantic HTML5 elements
        main_selectors = ["main", "article", '[role="main"]', ".main-content", ".content", ".post-content", ".article-content", ".entry-content", "#main-content", "#content", "#post-content"]

        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                # Return the first and largest element
                largest_element = max(elements, key=lambda x: len(x.get_text()) if x else 0)
                if len(largest_element.get_text().strip()) > 50:  # Reasonable content size
                    self.logger.debug(f"Found main content using selector: {selector}")
                    return largest_element

        # Strategy 2: GitHub-specific content detection
        title_tag = soup.find("title")
        if title_tag and "github.com" in title_tag.get_text().lower():
            github_content = self._extract_github_content(soup)
            if github_content:
                return github_content

        # Strategy 3: Heuristic-based content detection
        return self._extract_content_by_heuristics(soup)

    def _extract_github_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Extract GitHub-specific content areas."""
        # GitHub README content - Updated selectors for current GitHub structure
        readme_selectors = [
            "article.markdown-body", 
            ".markdown-body", 
            "#readme", 
            '[data-testid="readme"]',
            '[data-pjax="#repo-content-pjax-container"] .markdown-body',
            '.Box-body .markdown-body',
            '.readme .markdown-body',
            'div[data-target="readme-toc.content"]'
        ]

        for selector in readme_selectors:
            element = soup.select_one(selector)
            if element and len(element.get_text().strip()) > 30:
                self.logger.debug(f"Found GitHub README content using: {selector}")
                return element

        # Try broader GitHub content areas with better filtering
        broader_selectors = [
            ".repository-content",
            "#repo-content-turbo-frame", 
            "[data-target='readme-toc.content']",
            ".Box-body",
            "main[data-pjax-container]"
        ]

        for selector in broader_selectors:
            element = soup.select_one(selector)
            if element and len(element.get_text().strip()) > 100:
                self.logger.debug(f"Found broader GitHub content using: {selector}")
                
                # Clone the element to avoid modifying the original
                element_copy = element.extract() if hasattr(element, 'extract') else element
                
                # Remove navigation, header, and other non-content elements
                for unwanted in element_copy.select("nav, .subnav, .pagehead, .UnderlineNav, .Header, .footer, .js-repo-nav, .Layout-sidebar, .BorderGrid-cell--secondary"):
                    unwanted.decompose()
                
                # Remove GitHub-specific UI elements
                for ui_elem in element_copy.select('[data-view-component="true"], .octicon, .btn, .Button, .flash, .toast'):
                    ui_elem.decompose()
                
                # If still has substantial content, return it
                cleaned_text = element_copy.get_text().strip()
                if len(cleaned_text) > 50:
                    return element_copy

        # Final fallback - look for any content with README indicators
        readme_indicators = soup.find_all(text=lambda text: text and any(word in str(text).lower() 
                                                                         for word in ['readme', 'installation', 'usage', 'getting started']))
        for indicator in readme_indicators:
            parent = indicator.parent
            if parent and isinstance(parent, Tag):
                # Walk up the tree to find substantial content
                for _ in range(5):
                    if parent and len(parent.get_text().strip()) > 200:
                        self.logger.debug("Found README content using text indicator search")
                        return parent
                    parent = parent.parent

        self.logger.warning("No GitHub content found with any selectors")
        return None

    def _extract_content_by_heuristics(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Extract content using heuristic analysis."""
        # Remove noise elements first
        noise_selectors = ["script", "style", "nav", "header", "footer", ".sidebar", ".navigation", ".menu", ".ads", ".comments", ".social-media", ".related-posts"]

        for selector in noise_selectors:
            for element in soup.select(selector):
                if isinstance(element, Tag):
                    element.decompose()

        # Find content-rich elements
        content_candidates = []

        # Look for divs and sections with substantial text content
        all_elements = soup.find_all(["div", "section", "article"])
        elements = [elem for elem in all_elements if isinstance(elem, Tag)]

        for element in elements:
            text_content = element.get_text().strip()
            if len(text_content) < 30:  # Skip short content
                continue

            # Score based on content characteristics
            score = self._score_content_element(element)
            if score > 0:
                content_candidates.append((element, score))

        if content_candidates:
            # Return highest scoring element
            content_candidates.sort(key=lambda x: x[1], reverse=True)
            best_element = content_candidates[0][0]
            self.logger.debug(f"Found content using heuristics, score: {content_candidates[0][1]}")
            return best_element

        return None

    def _score_content_element(self, element: Tag) -> float:
        """Score an HTML element based on content quality indicators."""
        text_content = element.get_text()
        text_length = len(text_content)

        if text_length < 30:
            return 0

        score = text_length * 0.01  # Base score from text length

        # Positive indicators
        if element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            score += 20  # Has structured content

        if element.find_all(["code", "pre"]):
            score += 15  # Technical content bonus

        if element.find_all(["ul", "ol", "li"]):
            score += 10  # Lists indicate structured content

        # Content density (text vs markup ratio)
        html_length = len(str(element))
        if html_length > 0:
            density_ratio = text_length / html_length
            score += density_ratio * 30  # Higher density is better

        # Negative indicators
        if element.find_all(["nav", "footer", "header"]):
            score -= 30  # Structural elements

        if len(element.find_all("a")) > text_length / 50:  # Too many links
            score -= 20

        # Class and ID indicators
        class_attr = element.get("class")
        if isinstance(class_attr, list):
            class_names = " ".join(class_attr).lower()
        elif class_attr:
            class_names = str(class_attr).lower()
        else:
            class_names = ""
        element_id_attr = element.get("id", "")
        element_id = str(element_id_attr).lower() if element_id_attr else ""

        positive_indicators = ["content", "main", "article", "post", "entry", "body", "text"]
        negative_indicators = ["sidebar", "nav", "menu", "footer", "header", "ad", "social"]

        for indicator in positive_indicators:
            if indicator in class_names or indicator in element_id:
                score += 15

        for indicator in negative_indicators:
            if indicator in class_names or indicator in element_id:
                score -= 25

        return max(0, score)


    def extract_page_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page metadata."""
        metadata = {}

        # Title
        title_tag = soup.find("title")
        metadata["title"] = title_tag.get_text().strip() if title_tag else ""

        # Meta tags
        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            if isinstance(meta, Tag):
                name = meta.get("name") or meta.get("property")
                content = meta.get("content")
                if name and content:
                    metadata[f"meta_{name}"] = str(content)

        # Language
        html_tag = soup.find("html")
        if html_tag and isinstance(html_tag, Tag):
            lang = html_tag.get("lang", "")
            metadata["language"] = str(lang) if lang else ""

        return metadata

    async def hybrid_content_extraction(self, static_content: str, url: str, soup: BeautifulSoup) -> tuple[str, List[str]]:
        """Combine static analysis with LLM extraction for optimal results, also extracting links."""
        self.logger.info("Using hybrid content extraction")

        # Extract links from the raw HTML before any processing
        raw_links = self._extract_raw_links_from_soup(soup, url)
        
        # Get quality metrics for static content
        static_quality = self.assess_content_quality(static_content)

        # If static content is already high quality, use it directly
        if static_quality.is_high_quality:
            self.logger.info(f"Static content quality is high ({static_quality.score:.2f}), using static extraction")
            return static_content, raw_links

        # Try LLM extraction
        llm_content = await self._extract_useful_content(static_content, url)

        # If LLM extraction was successful, compare quality
        if llm_content != static_content:  # LLM extraction was used
            llm_quality = self.assess_content_quality(llm_content)

            self.logger.info(f"Content quality - Static: {static_quality.score:.2f}, LLM: {llm_quality.score:.2f}")

            # Choose the higher quality content
            if llm_quality.score > static_quality.score:
                self.logger.info("Using LLM-extracted content (higher quality)")
                return llm_content, raw_links
            else:
                self.logger.info("Using static-extracted content (higher quality)")
                return static_content, raw_links

        # Fallback to static content
        self.logger.info("Using static-extracted content (LLM extraction unchanged)")
        return static_content, raw_links

    async def _extract_useful_content(self, raw_content: str, url: str) -> str:
        """Use LLM to extract the most useful content from raw text with enhanced strategy."""
        try:
            # Only process if content is reasonably long
            if len(raw_content) < 500:
                return raw_content

            # Analyze content characteristics for better processing
            content_info = self.analyze_content_characteristics(raw_content, url)

            # Use different strategies based on content type
            if content_info.is_github:
                return await self._extract_github_content_llm(raw_content, url)
            elif content_info.is_documentation:
                return await self._extract_documentation_content_llm(raw_content, url)
            else:
                return await self._extract_general_content_llm(raw_content, url)

        except Exception as e:
            self.logger.warning(f"Failed to extract useful content with LLM: {e}")
            return raw_content

    def analyze_content_characteristics(self, content: str, url: str) -> ContentCharacteristics:
        """Analyze content characteristics to determine best extraction strategy."""
        content_lower = content.lower()
        url_lower = url.lower()

        return ContentCharacteristics(
            is_github="github.com" in url_lower,
            is_documentation=any(term in url_lower or term in content_lower for term in ["docs.", "documentation", "readme", "tutorial", "guide", "manual"]),
            is_technical=any(term in content_lower for term in ["kubectl", "kubernetes", "api", "installation", "configuration", "command"]),
            has_code=any(term in content_lower for term in ["```", "code", "$ ", "kubectl ", "docker "]),
        )

    async def _extract_github_content_llm(self, raw_content: str, url: str) -> str:
        """Extract GitHub-specific content using LLM."""
        # Truncate for processing, but keep more content for GitHub pages
        content_for_processing = raw_content[:12000] + "..." if len(raw_content) > 12000 else raw_content

        prompt = f"""This is a GitHub repository page. Extract the most important information including:
1. Project description and purpose
2. Key features and capabilities
3. Installation instructions
4. Usage examples
5. Important documentation sections
6. Any technical specifications or requirements

URL: {url}

GitHub page content:
{content_for_processing}

Extracted key information:"""

        extracted = str(self.llm.generate_response(prompt))

        # More lenient threshold for GitHub content
        if len(extracted) > len(raw_content) * 0.05:  # 5% threshold for GitHub
            self.logger.info("Successfully extracted GitHub content using LLM")
            return extracted.strip()
        else:
            self.logger.warning("LLM GitHub extraction too short, using original")
            return raw_content

    async def _extract_documentation_content_llm(self, raw_content: str, url: str) -> str:
        """Extract documentation content using LLM."""
        content_for_processing = raw_content[:10000] + "..." if len(raw_content) > 10000 else raw_content

        prompt = f"""This is a documentation page. Extract the main instructional and informational content, focusing on:
1. Step-by-step procedures
2. Technical explanations
3. Examples and code snippets
4. Configuration details
5. Important notes and warnings

Remove navigation, headers, footers, and sidebar content.

URL: {url}

Documentation content:
{content_for_processing}

Main documentation content:"""

        extracted = str(self.llm.generate_response(prompt))

        if len(extracted) > len(raw_content) * 0.08:  # 8% threshold for docs
            self.logger.info("Successfully extracted documentation content using LLM")
            return extracted.strip()
        else:
            self.logger.warning("LLM documentation extraction too short, using original")
            return raw_content

    async def _extract_general_content_llm(self, raw_content: str, url: str) -> str:
        """Extract general webpage content using LLM."""
        content_for_processing = raw_content[:8000] + "..." if len(raw_content) > 8000 else raw_content

        prompt = f"""Extract the main article or content from this webpage. Focus on:
1. The primary information or message
2. Key details and facts
3. Important explanations
4. Relevant examples

Remove navigation menus, advertisements, footers, sidebars, comments, and other non-essential elements.

URL: {url}

Webpage content:
{content_for_processing}

Main content:"""

        extracted = str(self.llm.generate_response(prompt))

        if len(extracted) > len(raw_content) * 0.1:  # 10% threshold for general content
            self.logger.info("Successfully extracted general content using LLM")
            return extracted.strip()
        else:
            self.logger.warning("LLM general extraction too short, using original")
            return raw_content

    def assess_content_quality(self, content: str) -> ContentQuality:
        """Assess the quality of extracted content."""
        if not content or len(content.strip()) == 0:
            return ContentQuality(score=0.0, reasons=["empty_content"], length=0)

        content = content.strip()
        content_lower = content.lower()

        score = 0.5  # Base score
        reasons = []

        # Length indicators
        if len(content) > 1000:
            score += 0.1
            reasons.append("good_length")
        elif len(content) < 200:
            score -= 0.2
            reasons.append("too_short")

        # Structure indicators
        if any(marker in content for marker in ["\n#", "##", "###", "####"]):
            score += 0.15
            reasons.append("has_headers")

        if content.count("\n\n") > 2:  # Multiple paragraphs
            score += 0.1
            reasons.append("multi_paragraph")

        # Technical content indicators
        if any(term in content_lower for term in ["kubectl", "installation", "usage", "command", "example"]):
            score += 0.15
            reasons.append("technical_content")

        # Code/command indicators
        if any(marker in content for marker in ["```", "$ ", "kubectl ", "docker "]):
            score += 0.1
            reasons.append("has_code")

        # Navigation/boilerplate penalties
        if any(term in content_lower for term in ["privacy policy", "terms of service", "cookie policy"]):
            score -= 0.2
            reasons.append("has_boilerplate")

        # High ratio of links suggests navigation content
        link_count = content_lower.count("http")
        if len(content) > 0 and link_count > len(content) / 100:  # More than 1% links
            score -= 0.15
            reasons.append("too_many_links")

        # GitHub-specific quality indicators
        if "github.com" in content_lower:
            if any(term in content_lower for term in ["readme", "features", "installation", "getting started"]):
                score += 0.2
                reasons.append("github_quality_content")

        return ContentQuality(score=max(0.0, min(1.0, score)), reasons=reasons, length=len(content))  # Clamp between 0 and 1

    def _extract_raw_links_from_soup(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all potentially relevant links from the HTML soup before content processing."""
        from urllib.parse import urljoin, urlparse
        import re
        
        links = []
        base_domain = urlparse(base_url).netloc.lower()
        
        # Find all anchor tags with href attributes
        for link_tag in soup.find_all('a', href=True):
            if not isinstance(link_tag, Tag):
                continue
                
            href = str(link_tag.get('href', '')).strip()
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Make relative URLs absolute
            if href.startswith('/') or not href.startswith('http'):
                absolute_url = urljoin(base_url, href)
            else:
                absolute_url = href
                
            # Filter for relevant links
            if self._is_potentially_relevant_link(absolute_url, link_tag, base_domain):
                links.append(absolute_url)
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(links))
        
        if unique_links:
            self.logger.info(f"🔗 Extracted {len(unique_links)} raw links from HTML before content processing")
            # Log first few links as examples
            for i, link in enumerate(unique_links[:5], 1):
                self.logger.debug(f"   {i}. {link}")
            if len(unique_links) > 5:
                self.logger.debug(f"   ... and {len(unique_links) - 5} more")
        
        return unique_links
    
    def _is_potentially_relevant_link(self, url: str, link_tag: Tag, base_domain: str) -> bool:
        """Check if a link is potentially relevant for documentation or examples."""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            
            # Skip invalid URLs
            if not parsed.netloc or parsed.scheme not in ['http', 'https']:
                return False
            
            # Focus on same domain or related subdomains for GitHub-like sites
            link_domain = parsed.netloc.lower()
            if not (link_domain == base_domain or 
                   link_domain.endswith('.' + base_domain) or 
                   base_domain.endswith('.' + link_domain)):
                # Allow some external documentation sites
                allowed_external = ['docs.', 'wiki.', 'github.io', 'readthedocs.', '.github.com']
                if not any(ext in link_domain for ext in allowed_external):
                    return False
            
            url_lower = url.lower()
            link_text = link_tag.get_text().lower().strip()
            
            # Skip navigation and UI elements
            skip_patterns = [
                '/login', '/signup', '/register', '/logout', '/settings', '/profile',
                '/notifications', '/security', '/billing', '/pricing',
                'javascript:', 'mailto:', '#', '/issues/new', '/pull/', '/compare/',
                'edit', 'delete', '/blame/', '/commits/', '/branches', '/tags',
                'avatar', 'profile', 'follow', 'unfollow', 'star', 'fork', 'watch'
            ]
            
            if any(skip in url_lower for skip in skip_patterns):
                return False
            
            # Prioritize documentation, examples, guides
            relevant_patterns = [
                'docs/', 'doc/', 'documentation', 'wiki/', 'examples/', 'example',
                'tutorial', 'guide', 'getting-started', 'readme', 'installation',
                'quick-start', 'how-to', 'api/', 'reference/', 'manual',
                'setup', 'configuration', 'config', 'usage', '/blob/main', '/blob/master',
                'releases/', 'changelog', 'migration', 'upgrade'
            ]
            
            # Check URL path and link text for relevant patterns
            has_relevant_pattern = any(pattern in url_lower or pattern in link_text for pattern in relevant_patterns)
            
            # For GitHub repositories, also include main branch files
            if 'github.com' in base_domain:
                github_relevant = any(pattern in url_lower for pattern in [
                    '/tree/', '/blob/', '/raw/', 'readme', '.md', '.txt', '.rst',
                    'examples', 'docs', 'documentation'
                ])
                has_relevant_pattern = has_relevant_pattern or github_relevant
            
            return has_relevant_pattern
            
        except Exception as e:
            self.logger.warning(f"Error checking link relevance for {url}: {e}")
            return False
