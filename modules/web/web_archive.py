"""Web page archive for caching fetched pages."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any

from .models import WebPage


class WebArchive:
    """Manages cached web pages to avoid redundant fetches."""

    def __init__(self, archive_dir: str = "data/web_archive", max_age_hours: int = 24, logger: Optional[logging.Logger] = None):
        """
        Initialize web archive.

        Args:
            archive_dir: Directory to store cached pages
            max_age_hours: Maximum age in hours before a cached page is considered stale
            logger: Logger instance
        """
        self.archive_dir = Path(archive_dir)
        self.max_age_seconds = max_age_hours * 3600
        self.logger = logger or logging.getLogger("ymca.web.archive")
        
        # Create archive directory if it doesn't exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Web archive initialized: {self.archive_dir} (max age: {max_age_hours}h)")

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename using hash."""
        # Create hash of URL for filename
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"{url_hash}.json"

    def _is_fresh(self, cached_page: Dict[str, Any]) -> bool:
        """Check if cached page is still fresh."""
        extraction_time = cached_page.get("extraction_timestamp", 0)
        age_seconds = time.time() - extraction_time
        return age_seconds < self.max_age_seconds

    def get_cached_page(self, url: str) -> Optional[WebPage]:
        """Get cached page if available and fresh."""
        try:
            filename = self._url_to_filename(url)
            cache_path = self.archive_dir / filename
            
            if not cache_path.exists():
                self.logger.debug(f"No cached page for URL: {url}")
                return None
            
            # Load cached data
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if still fresh
            if not self._is_fresh(cached_data):
                self.logger.info(f"Cached page expired for URL: {url}")
                # Clean up expired cache file
                cache_path.unlink()
                return None
            
            # Convert back to WebPage object
            webpage = WebPage(
                url=cached_data["url"],
                title=cached_data["title"],
                content=cached_data["content"],
                links=cached_data["links"],
                metadata=cached_data["metadata"],
                extraction_timestamp=cached_data["extraction_timestamp"]
            )
            
            age_minutes = (time.time() - cached_data["extraction_timestamp"]) / 60
            self.logger.info(f"🗃️  Using cached page for {url} (age: {age_minutes:.1f} min)")
            return webpage
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached page for {url}: {e}")
            return None

    def cache_page(self, webpage: WebPage) -> bool:
        """Cache a webpage for future use."""
        try:
            filename = self._url_to_filename(webpage.url)
            cache_path = self.archive_dir / filename
            
            # Convert WebPage to JSON-serializable dict
            cache_data = {
                "url": webpage.url,
                "title": webpage.title,
                "content": webpage.content,
                "links": webpage.links,
                "metadata": webpage.metadata,
                "extraction_timestamp": webpage.extraction_timestamp,
                "cached_at": time.time()
            }
            
            # Write to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            content_kb = len(webpage.content) / 1024
            self.logger.info(f"📁 Cached webpage: {webpage.url} ({content_kb:.1f} KB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache page {webpage.url}: {e}")
            return False

    def clear_expired(self) -> int:
        """Clear expired cache entries."""
        cleared_count = 0
        try:
            for cache_file in self.archive_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    if not self._is_fresh(cached_data):
                        cache_file.unlink()
                        cleared_count += 1
                        self.logger.debug(f"Cleared expired cache: {cache_file.name}")
                        
                except Exception as e:
                    self.logger.warning(f"Error checking cache file {cache_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
            
        if cleared_count > 0:
            self.logger.info(f"🧹 Cleared {cleared_count} expired cache entries")
            
        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_cached_pages": 0,
            "fresh_pages": 0,
            "expired_pages": 0,
            "total_size_mb": 0,
            "cache_dir": str(self.archive_dir)
        }
        
        try:
            for cache_file in self.archive_dir.glob("*.json"):
                stats["total_cached_pages"] += 1
                stats["total_size_mb"] += cache_file.stat().st_size / (1024 * 1024)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    if self._is_fresh(cached_data):
                        stats["fresh_pages"] += 1
                    else:
                        stats["expired_pages"] += 1
                        
                except Exception:
                    stats["expired_pages"] += 1
                    
        except Exception as e:
            self.logger.error(f"Error calculating cache stats: {e}")
            
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    def remove_url(self, url: str) -> bool:
        """Remove a specific URL from cache."""
        try:
            filename = self._url_to_filename(url)
            cache_path = self.archive_dir / filename
            
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"🗑️  Removed cached page: {url}")
                return True
            else:
                self.logger.debug(f"No cached page to remove: {url}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove cached page {url}: {e}")
            return False
