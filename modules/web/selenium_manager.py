"""Selenium WebDriver management for web browser automation."""

import asyncio
import logging
import random
from typing import Optional, Union

from fake_useragent import UserAgent
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeWebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxWebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class SeleniumManager:
    """Manages Selenium WebDriver instances for web browser automation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Selenium manager.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("ymca.web.selenium")

        # User agent rotation
        self.ua = UserAgent()
        self.current_user_agent = self.ua.random

        # Selenium driver (created on demand)
        self.selenium_driver: Optional[Union[ChromeWebDriver, FirefoxWebDriver]] = None

        self.logger.info("Selenium manager initialized")

    def rotate_user_agent(self) -> None:
        """Rotate user agent to avoid detection."""
        self.current_user_agent = self.ua.random
        self.logger.debug(f"Rotated user agent: {self.current_user_agent[:50]}...")

    def get_driver(self) -> Optional[Union[ChromeWebDriver, FirefoxWebDriver]]:
        """Get selenium driver, creating it if needed."""
        if self.selenium_driver is None:
            self.selenium_driver = self._create_driver()
        return self.selenium_driver

    def _create_driver(self) -> Optional[Union[ChromeWebDriver, FirefoxWebDriver]]:
        """Create a new Selenium WebDriver instance."""
        try:
            # Try Chrome first (more reliable)
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920x1080")
            chrome_options.add_argument(f"--user-agent={self.current_user_agent}")

            # Additional stealth options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)

            chrome_driver = webdriver.Chrome(options=chrome_options)
            chrome_driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.logger.debug("Initialized Chrome WebDriver")
            return chrome_driver

        except WebDriverException:
            try:
                # Fallback to Firefox
                firefox_options = FirefoxOptions()
                firefox_options.add_argument("--headless")
                firefox_options.set_preference("general.useragent.override", self.current_user_agent)

                firefox_driver = webdriver.Firefox(options=firefox_options)
                self.logger.debug("Initialized Firefox WebDriver")
                return firefox_driver

            except WebDriverException as e:
                self.logger.error(f"Failed to initialize WebDriver: {e}")
                return None

    async def request_page(self, url: str, timeout: int = 30) -> Optional[str]:
        """Make a request using selenium for more robust scraping."""
        driver = self.get_driver()
        if not driver:
            return None

        try:
            # Add random delay to appear human-like
            await asyncio.sleep(random.uniform(0.5, 2.0))

            driver.get(url)

            # Wait for the page to load
            try:
                WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            except TimeoutException:
                self.logger.warning(f"Page load timeout for {url}")

            # Additional wait for dynamic content
            await asyncio.sleep(random.uniform(1.0, 3.0))

            page_source = str(driver.page_source)
            self.logger.debug(f"Successfully fetched page source for {url}")
            return page_source

        except Exception as e:
            self.logger.error(f"Selenium request failed for {url}: {e}")
            return None

    def close_driver(self) -> None:
        """Close selenium driver if it exists."""
        if self.selenium_driver:
            try:
                self.selenium_driver.quit()
                self.selenium_driver = None
                self.logger.debug("Closed WebDriver")
            except Exception as e:
                self.logger.warning(f"Error closing WebDriver: {e}")

    def refresh_driver(self) -> None:
        """Refresh the WebDriver by closing and recreating it."""
        self.close_driver()
        self.rotate_user_agent()
        self.logger.info("Refreshed WebDriver with new user agent")

    def is_driver_available(self) -> bool:
        """Check if a WebDriver is available and working."""
        if not self.selenium_driver:
            return False

        try:
            # Try a simple operation to check if driver is working
            self.selenium_driver.current_url
            return True
        except Exception:
            return False

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        try:
            self.close_driver()
        except Exception:
            pass  # Ignore errors during cleanup
