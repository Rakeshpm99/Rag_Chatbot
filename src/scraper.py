import asyncio
import nest_asyncio
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document

nest_asyncio.apply()


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


async def scrape_single_url(url: str) -> list:
    """
    Optimized scraper:
    - magic=True preserves HTML tables as Markdown
    - stealth mode bypasses bot detection
    - word_count_threshold skips empty pages
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    browser_cfg = BrowserConfig(
        headless=True,
        enable_stealth=True
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for="body",
        magic=True,
        page_timeout=60000,
        word_count_threshold=10
    )

    docs = []
    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            print(f"🌐 Scraping: {url}...")
            result = await crawler.arun(url=url, config=run_cfg)
            if result.success and result.markdown:
                docs.append(Document(
                    page_content=result.markdown,
                    metadata={"source": result.url}
                ))
                print(f"✅ Scraped! ({len(result.markdown)} chars)")
            else:
                print(f"❌ Failed: {result.error_message}")
    except Exception as e:
        print(f"🚨 Error: {e}")

    return docs


def scrape_url(url: str) -> list:
    """Sync wrapper for internal use."""
    return asyncio.get_event_loop().run_until_complete(
        scrape_single_url(url)
    )
