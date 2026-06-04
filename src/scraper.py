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


async def _scrape_async(url: str) -> list:
    browser_cfg = BrowserConfig(
        headless=True,
        enable_stealth=True
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for="body",
        magic=True,
        page_timeout=60000
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
                print(f"✅ Scraped successfully!")
            else:
                print(f"❌ Failed: {result.error_message}")

    except Exception as e:
        print(f"🚨 Connection error: {e}")

    return docs


def scrape_url(url: str) -> list:
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    return asyncio.get_event_loop().run_until_complete(
        _scrape_async(url)
    )
