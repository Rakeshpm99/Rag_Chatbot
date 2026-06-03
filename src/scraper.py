import asyncio
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

async def scrape_single_url(url: str):
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL format provided: {url}")

    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, wait_for="body", page_timeout=60000)

    docs = []
    
    # We are removing the try/except here so the raw error bubbles up to the UI
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        
        if result.success and result.markdown and len(result.markdown.strip()) > 0:
            docs.append(Document(page_content=result.markdown, metadata={"source": result.url}))
        else:
            # THIS WILL PRINT THE EXACT REASON IT FAILED ON YOUR SCREEN
            raise Exception(f"Internal Scraper Failure -> Success: {result.success} | Status Code: {result.status_code} | Error: {result.error_message}")

    return docs
