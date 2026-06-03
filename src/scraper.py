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
            result = await crawler.arun(url=url, config=run_cfg)
            
            if result.success and result.markdown and len(result.markdown.strip()) > 0:
                docs.append(Document(page_content=result.markdown, metadata={"source": result.url}))
            else:
                print(f"Scraper failed or returned no text. Error: {result.error_message}")
                
    except Exception as e:
        print(f"Connection error: {e}")


    return docs
