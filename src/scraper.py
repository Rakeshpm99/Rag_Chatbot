import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document

async def ingest_urls(urls: List[str]) -> List[Document]:
    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, wait_for="body")
    
    docs = []
    print(f"Starting sequential scraping for {len(urls)} targets (one by one)...")
    
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=run_cfg)
            if result.success:
                docs.append(Document(page_content=result.markdown, metadata={"source": result.url}))
                print(f"Scraped: {result.url}")
            else:
                print(f"Failed: {result.url}")
    return docs
