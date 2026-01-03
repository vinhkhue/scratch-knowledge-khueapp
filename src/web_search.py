import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearch:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, max_results: int = 5) -> str:
        """
        Perform a web search and return a summary string.
        """
        try:
            results = self.ddgs.text(query, max_results=max_results)
            if not results:
                return ""
            
            summary_parts = []
            for i, r in enumerate(results):
                summary_parts.append(f"Result {i+1}:")
                summary_parts.append(f"Title: {r.get('title', 'No Title')}")
                summary_parts.append(f"Content: {r.get('body', '')}")
                summary_parts.append(f"Source: {r.get('href', '')}")
                summary_parts.append("---")
            
            return "\n".join(summary_parts)
        except Exception as e:
            logger.error(f"Web Search Error: {e}")
            return ""
