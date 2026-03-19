import unittest
from importlib import import_module
from unittest.mock import patch

prompts_module = import_module("production_rag_pipeline.prompts")
search_module = import_module("production_rag_pipeline.search")
fetch_module = import_module("production_rag_pipeline.fetch")
rerank_module = import_module("production_rag_pipeline.rerank")
extract_module = import_module("production_rag_pipeline.extract")


class SearchIntegrationTests(unittest.TestCase):
    @patch.object(search_module, "search_ddg")
    @patch.object(search_module, "search_bing")
    def test_search_merges_results_and_enables_news_mode(self, mock_bing, mock_ddg):
        mock_bing.return_value = [
            {
                "title": "Latest AI funding round",
                "url": "https://example.com/story",
                "snippet": "Funding round coverage from Bing.",
            }
        ]
        mock_ddg.return_value = [
            {
                "title": "Latest AI funding round updated",
                "url": "https://example.com/story?ref=ddg",
                "snippet": "Funding round coverage from DDG.",
            }
        ]

        results = search_module.search("latest ai news", num=10, lang="en")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["engines"], ["bing", "ddg"])
        self.assertEqual(results[0]["url"], "https://example.com/story")
        self.assertTrue(mock_ddg.call_args.kwargs["news_mode"])


class PromptIntegrationTests(unittest.TestCase):
    @patch.object(extract_module, "filter_low_quality_chunks")
    @patch.object(extract_module, "chunk_text")
    @patch.object(rerank_module, "rerank_chunks")
    @patch.object(rerank_module, "filter_results_by_relevance")
    @patch.object(fetch_module, "fetch_pages_parallel")
    @patch.object(search_module, "search")
    def test_build_llm_prompt_runs_end_to_end_with_mocked_search_and_fetch(
        self,
        mock_search,
        mock_fetch,
        mock_filter_results,
        mock_rerank,
        mock_chunk_text,
        mock_filter_chunks,
    ):
        mock_search.return_value = [
            {
                "title": "Bitcoin price overview",
                "url": "https://example.com/btc",
                "snippet": "Bitcoin is trading near 67000 USD with strong volume.",
                "score": 1.0,
                "engines": ["bing", "ddg"],
            }
        ]
        mock_fetch.return_value = {
            "https://example.com/btc": {
                "text": (
                    "Bitcoin is trading near 67000 USD today. "
                    "Market participants are watching ETF flows and macro signals. "
                    "The article includes market cap, price action, and source context."
                ),
                "pub_date": None,
            }
        }
        mock_filter_results.side_effect = lambda query, results, threshold=0.25, lang="en": results
        mock_chunk_text.return_value = [
            "Bitcoin is trading near 67000 USD today with strong market volume and ETF-driven demand."
        ]
        mock_filter_chunks.side_effect = lambda chunks: chunks
        mock_rerank.return_value = [
            {
                "text": "Bitcoin is trading near 67000 USD today with strong market volume and ETF-driven demand.",
                "source_idx": 0,
                "source_url": "https://example.com/btc",
                "source_title": "Bitcoin price overview",
                "chunk_idx": 0,
                "pub_date": None,
                "relevance": 0.95,
                "bm25": 0.88,
                "semantic": 0.74,
            }
        ]

        prompt = prompts_module.build_llm_prompt("bitcoin rate", lang="en")

        self.assertIn("QUESTION: bitcoin rate", prompt)
        self.assertIn("[1] Bitcoin price overview", prompt)
        self.assertIn("https://example.com/btc", prompt)
        self.assertIn("## Источники:", prompt)


if __name__ == "__main__":
    unittest.main()
