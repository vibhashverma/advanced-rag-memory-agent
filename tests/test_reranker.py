"""TEST 2: Cohere rerank filtering (no real Cohere calls)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import cohere
import pytest
from langchain_cohere import CohereRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.compression.reranker import RerankError, ThresholdCohereRerank, build_compression_retriever
from tests.conftest import fake_cohere_reranker


def docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc {i}", metadata={"source": "s.txt", "parent_id": f"p{i}"}) for i in range(n)]


def compressor(mock, top_n=5, min_score=0.3) -> ThresholdCohereRerank:
    return ThresholdCohereRerank(reranker=mock, top_n=top_n, min_score=min_score)


def test_sorted_by_score_and_below_threshold_removed():
    # API returns unsorted scores; 3 and 6 fall below 0.3.
    mock = fake_cohere_reranker({0: 0.5, 1: 0.9, 2: 0.31, 3: 0.29, 4: 0.7, 5: 0.99, 6: 0.01})
    out = compressor(mock).compress_documents(docs(7), "q")
    assert [d.page_content for d in out] == ["doc 5", "doc 1", "doc 4", "doc 0", "doc 2"]
    scores = [d.metadata["relevance_score"] for d in out]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= 0.3 for s in scores)


def test_threshold_is_inclusive():
    out = compressor(fake_cohere_reranker({0: 0.3, 1: 0.2999}), min_score=0.3).compress_documents(docs(2), "q")
    assert [d.page_content for d in out] == ["doc 0"]


def test_max_five_returned_even_if_api_returns_more():
    mock = fake_cohere_reranker({i: 0.9 - i * 0.01 for i in range(20)})  # all 20 eligible
    out = compressor(mock, top_n=5).compress_documents(docs(20), "q")
    assert len(out) == 5
    assert [d.page_content for d in out] == [f"doc {i}" for i in range(5)]
    assert mock.calls[0]["n_docs"] == 20 and mock.calls[0]["top_n"] == 5  # 20 in, 5 requested


def test_metadata_preserved_and_score_added():
    out = compressor(fake_cohere_reranker({2: 0.8})).compress_documents(docs(3), "q")
    assert out[0].metadata == {"source": "s.txt", "parent_id": "p2", "relevance_score": 0.8}


def test_source_documents_not_mutated():
    src = docs(2)
    compressor(fake_cohere_reranker({0: 0.9})).compress_documents(src, "q")
    assert "relevance_score" not in src[0].metadata


def test_empty_input_makes_no_api_call():
    mock = fake_cohere_reranker()
    assert compressor(mock).compress_documents([], "q") == []
    assert mock.calls == []


def test_all_below_threshold_gives_empty_result():
    assert compressor(fake_cohere_reranker({0: 0.1, 1: 0.05})).compress_documents(docs(2), "q") == []


def test_empty_api_result():
    assert compressor(fake_cohere_reranker({})).compress_documents(docs(3), "q") == []


def test_out_of_range_index_ignored():
    mock = SimpleNamespace(rerank=lambda *a, **k: [{"index": 99, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}])
    out = compressor(mock).compress_documents(docs(2), "q")
    assert [d.page_content for d in out] == ["doc 0"]


def test_api_failure_raises_rerank_error_without_leaking_secret():
    secret = "co-SECRETKEY1234567890"
    mock = SimpleNamespace(
        rerank=MagicMock(side_effect=RuntimeError(f"401 unauthorized, token {secret}")),
        cohere_api_key=SimpleNamespace(get_secret_value=lambda: secret),
    )
    with pytest.raises(RerankError) as e:
        compressor(mock).compress_documents(docs(3), "q")
    assert secret not in str(e.value)


def test_real_cohere_rerank_wrapper_with_mocked_client():
    """Exercise the actual langchain_cohere.CohereRerank class against a mocked ClientV2."""
    client = MagicMock(spec=cohere.ClientV2)
    client.rerank.return_value = SimpleNamespace(
        results=[SimpleNamespace(index=i, relevance_score=0.95 - i * 0.1) for i in (3, 1, 0, 2, 4)]
    )
    cr = CohereRerank(model="rerank-english-v3.0", top_n=5, client=client, cohere_api_key="x")
    out = ThresholdCohereRerank(reranker=cr, top_n=5, min_score=0.6).compress_documents(docs(20), "q")
    kwargs = client.rerank.call_args.kwargs
    assert kwargs["model"] == "rerank-english-v3.0" and kwargs["top_n"] == 5 and len(kwargs["documents"]) == 20
    # API order was (3,1,0,2,4); scores are .65,.85,.95,.75,.55 -> sorted desc, doc 4 (.55) dropped
    assert [d.page_content for d in out] == ["doc 0", "doc 1", "doc 2", "doc 3"]


class _Base(BaseRetriever):
    items: list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.items


def test_contextual_compression_retriever_reduces_20_to_5():
    mock = fake_cohere_reranker({i: 0.99 - i * 0.02 for i in range(20)})
    retriever = build_compression_retriever(compressor(mock, top_n=5, min_score=0.0), _Base(items=docs(20)))
    out = retriever.invoke("what is hybrid search?")
    assert len(out) == 5
    assert all("relevance_score" in d.metadata for d in out)
