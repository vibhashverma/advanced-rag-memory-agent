"""TEST 1: Reciprocal Rank Fusion."""
import pytest
from langchain_core.documents import Document

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rrf import reciprocal_rank_fusion
from tests.conftest import StaticRetriever


def doc(cid: str, text: str = "") -> Document:
    return Document(page_content=text or cid, metadata={"child_id": cid})


DENSE = [doc("A"), doc("B"), doc("C")]  # mocked dense results, best first
SPARSE = [doc("B"), doc("D"), doc("A")]  # mocked sparse results, best first


def expected(k: int = 60) -> dict[str, float]:
    return {
        "A": 1 / (k + 1) + 1 / (k + 3),
        "B": 1 / (k + 2) + 1 / (k + 1),
        "C": 1 / (k + 3),
        "D": 1 / (k + 2),
    }


def test_ranking_and_scores_match_formula():
    fused = reciprocal_rank_fusion([DENSE, SPARSE], k=60)
    exp = expected()
    assert [d.metadata["child_id"] for d in fused] == ["B", "A", "D", "C"]
    for d in fused:
        assert d.metadata["rrf_score"] == pytest.approx(exp[d.metadata["child_id"]])


def test_duplicates_are_merged():
    fused = reciprocal_rank_fusion([DENSE, SPARSE])
    ids = [d.metadata["child_id"] for d in fused]
    assert len(ids) == len(set(ids)) == 4  # 6 inputs, 4 unique
    b = next(d for d in fused if d.metadata["child_id"] == "B")
    assert b.metadata["rrf_ranks"] == {0: 2, 1: 1}  # rank in each source list


def test_custom_k_changes_scores():
    fused = reciprocal_rank_fusion([DENSE, SPARSE], k=1)
    top = fused[0]
    assert top.metadata["rrf_score"] == pytest.approx(expected(1)["B"])


def test_limit_and_empty_inputs():
    assert len(reciprocal_rank_fusion([DENSE, SPARSE], limit=2)) == 2
    assert reciprocal_rank_fusion([[], []]) == []
    only_dense = reciprocal_rank_fusion([DENSE, []])
    assert [d.metadata["child_id"] for d in only_dense] == ["A", "B", "C"]


def test_duplicate_inside_one_list_not_double_counted():
    fused = reciprocal_rank_fusion([[doc("A"), doc("A")]], k=60)
    assert len(fused) == 1
    assert fused[0].metadata["rrf_score"] == pytest.approx(1 / 61)


def test_ties_are_deterministic():
    fused = reciprocal_rank_fusion([[doc("X")], [doc("Y")]])
    assert [d.metadata["child_id"] for d in fused] == ["X", "Y"]


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([DENSE], k=-1)


def test_hybrid_retriever_runs_both_and_fuses():
    dense, sparse = StaticRetriever(DENSE), StaticRetriever(SPARSE)
    res = HybridRetriever(dense, sparse, rrf_k=60).retrieve("q")
    assert dense.calls == ["q"] and sparse.calls == ["q"]  # both retrievers executed
    assert (res.dense_count, res.sparse_count, res.fused_count) == (3, 3, 4)
    assert res.children[0].metadata["child_id"] == "B"
