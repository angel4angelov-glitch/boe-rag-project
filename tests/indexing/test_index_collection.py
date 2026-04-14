"""Tests for index_collection — idempotency, batching, metadata upsert."""

from __future__ import annotations

from dataclasses import dataclass, field

from boe_rag.indexing.chroma_store import index_collection


@dataclass
class _FakeCollection:
    """Minimal in-memory stand-in for chromadb.Collection.

    Tracks upsert / delete calls so tests can assert on behaviour without
    involving the real embedding API.
    """

    name: str = "test_collection"
    _ids: list[str] = field(default_factory=list)
    _documents: list[str] = field(default_factory=list)
    _metadatas: list[dict | None] = field(default_factory=list)
    upsert_calls: list[dict] = field(default_factory=list)
    delete_calls: list[list[str]] = field(default_factory=list)

    def count(self) -> int:
        return len(self._ids)

    def upsert(self, *, ids, documents, metadatas=None):
        self.upsert_calls.append(
            {"ids": list(ids), "documents": list(documents), "metadatas": metadatas}
        )
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                idx = self._ids.index(id_)
                self._documents[idx] = documents[i]
                if metadatas is not None:
                    self._metadatas[idx] = metadatas[i]
                continue
            self._ids.append(id_)
            self._documents.append(documents[i])
            self._metadatas.append(metadatas[i] if metadatas is not None else None)

    def delete(self, *, ids):
        self.delete_calls.append(list(ids))
        for id_ in list(ids):
            if id_ in self._ids:
                idx = self._ids.index(id_)
                self._ids.pop(idx)
                self._documents.pop(idx)
                self._metadatas.pop(idx)

    def get(self):
        return {"ids": list(self._ids)}


def _records(n: int, *, with_metadata: bool) -> list[dict]:
    out = []
    for i in range(n):
        r = {"id": f"c_{i:03d}", "text": f"text {i}"}
        if with_metadata:
            r["metadata"] = {"date": "2025-11", "idx": i}
        out.append(r)
    return out


def test_index_collection_skips_when_count_matches() -> None:
    col = _FakeCollection()
    records = _records(5, with_metadata=False)
    # Pre-populate so count matches.
    col.upsert(ids=[r["id"] for r in records], documents=[r["text"] for r in records])
    col.upsert_calls.clear()

    index_collection(col, records)

    assert col.upsert_calls == [], "should skip when count already matches"
    assert col.delete_calls == []


def test_index_collection_force_re_indexes_even_when_count_matches() -> None:
    col = _FakeCollection()
    records = _records(5, with_metadata=False)
    col.upsert(ids=[r["id"] for r in records], documents=[r["text"] for r in records])
    col.upsert_calls.clear()

    index_collection(col, records, force=True)

    # Force path clears then re-upserts.
    assert len(col.delete_calls) == 1
    assert len(col.upsert_calls) >= 1


def test_index_collection_count_mismatch_clears_and_reupserts() -> None:
    col = _FakeCollection()
    # Stale content (3 chunks) that does not match new records (5).
    col.upsert(ids=["old_1", "old_2", "old_3"], documents=["a", "b", "c"])
    col.upsert_calls.clear()

    index_collection(col, _records(5, with_metadata=False))

    assert col.delete_calls, "expected old IDs to be cleared"
    assert "old_1" in col.delete_calls[0]
    assert col.count() == 5
    # New ids replaced the old ones.
    assert "old_1" not in col._ids


def test_index_collection_batches_records() -> None:
    col = _FakeCollection()
    records = _records(25, with_metadata=False)
    index_collection(col, records, batch_size=10)

    # 25 records / 10 batch_size = 3 upsert calls (10, 10, 5).
    assert len(col.upsert_calls) == 3
    assert [len(call["ids"]) for call in col.upsert_calls] == [10, 10, 5]


def test_index_collection_omits_metadata_when_records_have_none() -> None:
    col = _FakeCollection()
    index_collection(col, _records(3, with_metadata=False))
    assert col.upsert_calls[0]["metadatas"] is None


def test_index_collection_passes_metadata_when_records_have_some() -> None:
    col = _FakeCollection()
    index_collection(col, _records(3, with_metadata=True))
    assert col.upsert_calls[0]["metadatas"] is not None
    assert col.upsert_calls[0]["metadatas"][0] == {"date": "2025-11", "idx": 0}


def test_index_collection_splits_batches_by_token_budget() -> None:
    """A batch exceeding the OpenAI per-request token budget must split.

    Simulates mixing long (box-sized) and short records; with max_records=100
    but per-chunk ~900 tokens, token budget (30 000) binds first.
    """
    col = _FakeCollection()
    # 60 records, ~900 tokens each (simulating box-heavy mix).
    long_text = "word " * 900  # count_tokens ≈ 900
    records = [{"id": f"c_{i:03d}", "text": long_text} for i in range(60)]

    index_collection(col, records, batch_size=100)

    # 60 * ~900 = 54 000 tokens total. Budget 30 000 per batch → 2 batches.
    assert len(col.upsert_calls) >= 2, (
        f"expected token-budget split, got {len(col.upsert_calls)} batches"
    )
    # All records got indexed exactly once.
    assert sum(len(call["ids"]) for call in col.upsert_calls) == 60


def test_index_collection_empty_records_does_nothing() -> None:
    col = _FakeCollection()
    index_collection(col, [])
    assert col.upsert_calls == []
    assert col.delete_calls == []
