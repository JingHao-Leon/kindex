"""Evidence order and labels in kin ask's context (answer.assemble): rank_order,
position_labels, fact_sources; and queueing many nodes for embedding at once."""

from kindex import answer
from kindex.config import Config
from kindex.store import Store


def _chunk(nid, when, position, text):
    return {"id": nid, "type": "document", "title": "", "content": text, "prov_when": when,
            "prov_activity": "conversation-ingest", "prov_source": "c1_p2",
            "extra": {"conversation_id": "c1_p2", "position": position}}


def _claim(nid, text, position, excerpt):
    return {"id": nid, "type": "document", "title": "", "content": text, "prov_when": "2024-03-15",
            "prov_activity": "conversation-digest",
            "extra": {"conversation_id": "c1_p2", "kind": "conversation-fact", "fact_date": "2024-03-15",
                      "turn_claim": True, "source": {"position": position, "role": "user", "excerpt": excerpt}}}


RESULTS = [_chunk("late", "2024-03-15", 7, "user: the rate is now 12%"),
           _chunk("early", "2024-03-15", 2, "user: the rate is 15%"),
           _claim("f", "The estate-tax estimate is now 12%.", 7, "the rate is now 12%")]


def test_default_order_is_by_date_with_facts_first():
    text = answer.assemble(RESULTS, [], 4000).text
    assert text.index("Facts recorded") < text.index("Evidence, oldest first")
    assert "s2 #7" not in text and "said:" not in text


def test_rank_order_keeps_the_search_order_with_labels_and_sources():
    text = answer.assemble(RESULTS, [], 4000, rank_order=True, position_labels=True, fact_sources=True).text
    assert "most relevant first" in text
    late, early, fact = (text.index(s) for s in ("now 12%\n", "is 15%", "The estate-tax"))
    assert late < early < fact
    assert "[2024-03-15 · s2 #7]" in text and "[2024-03-15 · s2 #2]" in text
    assert "[s2 #7; the user said: “the rate is now 12%”]" in text


def test_position_order_sorts_a_days_excerpts_by_place_in_the_conversation():
    text = answer.assemble(RESULTS[:2], [], 4000, source_order=True).text
    assert text.index("is 15%") < text.index("now 12%")


def test_enqueue_embeddings_writes_the_queue_once(tmp_path, monkeypatch):
    from kindex import vectors

    store = Store(Config(data_dir=str(tmp_path)))
    try:
        ids = [store.add_node(f"n{i}", "text", queue_embedding=False) for i in range(5)]
        assert vectors._load_embedding_queue(store) == []
        writes = []
        real = vectors._write_embedding_meta
        monkeypatch.setattr(vectors, "_write_embedding_meta",
                            lambda *a, **kw: (writes.append(1), real(*a, **kw))[1])
        assert vectors.enqueue_embeddings(store, ids + ids[:2]) == 5
        assert writes == [1] and vectors._load_embedding_queue(store) == ids
    finally:
        store.close()

