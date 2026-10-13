"""Tests for conversation ingest and digests (kindex.conversations)."""

import json
from types import SimpleNamespace

import pytest

from kindex import conversations as conv
from kindex.config import Config, LLMConfig
from kindex.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(Config(data_dir=str(tmp_path)))
    yield s
    s.close()


MESSAGES = [
    {"role": "user", "content": "I bought a red kayak for $450."},
    {"role": "assistant", "content": "Nice! " + "Paddle safely. " * 300},
    {"role": "user", "content": "Always give distances in kilometres."},
]


def test_pack_keeps_whole_messages_and_splits_long_ones():
    chunks = conv.pack(["a" * 10, "b" * 25, "c" * 5], limit=20)
    assert chunks == ["a" * 10, "b" * 20, "b" * 5 + "\n" + "c" * 5]
    assert all(len(c) <= 20 for c in chunks)


def test_ingest_is_lossless_dated_linked_and_idempotent(store):
    ids = conv.ingest_conversation(store, "c1", MESSAGES, "2024/03/10 (Sun) 09:00")
    assert len(ids) == 3  # the 4,500-character reply is split across two nodes
    text = "\n".join(store.get_node(i)["content"] for i in ids)
    for m in MESSAGES:
        assert m["content"].strip() in text.replace("\n", "")
    first = store.get_node(ids[0])
    assert first["prov_when"] == "2024/03/10 (Sun) 09:00"
    assert first["type"] == "document"
    assert [e["to_id"] for e in store.edges_from(ids[0])] == [ids[1]]
    assert conv.ingest_conversation(store, "c1", MESSAGES, "2024/03/10 (Sun) 09:00") == []


def test_ingest_directory_reads_json_and_jsonl(store, tmp_path):
    d = tmp_path / "convs"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"id": "a", "date": "2024-01-01", "messages": MESSAGES[:1]}))
    (d / "b.jsonl").write_text(json.dumps({"id": "b", "messages": MESSAGES[2:]}) + "\n")
    assert conv.ingest_directory(store, d) == 2


def test_conversations_adapter_is_discovered():
    from kindex.adapters.registry import discover

    assert "conversations" in discover()


def _fake_client(payload):
    calls = []

    def create(**kw):
        calls.append(kw)
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))], usage=usage)

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def test_digest_records_directives_once_and_a_linked_summary(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": ["Always give distances in kilometres."],
                                  "summary": "The user bought a red kayak for $450 on 10 March 2024."})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    cfg = Config(data_dir=str(tmp_path), llm=LLMConfig(enabled=True, provider="openai", model="m"))
    monkeypatch.setattr(conv, "SUMMARY_MIN_TOKENS", 0)
    ids = conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    assert conv.backfill_digests(store, cfg) == 1
    directives = store.all_nodes(node_type="directive")
    assert [d["title"] for d in directives] == ["Always give distances in kilometres."]
    summary = [n for n in store.all_nodes(node_type="document") if n["id"].startswith("convsum-")][0]
    assert "$450" in summary["content"] and summary["prov_when"] == "2024-03-10"
    assert {e["to_id"] for e in store.edges_from(summary["id"])} == set(ids)
    assert calls[0]["json_schema"]["name"] == "conversation_digest"
    # Already digested: no second call, no duplicate directive.
    assert conv.backfill_digests(store, cfg) == 0
    assert len(calls) == 1


def test_short_conversations_get_directives_but_no_summary(store, tmp_path, monkeypatch):
    from kindex import llm

    from kindex.config import ConversationsConfig

    client, calls = _fake_client({"directives": [], "summary": "should not be stored"})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    cfg = Config(data_dir=str(tmp_path), llm=LLMConfig(enabled=True, provider="openai", model="m"),
                 conversations=ConversationsConfig(facts=False))
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert conv.backfill_digests(store, cfg) == 1
    assert calls[0]["json_schema"]["schema"]["required"] == ["directives"]
    assert not [n for n in store.all_nodes(node_type="document") if n["id"].startswith("convsum-")]


def test_user_messages_keeps_only_the_users_side():
    text = "user: line one\ncontinues here\nassistant: a long reply\nuser: second"
    assert conv.user_messages(text) == "user: line one\ncontinues here\nuser: second"
    assert conv.user_messages("Caroline: hi\nMelanie: hello") == ""


def test_facts_become_dated_nodes_when_enabled(store, tmp_path, monkeypatch):
    from kindex import llm
    from kindex.config import ConversationsConfig

    client, calls = _fake_client({"directives": [], "facts": [
        {"date": "2024-03-09", "subject": "user", "text": "The user bought a red kayak for $450 on the Saturday before 2024-03-10."}]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    cfg = Config(data_dir=str(tmp_path), llm=LLMConfig(enabled=True, provider="openai", model="m"),
                 conversations=ConversationsConfig(facts=True))
    conv.ingest_conversation(store, "c1", MESSAGES[:2], "2024-03-10")
    conv.backfill_digests(store, cfg)
    prompt = calls[0]["messages"][0]["content"]
    assert "- facts:" in prompt and "assistant: Nice!" in prompt  # full text, not the user's side only
    facts = [n for n in store.all_nodes(node_type="document") if n["id"].startswith("convfact-")]
    assert len(facts) == 1 and facts[0]["extra"]["fact_date"] == "2024-03-09"


def test_facts_are_on_by_default_and_can_be_turned_off(store, tmp_path, monkeypatch):
    from kindex import llm
    from kindex.config import ConversationsConfig

    client, calls = _fake_client({"directives": [], "facts": []})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    conv.backfill_digests(store, Config(data_dir=str(tmp_path), llm=LLMConfig(enabled=True, provider="openai",
                                                                             model="m")))
    assert "- facts:" in calls[0]["messages"][0]["content"]
    conv.ingest_conversation(store, "c2", MESSAGES[2:], "2024-03-11")
    conv.backfill_digests(store, Config(data_dir=str(tmp_path), conversations=ConversationsConfig(facts=False),
                                        llm=LLMConfig(enabled=True, provider="openai", model="m")))
    assert "- facts:" not in calls[-1]["messages"][0]["content"]


# ── Re-ingest reconciles a conversation with what was stored before ──

def _chunks(store, cid):
    nodes = [n for n in store.all_nodes(node_type="document", limit=10_000)
             if (n.get("extra") or {}).get("conversation_id") == cid and "kind" not in (n.get("extra") or {})]
    return sorted(nodes, key=lambda n: n["extra"]["position"])


def _linked_in_order(store, nodes):
    return all(any(e["to_id"] == b["id"] for e in store.edges_from(a["id"])) for a, b in zip(nodes, nodes[1:]))


def test_an_append_inside_the_last_chunk_is_stored(store):
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    changed = conv.ingest_conversation(store, "c1", MESSAGES[:1] + [{"role": "user", "content": "And a paddle."}],
                                       "2024-03-10")
    assert changed == [conv.chunk_id("c1", 0)]
    assert "And a paddle." in store.get_node(changed[0])["content"]


def test_an_append_across_a_chunk_boundary_is_stored_and_linked(store):
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    chunks = _chunks(store, "c1")
    assert len(chunks) == 3 and _linked_in_order(store, chunks)
    assert "kilometres" in chunks[-1]["content"]


def test_an_edit_replaces_only_the_changed_chunk(store):
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    edited = [{"role": "user", "content": "I bought a blue kayak for $450."}] + MESSAGES[1:]
    assert conv.ingest_conversation(store, "c1", edited, "2024-03-10") == [conv.chunk_id("c1", 0)]
    assert "blue kayak" in _chunks(store, "c1")[0]["content"]
    assert _linked_in_order(store, _chunks(store, "c1"))


def test_a_shorter_conversation_drops_its_old_tail(store):
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert len(_chunks(store, "c1")) == 1


def test_an_interrupted_ingest_is_completed_next_time(store, monkeypatch):
    real = store.add_node
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(store, "add_node", flaky)
    with pytest.raises(OSError):
        conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    monkeypatch.setattr(store, "add_node", real)
    assert len(_chunks(store, "c1")) == 1
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    chunks = _chunks(store, "c1")
    assert len(chunks) == 3 and _linked_in_order(store, chunks)


# ── Digests: grouping, status and revisions ──

def _cfg(tmp_path, **conversations):
    from kindex.config import ConversationsConfig

    return Config(data_dir=str(tmp_path), llm=LLMConfig(enabled=True, provider="openai", model="m"),
                  conversations=ConversationsConfig(**conversations))


def test_conversations_sharing_a_file_are_digested_separately(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": []})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    d = tmp_path / "convs"
    d.mkdir()
    (d / "export.jsonl").write_text(
        json.dumps({"date": "2024-01-01", "messages": [{"role": "user", "content": "January kayak"}]}) + "\n"
        + json.dumps({"date": "2024-02-01", "messages": [{"role": "user", "content": "February canoe"}]}) + "\n")
    conv.ingest_directory(store, d)
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 2
    prompts = sorted(c["messages"][0]["content"] for c in calls)
    assert "on 2024-01-01" in prompts[0] and "January kayak" in prompts[0] and "February" not in prompts[0]
    assert "on 2024-02-01" in prompts[1] and "February canoe" in prompts[1] and "January" not in prompts[1]
    for cid in ("export#0", "export#1"):
        assert _chunks(store, cid)[0]["prov_source"] == str(d / "export.jsonl")


def test_a_failed_digest_is_retried(store, tmp_path, monkeypatch):
    from kindex import llm

    def broken(**kw):
        raise RuntimeError("OpenAI API error 500")

    monkeypatch.setattr(llm, "get_client", lambda config, **kw: SimpleNamespace(
        messages=SimpleNamespace(create=broken)))
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 0
    client, calls = _fake_client({"directives": ["Always give distances in kilometres."]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 1
    assert len(store.all_nodes(node_type="directive")) == 1


def test_an_unparseable_digest_is_retried(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client("not an object")
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 0
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 0
    assert len(calls) == 2


def test_a_spent_budget_records_nothing(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": []})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    spent = SimpleNamespace(can_spend=lambda: False)
    assert conv.backfill_digests(store, _cfg(tmp_path), spent) == 0
    assert calls == []
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 1


def test_a_changed_conversation_is_digested_again(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": ["Always give distances in kilometres."]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 1
    conv.ingest_conversation(store, "c1", MESSAGES[:1] + MESSAGES[2:], "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 1
    assert "kilometres" in calls[-1]["messages"][0]["content"]
    assert len(store.all_nodes(node_type="directive")) == 1
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 0


def test_turning_on_facts_digests_again(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": [], "facts": []})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path, facts=False)) == 1
    assert conv.backfill_digests(store, _cfg(tmp_path, facts=True)) == 1
    assert "- facts:" in calls[-1]["messages"][0]["content"]


def test_a_record_from_before_revisions_is_honoured(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": []})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    store.set_meta("conversation_digests", json.dumps(["c1"]))
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 0 and calls == []


# ── Retention: expiry and retraction reach everything derived ──

def test_retraction_removes_the_conversation_and_what_only_it_gave(store, tmp_path, monkeypatch):
    from kindex import llm

    client, _ = _fake_client({"directives": ["Always give distances in kilometres."],
                              "summary": "A kayak purchase."})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    monkeypatch.setattr(conv, "SUMMARY_MIN_TOKENS", 0)
    d = tmp_path / "convs"
    d.mkdir()
    (d / "a.json").write_text(json.dumps([{"id": "a", "date": "2024-01-01", "messages": MESSAGES},
                                          {"id": "b", "date": "2024-01-02", "messages": MESSAGES[2:]}]))
    conv.ingest_directory(store, d)
    assert conv.backfill_digests(store, _cfg(tmp_path)) == 2
    (d / "a.json").write_text(json.dumps([{"id": "a", "retracted": True},
                                          {"id": "b", "date": "2024-01-02", "messages": MESSAGES[2:]}]))
    conv.ingest_directory(store, d)
    assert _chunks(store, "a") == []
    assert not [n for n in store.all_nodes(node_type="document", limit=1000)
                if (n.get("extra") or {}).get("conversation_id") == "a"]
    directive = store.all_nodes(node_type="directive")
    assert len(directive) == 1 and directive[0]["extra"]["conversations"] == {"b": None}
    (d / "a.json").write_text(json.dumps([{"id": "b", "retracted": True}]))
    conv.ingest_directory(store, d)
    assert store.all_nodes(node_type="directive") == []
    assert "a" not in json.loads(store.get_meta("conversation_digests"))


def test_expiry_reaches_chunks_and_everything_derived(store, tmp_path, monkeypatch):
    from kindex import llm
    from kindex.store import node_expired

    client, _ = _fake_client({"directives": ["Always give distances in kilometres."], "summary": "Kayak."})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    monkeypatch.setattr(conv, "SUMMARY_MIN_TOKENS", 0)
    conv.ingest_conversation(store, "old", MESSAGES, "2024-01-01", expires="2024-06-01")
    conv.backfill_digests(store, _cfg(tmp_path))
    derived = [n for n in store.all_nodes(limit=1000)
               if (n.get("extra") or {}).get("conversation_id") == "old" or n["type"] == "directive"]
    assert len(derived) == 5  # three chunks, a summary, a directive
    assert all(node_expired(n, today="2025-01-01") for n in derived)
    # Another conversation giving the same directive, without an expiry, keeps it live.
    conv.ingest_conversation(store, "new", MESSAGES[2:], "2024-02-01")
    conv.backfill_digests(store, _cfg(tmp_path))
    directive = store.all_nodes(node_type="directive")[0]
    assert not node_expired(directive, today="2025-01-01")


# ── Profiles: written once from the facts, shown for the people a question names ──

def _facts(store, subject, n, cid="c1", start=1):
    for i in range(start, start + n):
        store.add_node(f"{subject} fact {i}", content=f"{subject} did thing {i}.", node_type="document",
                       node_id=f"f-{subject}-{cid}-{i}", prov_activity="conversation-digest", prov_when="2024-03-10",
                       extra={"kind": "conversation-fact", "subject": subject, "fact_date": f"2024-03-{i:02d}",
                              "conversation_id": cid})


def test_profiles_are_written_from_facts_and_kept_until_they_change(store, tmp_path, monkeypatch):
    from kindex import llm

    calls = []

    def create(**kw):
        calls.append(kw)
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(text="- Likely single.")], usage=usage)

    monkeypatch.setattr(llm, "get_client", lambda config, **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    _facts(store, "Caroline", 9)
    _facts(store, "Sam", 3)  # too few facts for a profile
    cfg = _cfg(tmp_path)
    assert conv.build_profiles(store, cfg) == 1
    profile = store.get_node("convprofile-" + __import__("hashlib").sha256(b"caroline").hexdigest()[:16])
    assert profile["extra"]["entity"] == "Caroline" and "Likely single" in profile["content"]
    assert "Caroline did thing 9." in calls[0]["messages"][0]["content"]
    assert conv.build_profiles(store, cfg) == 0 and len(calls) == 1  # unchanged facts: no call
    _facts(store, "Caroline", 1, start=10)
    assert conv.build_profiles(store, cfg) == 1  # a new fact: rebuilt
    conv.retract_conversation(store, "c1")
    assert not [n for n in store.all_nodes(node_type="document")
                if (n.get("extra") or {}).get("kind") == "entity-profile"]


def test_a_question_gets_the_profiles_of_the_people_it_names(store):
    from kindex import answer

    for entity in ("Caroline", "user"):
        store.add_node(f"Profile of {entity}", content=f"About {entity}.", node_type="document",
                       node_id=f"p-{entity}", extra={"kind": "entity-profile", "entity": entity})
    assert [p["id"] for p in answer.question_profiles(store, "Would Caroline be considered religious?")] == ["p-Caroline"]
    assert [p["id"] for p in answer.question_profiles(store, "What should I cook for my partner?")] == ["p-user"]
    assert answer.question_profiles(store, "What did Melanie paint?") == []
    text = answer.assemble([], [], 2000, profiles=answer.question_profiles(store, "Is Caroline religious?")).text
    assert "## Profiles" in text and "Profile of Caroline:\nAbout Caroline." in text


# ── Review fixes ──

def test_a_named_message_keeps_its_role(store):
    lines = conv._lines([{"role": "assistant", "name": "Bot", "content": "Hi"},
                         {"role": "user", "name": "Caroline", "content": "Hello"},
                         {"role": "user", "content": "Plain"}])
    assert lines == ["assistant (Bot): Hi", "user (Caroline): Hello", "user: Plain"]
    assert conv.user_messages("\n".join(lines)) == "user: Hello\nuser: Plain"


def test_a_failed_replacement_keeps_the_stored_chunk_and_its_links(store, monkeypatch):
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    first = conv.chunk_id("c1", 0)
    store.add_node("Other note", node_id="other")
    store.add_edge("other", first, edge_type="relates_to")

    def broken(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(store, "add_node", broken)
    edited = [{"role": "user", "content": "I bought a blue kayak."}] + MESSAGES[1:]
    with pytest.raises(OSError):
        conv.ingest_conversation(store, "c1", edited, "2024-03-10")
    monkeypatch.undo()
    assert "red kayak" in store.get_node(first)["content"]
    assert any(e["to_id"] == first for e in store.edges_from("other"))


def test_an_undated_conversation_is_not_rewritten(store):
    assert conv.ingest_conversation(store, "u1", MESSAGES[:1], None)
    assert conv.ingest_conversation(store, "u1", MESSAGES[:1], None) == []


def test_an_emptied_conversation_takes_its_derived_records_with_it(store, tmp_path, monkeypatch):
    from kindex import llm

    client, _ = _fake_client({"directives": ["Always give distances in kilometres."], "facts": [
        {"date": "2024-03-09", "subject": "user", "text": "The user bought a red kayak."}]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    conv.backfill_digests(store, _cfg(tmp_path))
    assert store.all_nodes(node_type="directive")
    assert conv.ingest_conversation(store, "c1", [], "2024-03-10") == []
    assert not [n for n in store.all_nodes(node_type="document", limit=1000)
                if (n.get("extra") or {}).get("conversation_id") == "c1"]
    assert store.all_nodes(node_type="directive") == []
    assert "c1" not in json.loads(store.get_meta("conversation_digests") or "{}")


def test_a_retried_digest_replaces_what_an_interrupted_one_wrote(store, tmp_path, monkeypatch):
    from kindex import llm

    conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10")
    # An interrupted earlier run left a fact at a position the new digest also uses.
    store.add_node("stale", content="A stale fact.", node_type="document",
                   node_id="convfact-" + __import__("hashlib").sha256(b"c1|5").hexdigest()[:16],
                   extra={"conversation_id": "c1", "kind": "conversation-fact"})
    client, _ = _fake_client({"directives": [], "facts": [
        {"date": "2024-03-09", "subject": "user", "text": f"Fact {i}."} for i in range(7)]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.backfill_digests(store, _cfg(tmp_path))
    facts = sorted(n["content"] for n in store.all_nodes(node_type="document", limit=1000)
                   if (n.get("extra") or {}).get("kind") == "conversation-fact")
    assert facts == [f"Fact {i}." for i in range(7)]


def test_an_edited_fact_rebuilds_the_profile(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"profile": "x"})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    _facts(store, "Caroline", 9)
    assert conv.build_profiles(store, _cfg(tmp_path)) == 1
    store.update_node("f-Caroline-c1-3", content="Caroline moved to Lisbon.")
    assert conv.build_profiles(store, _cfg(tmp_path)) == 1


def test_an_expiry_that_is_not_a_date_is_refused(store, tmp_path):
    with pytest.raises(ValueError):
        conv.ingest_conversation(store, "c1", MESSAGES[:1], "2024-03-10", expires="never")
    assert store.get_node(conv.chunk_id("c1", 0)) is None
    d = tmp_path / "convs"
    d.mkdir()
    (d / "a.json").write_text(json.dumps([{"id": "a", "messages": MESSAGES[:1], "expires": "soon"},
                                          {"id": "b", "messages": MESSAGES[:1], "expires": "2030-01-01"}]))
    errors = []
    assert conv.ingest_directory(store, d, errors=errors) == 1
    assert len(errors) == 1 and "conversation a" in errors[0]


def test_ingest_honours_limit_and_since(store, tmp_path):
    from kindex.adapters.conversations import adapter

    d = tmp_path / "convs"
    d.mkdir()
    (d / "a.jsonl").write_text("\n".join(json.dumps({"id": f"c{i}", "date": f"2024-0{i + 1}-01",
                                                     "messages": MESSAGES[:1]}) for i in range(4)))
    assert adapter.ingest(store, directory=str(d), limit=2).created == 2
    assert store.get_node(conv.chunk_id("c2", 0)) is None
    assert adapter.ingest(store, directory=str(d), limit=10, since="2024-04-01").created == 1
    assert store.get_node(conv.chunk_id("c3", 0)) is not None and store.get_node(conv.chunk_id("c2", 0)) is None


# ── Turn-level claims (conversations.claims) ──

CLAIMS = {"directives": [], "summary": "A kayak purchase.", "claims": [
    {"m": 1, "kind": "fact", "subject": "user", "date": "2024-03-10", "text": "The user bought a red kayak for $450."},
    {"m": 3, "kind": "preference", "subject": "user", "date": "", "text": "The user wants distances in kilometres."},
    {"m": 9, "kind": "fact", "subject": "user", "date": "", "text": "Out of range: dropped."},
]}


def test_messages_split_on_roles_and_recurring_names_only():
    chunks = [{"id": "a", "position": 0, "when": "d", "content": "Caroline: Hi Mel!\nMelanie: Hey!\nNote: not a speaker"},
              {"id": "b", "position": 1, "when": "d", "content": "still Melanie\nCaroline: Bye\nMelanie: Bye!"}]
    msgs = conv.conversation_messages(chunks)
    assert [(m["speaker"], m["position"]) for m in msgs] == [
        ("Caroline", 0), ("Melanie", 0), ("Caroline", 1), ("Melanie", 1)]
    assert msgs[1]["text"] == "Hey!\nNote: not a speaker\nstill Melanie"
    roles = conv.conversation_messages([{"id": "c", "position": 0, "when": "d",
                                         "content": "user: hello\nassistant (Ana): hi\nmore"}])
    assert [(m["speaker"], m["role"], m["text"]) for m in roles] == [
        ("user", "user", "hello"), ("assistant (Ana)", "assistant", "hi\nmore")]


def test_claim_windows_share_short_conversations_and_split_long_ones():
    msgs = [{"conversation_id": c} for c in ["a"] * 3 + ["b"] * 3 + ["c"] * 9]
    assert [len(w) for w in conv.claim_windows(msgs, window=4)] == [6, 8, 1]


def test_digest_with_claims_ties_each_claim_to_its_message(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client(CLAIMS)
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    monkeypatch.setattr(conv, "SUMMARY_MIN_TOKENS", 0)
    ids = conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path, claims=True)) == 1
    assert [c["json_schema"]["name"] for c in calls] == ["conversation_digest", "conversation_claims"]
    assert "- facts:" not in calls[0]["messages"][0]["content"]  # claims replace the session facts
    claims = sorted((n for n in store.all_nodes(node_type="document", limit=100)
                     if (n.get("extra") or {}).get("turn_claim")), key=lambda n: n["content"])
    assert [n["content"] for n in claims] == ["The user bought a red kayak for $450.",
                                              "The user wants distances in kilometres."]
    kayak, km = claims
    assert kayak["extra"]["source"] == {"position": 0, "role": "user", "excerpt": "I bought a red kayak for $450."}
    assert km["extra"]["source"]["position"] == 2 and km["extra"]["kind"] == "conversation-fact"
    assert {e["to_id"] for e in store.edges_from(kayak["id"])} == {ids[0]}
    from kindex.vectors import _load_embedding_queue
    assert {kayak["id"], km["id"]} <= set(_load_embedding_queue(store))
    assert conv.backfill_digests(store, _cfg(tmp_path, claims=True)) == 0


def test_backfill_claims_replaces_facts_once(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": [], "facts": [{"date": "", "subject": "user",
                                                              "text": "A session-level fact."}]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path, facts=True)) == 1
    client2, calls2 = _fake_client(CLAIMS)
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client2)
    assert conv.backfill_claims(store, _cfg(tmp_path, facts=True)) == 1
    facts = [n["content"] for n in store.all_nodes(node_type="document", limit=100)
             if (n.get("extra") or {}).get("kind") == "conversation-fact"]
    assert sorted(facts) == ["The user bought a red kayak for $450.", "The user wants distances in kilometres."]
    assert conv.backfill_claims(store, _cfg(tmp_path, facts=True)) == 0 and len(calls2) == 1


def test_a_failed_claims_call_writes_nothing(store, tmp_path, monkeypatch):
    from kindex import llm

    client, _ = _fake_client({"directives": [], "summary": "s"})  # no "claims": the answer is unusable
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")
    assert conv.backfill_digests(store, _cfg(tmp_path, claims=True)) == 0
    assert store.get_meta("conversation_digests") in (None, "{}")


def test_named_dialogue_keeps_its_facts(store, tmp_path, monkeypatch):
    from kindex import llm

    client, calls = _fake_client({"directives": [], "facts": [{"date": "", "subject": "Caroline",
                                                              "text": "Caroline went to a support group."}]})
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    conv.ingest_conversation(store, "s1", [{"role": "Caroline", "content": "I went to a support group."},
                                           {"role": "Melanie", "content": "That's great!"},
                                           {"role": "Caroline", "content": "It was powerful."}], "2023-05-08")
    assert conv.backfill_digests(store, _cfg(tmp_path, claims=True)) == 1
    assert [c["json_schema"]["name"] for c in calls] == ["conversation_digest"]
    facts = [n for n in store.all_nodes(node_type="document", limit=100)
             if (n.get("extra") or {}).get("kind") == "conversation-fact"]
    assert [f["content"] for f in facts] == ["Caroline went to a support group."]
    assert not any(f["extra"].get("turn_claim") for f in facts)
    assert conv.backfill_claims(store, _cfg(tmp_path, claims=True)) == 0 and len(calls) == 1


def test_digest_claims_reads_each_conversations_claims_once(store, tmp_path, monkeypatch, capsys):
    # `kin digest --claims` with conversations.claims on: the digest pass reads the
    # claims, and the --claims pass must not read them a second time.
    from argparse import Namespace

    from kindex import cli, llm, vectors

    client, calls = _fake_client(CLAIMS)
    monkeypatch.setattr(llm, "get_client", lambda config, **kw: client)
    monkeypatch.setattr(conv, "SUMMARY_MIN_TOKENS", 0)
    monkeypatch.setattr(vectors, "drain_embedding_queue", lambda *a, **kw: None)
    cfg = _cfg(tmp_path, claims=True)
    monkeypatch.setattr(cli, "_store", lambda args: store)
    monkeypatch.setattr(cli, "_ledger", lambda args: (None, cfg))
    monkeypatch.setattr(store, "close", lambda: None)
    conv.ingest_conversation(store, "c1", MESSAGES, "2024-03-10")

    cli.cmd_digest(Namespace(claims=True))

    assert [c["json_schema"]["name"] for c in calls].count("conversation_claims") == 1
    assert "Gave 0 conversation(s) turn-level claims." in capsys.readouterr().out
    assert conv.backfill_claims(store, cfg) == 0 and len(calls) == 2
