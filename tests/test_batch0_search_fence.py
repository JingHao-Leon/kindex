"""Run batch0 acceptance suite -- S3: retired knowledge stays retired.

Tester-lane authored, blind to the implementation; the oracle is the signed
run artifacts only. Citation shorthand:

  spec@1f0cdd71  = product-specification.md
      sha256 1f0cdd7134ad671d3795252cbaedde858ae82dd0ef4d5a65d7d09a048bca617b
  arch@59540239  = architecture.md
      sha256 595402395f445c619122ede5435a303b31e8ed9df5d4c1f96e2db73a6ec4f1e6
  strat@e58068c2 = testing-strategy.md
      sha256 e58068c20913db342c6e24af5896f5f24d429ecc699103a8a787fda0372eece5

Markers per strat@e58068c2 T0.2: ``red_now`` = asserts FIXED behavior,
expected to fail on unfixed base 8c5cc925648c; unmarked = green-now guard.
Determinism per T0.4. Status construction uses the same surfaces the
read-only suite uses: update_node(status=...) (tests/test_archive.py),
supersede_node(...) (tests/test_supersede_lifecycle.py), and direct SQL on
fixture rows (tests/test_archive.py). The ``_section`` helper is replicated
from the read-only tests/test_expiry.py (section-scoped assertions, because
the Recent-activity audit section legitimately mentions retired titles).

NOTE reported upward (not resolved here): spec R3.1 says the escape hatch
"restores today's behavior", while the R3.5 note text says --include-archived
reveals the fenced "archived/superseded" results; today's behavior already
excludes superseded. These tests therefore assert the uncontested core only:
default excludes BOTH statuses; include_archived=True reveals ARCHIVED. What
include_archived does with superseded nodes is left unasserted.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from kindex.config import Config
from kindex.retrieve import hybrid_search
from kindex.store import Store


def _section(out: str, header: str) -> str:
    """Extract one '### <header>' section from prime output (empty if
    absent). Replicated from the read-only tests/test_expiry.py."""
    captured = []
    inside = False
    for line in out.splitlines():
        if line.startswith("### "):
            inside = line.startswith(f"### {header}")
            continue
        if inside:
            captured.append(line)
    return "\n".join(captured)


def _env(tmp_path):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("KIN_"):
            env.pop(k)
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
              "GOOGLE_API_KEY", "VOYAGE_API_KEY"):
        env.pop(k, None)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    return env


def _run_cli(tmp_path, *args, data_dir):
    cmd = [sys.executable, "-m", "kindex.cli", *args,
           "--data-dir", str(data_dir)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          env=_env(tmp_path), cwd=str(tmp_path))


@pytest.fixture
def fence_store(tmp_path):
    """One live, one archived, one superseded node -- all matching the
    unique token ``quantumfence`` (strat@e58068c2 oracle row R3.1 fixture:
    active node A + archived node B, both matching a query; superseded added
    for the spec's full status list)."""
    cfg = Config(data_dir=str(tmp_path / "data"))
    s = Store(cfg)
    s.add_node("Fence live alpha", content="quantumfence telemetry live",
               node_id="live1", weight=1.0)
    s.add_node("Fence archived alpha",
               content="quantumfence telemetry archived",
               node_id="arch1", weight=1.0)
    s.update_node("arch1", status="archived")
    old_id = s.add_node("Fence superseded alpha",
                        content="quantumfence telemetry superseded",
                        node_id="sup1", weight=1.0)
    s.supersede_node(old_id, "Completely different replacement zebra text")
    yield s
    s.close()


# -- R3.1: the fence, on every default search surface ---------------------


@pytest.mark.red_now
def test_r3_1_fts_default_excludes_archived_and_superseded(fence_store):
    """spec@1f0cdd71 R3.1; strat@e58068c2 oracle row R3.1 (red_now).

    Default fts_search returns the live node and neither the archived nor
    the superseded one.

    Red if: the fence reverts to excluding only 'superseded' (the S3
    defect: archived nodes remain first-class FTS candidates).
    """
    ids = {r["id"] for r in fence_store.fts_search("quantumfence")}
    assert "live1" in ids
    assert "arch1" not in ids, "archived node leaked into default search"
    assert "sup1" not in ids


@pytest.mark.red_now
def test_r3_1_fts_include_archived_escape_hatch(fence_store):
    """spec@1f0cdd71 R3.1 (escape hatch); strat@e58068c2 oracle row R3.1
    (red_now): ``include_archived=True`` includes the archived node B.

    Red if: the additive ``include_archived`` parameter is absent
    (TypeError on base) or does not reveal archived nodes.
    """
    ids = {r["id"]
           for r in fence_store.fts_search("quantumfence",
                                           include_archived=True)}
    assert "live1" in ids
    assert "arch1" in ids, "escape hatch must reveal archived nodes"


def test_r3_1_include_archived_uses_fts_and_preserves_rank(fence_store):
    """R3.1 mechanism guard: archive escape-hatch searches stay indexed.

    This is a green-now regression guard: the base projection already uses FTS,
    while a later implementation regression silently swallowed malformed FTS
    SQL and fell back to an unindexed LIKE scan with every rank set to zero.
    Reachability/non-degeneracy: the fixture has live, archived, and
    superseded matches; sqlite trace callbacks capture every statement for both
    flag states, and returned rows expose real rank values.  The assertion is
    on the prohibited mechanism (FTS in both calls) and non-degenerate ranking,
    not merely plausible returned titles.
    """
    # Add a distinct-frequency live match so the FTS BM25 rank has at least two
    # observable values independent of the shared fence fixture's equal text.
    fence_store.add_node(
        "Fence rank contrast",
        content="quantumfence " * 8,
        node_id="rank-contrast",
    )
    traces: list[str] = []
    fence_store.conn.set_trace_callback(traces.append)
    try:
        default = fence_store.fts_search("quantumfence")
        default_trace = list(traces)
        traces.clear()
        included = fence_store.fts_search(
            "quantumfence", include_archived=True
        )
        included_trace = list(traces)
    finally:
        fence_store.conn.set_trace_callback(None)

    assert any("nodes_fts" in sql.lower() and "match" in sql.lower()
               for sql in included_trace), (
        f"include_archived search did not execute an FTS MATCH query: {included_trace!r}"
    )
    assert any("nodes_fts" in sql.lower() and "match" in sql.lower()
               for sql in default_trace), (
        f"default search did not execute an FTS MATCH query: {default_trace!r}"
    )
    ranks = [row["rank"] for row in included]
    assert len(ranks) >= 2, "fixture must produce multiple ranked matches"
    assert len(set(ranks)) > 1, f"include_archived ranks collapsed: {ranks!r}"


@pytest.mark.red_now
def test_r3_1_r3_4_hybrid_fence_and_order_preserved(fence_store):
    """spec@1f0cdd71 R3.1 + R3.4; strat@e58068c2 oracle rows R3.1, R3.4
    (red_now via the new parameter; the order property itself is the R3.4
    contract).

    Default hybrid_search excludes the archived node; with
    include_archived=True it appears. No re-scoring: the surviving live
    candidates keep the same relative order in the default call as they
    have in the fence-free call.

    Red if: hybrid lacks the escape hatch (base: TypeError), the default
    result still carries archived nodes, or the fence re-orders surviving
    candidates relative to the fence-free ranking.
    """
    default_ids = [r["id"] for r in hybrid_search(fence_store,
                                                  "quantumfence")]
    assert "live1" in default_ids
    assert "arch1" not in default_ids
    assert "sup1" not in default_ids

    unfenced_ids = [r["id"] for r in hybrid_search(fence_store,
                                                   "quantumfence",
                                                   include_archived=True)]
    assert "arch1" in unfenced_ids
    live_subseq = [i for i in unfenced_ids if i in set(default_ids)]
    assert live_subseq == default_ids, (
        "fencing must not re-order the surviving candidates (R3.4)")


@pytest.mark.red_now
def test_r3_1_mcp_search_fences_archived(fence_store, tmp_path, monkeypatch):
    """spec@1f0cdd71 R3.1 (MCP search surface); strat@e58068c2 oracle row
    R3.1 (red_now). Convention per read-only tests/test_mcp.py: patch the
    module singletons.

    Red if: MCP default search still returns archived titles, or the
    additive include_archived tool parameter is absent/inert.
    """
    try:
        import kindex.mcp_server as mcp_mod
    except (ImportError, SystemExit):
        pytest.skip("kindex.mcp_server unavailable (mcp extra missing or "
                    "incompatible) -- MCP coverage requires the [mcp] extra "
                    "in the authoritative environment")

    cfg = Config(data_dir=str(tmp_path / "data"))
    monkeypatch.setattr(mcp_mod, "_store", fence_store)
    monkeypatch.setattr(mcp_mod, "_config", cfg)

    default_out = mcp_mod.search("quantumfence")
    assert "Fence live alpha" in default_out
    assert "Fence archived alpha" not in default_out, (
        "archived node leaked into default MCP search")

    unfenced_out = mcp_mod.search("quantumfence", include_archived=True)
    assert "Fence archived alpha" in unfenced_out


@pytest.mark.red_now
def test_r3_1_cli_search_fences_archived_with_flag_escape(tmp_path):
    """spec@1f0cdd71 R3.1 (CLI surface: --include-archived flag);
    strat@e58068c2 oracle row R3.1 (red_now).

    Red if: `kin search` still prints archived titles by default, or the
    additive --include-archived flag is absent (argparse error) or inert.
    """
    data_dir = tmp_path / "data"
    s = Store(Config(data_dir=str(data_dir)))
    s.add_node("Fence live alpha", content="quantumfence telemetry live",
               node_id="live1")
    s.add_node("Fence archived alpha",
               content="quantumfence telemetry archived", node_id="arch1")
    s.update_node("arch1", status="archived")
    s.close()

    r = _run_cli(tmp_path, "search", "quantumfence", data_dir=data_dir)
    assert r.returncode == 0, r.stderr
    assert "Fence live alpha" in r.stdout
    assert "Fence archived alpha" not in r.stdout

    r2 = _run_cli(tmp_path, "search", "quantumfence", "--include-archived",
                  data_dir=data_dir)
    assert r2.returncode == 0, r2.stderr
    assert "Fence archived alpha" in r2.stdout


@pytest.mark.red_now
def test_r3_1_prime_key_concepts_fences_archived(fence_store):
    """spec@1f0cdd71 R3.1 (hook priming is a default-search surface);
    strat@e58068c2 oracle row R3.1 (red_now).

    prime_context's Key-concepts section shows the live node and not the
    archived one. Section-scoped per the read-only test_expiry.py pattern
    (the Recent-activity audit section may legitimately name any title).

    Red if: hook priming still surfaces archived nodes (the 360-junk-nodes
    symptom in the S3 defect).
    """
    from kindex.hooks import prime_context

    out = prime_context(fence_store, topic="quantumfence", max_tokens=1500)
    section = _section(out, "Key concepts")
    assert "Fence live alpha" in section
    assert "Fence archived alpha" not in section


def test_r3_1_legacy_default_status_row_still_searchable(tmp_path):
    """spec@1f0cdd71 R3.1 (the fence excludes archived/superseded -- and
    nothing else); strat@e58068c2 edge-case list ("a node whose status is
    the legacy default") (green-now: schema v7 makes nodes.status NOT NULL
    DEFAULT 'active', so the legacy default is the column default -- a NULL
    status is unconstructible, and base search includes default-status rows
    too; classification observed at the 0.29.0 authoring baseline, Validator
    re-verifies per T0.5).

    A row INSERTed directly (legacy-shaped: no add_node bookkeeping, every
    omitted column taking its schema default, including status='active') is
    neither archived nor superseded, so default FTS and hybrid search MUST
    return it, before and after the fix.

    Red if: the new fence predicate drops default-status rows (mis-spelled
    status literal, over-broad IN-list, or a JOIN that loses rows lacking
    add_node-era bookkeeping).
    """
    cfg = Config(data_dir=str(tmp_path / "data"))
    s = Store(cfg)
    try:
        s.conn.execute(
            "INSERT INTO nodes (id, title, content) "
            "VALUES ('leg1', 'Legacy status row', 'quantumlegacy payload')")
        s.conn.commit()
        assert s.conn.execute(
            "SELECT status FROM nodes WHERE id = 'leg1'"
        ).fetchone()[0] == "active", "fixture: column default expected"
        fts_ids = {r["id"] for r in s.fts_search("quantumlegacy")}
        hyb_ids = {r["id"] for r in hybrid_search(s, "quantumlegacy")}
    finally:
        s.close()
    assert "leg1" in fts_ids, "legacy default-status row dropped by FTS fence"
    assert "leg1" in hyb_ids, (
        "legacy default-status row dropped by hybrid fence")


# -- R3.2: operational pulls filter to active status ----------------------


@pytest.mark.red_now
def test_r3_2_archived_decision_absent_from_operational_pull(tmp_path):
    """spec@1f0cdd71 R3.2; strat@e58068c2 oracle row R3.2 (red_now).

    An archived decision does not appear in the context formatter's
    recent-decisions (operational) output while an active decision does.
    The active twin is the in-test reachability control: if it does not
    render, the declared surface does not exist as stated and this test
    fails loudly instead of passing vacuously -- that failure is a
    spec-defect signal, not a green.

    Red if: operational pulls still select decisions without a status
    filter (the S3 defect), surfacing the archived decision.
    """
    from kindex.hooks import prime_context

    cfg = Config(data_dir=str(tmp_path / "data"))
    s = Store(cfg)
    try:
        s.add_node("Quantumdecide anchor concept",
                   content="quantumdecide anchor content",
                   node_type="concept", node_id="anchor1")
        s.add_node("Decision keep postgres for quantumdecide",
                   content="We chose postgres.", node_type="decision",
                   node_id="dec-live")
        s.add_node("Decision retired mysql for quantumdecide",
                   content="We chose mysql.", node_type="decision",
                   node_id="dec-arch")
        s.update_node("dec-arch", status="archived")
        out = prime_context(s, topic="quantumdecide", max_tokens=4000)
    finally:
        s.close()
    assert "Decision keep postgres for quantumdecide" in out, (
        "reachability control failed: an ACTIVE decision does not render in "
        "prime output at all -- declared surface mismatch, report upward")
    assert "Decision retired mysql for quantumdecide" not in out, (
        "archived decision leaked into the operational pull")


# -- R3.3: backfill makes result counts honest ----------------------------


@pytest.fixture
def backfill_store(tmp_path):
    """8 strongly-matching EXPIRED nodes + 5 weaker LIVE nodes, all
    matching ``quantumfill`` (strat@e58068c2 oracle row R3.3 fixture:
    fenced/dropped candidates previously consumed top_k slots). Expiry via
    extra={'expires': PAST} per the read-only tests/test_expiry.py."""
    cfg = Config(data_dir=str(tmp_path / "data"))
    s = Store(cfg)
    for i in range(8):
        s.add_node(f"Quantumfill expired {i}",
                   content="quantumfill payload quantumfill strong",
                   node_id=f"exp{i}", weight=1.0,
                   extra={"expires": "2020-01-01"})
    for i in range(5):
        s.add_node(f"Quantumfill live {i}",
                   content="some quantumfill payload",
                   node_id=f"live{i}", weight=0.4)
    yield s
    s.close()


@pytest.mark.red_now
def test_r3_3_hybrid_backfills_to_honest_top_k(backfill_store):
    """spec@1f0cdd71 R3.3; strat@e58068c2 oracle row R3.3 (red_now).

    With >= top_k live candidates available, default hybrid_search returns
    EXACTLY top_k live results: dropped (expired) entries are backfilled
    from the merged candidate list, not silently subtracted.

    Red if: hybrid_search again drop-filters AFTER slicing top_k (the S3
    defect) -- the strong expired nodes consume slice slots and the call
    returns fewer than top_k results.
    """
    results = hybrid_search(backfill_store, "quantumfill", top_k=5)
    ids = [r["id"] for r in results]
    assert len(ids) == 5, (
        f"result count lies: expected top_k=5 after backfill, got "
        f"{len(ids)}: {ids}")
    assert all(i.startswith("live") for i in ids), (
        f"dropped/retired candidates leaked into results: {ids}")


def test_r3_3_backfill_exhaustion_returns_all_live(backfill_store):
    """spec@1f0cdd71 R3.3 (backfill until top_k or exhaustion);
    strat@e58068c2 oracle row R3.3, exhaustion half (green-now: with top_k
    above the whole candidate population the base slice already covers
    everything, so base and fixed return the identical live set --
    classification observed at the 0.29.0 authoring baseline, Validator
    re-verifies per T0.5).

    With top_k above the live population, the result is every live match
    and nothing else -- exhaustion, not padding, and still no retired
    entries.

    Red if: backfill overshoots by padding with dropped/retired candidates
    to reach top_k, or the exhaustion path miscounts or duplicates.
    """
    results = hybrid_search(backfill_store, "quantumfill", top_k=20)
    ids = sorted(r["id"] for r in results)
    assert ids == [f"live{i}" for i in range(5)], ids


# -- R3.4: no re-scoring, deterministic ranking (green guard) -------------


def test_r3_4_all_active_ranking_is_deterministic(tmp_path):
    """spec@1f0cdd71 R3.4; strat@e58068c2 oracle row R3.4 (green-now).

    On an all-active fixture the fence has nothing to do: repeated default
    calls return the same complete result set in the same order -- the
    decision path is a pure function of its inputs (no clock, no
    randomness). This green guard is the half of R3.4 assertable within one
    build; the cross-build "order unchanged vs base" half is established by
    the Validator running this same test at base and final (recorded in the
    handover as such).

    Red if: ranking acquires nondeterminism (random tie-breaks, clock
    dependence), drops a live candidate, or duplicates one.
    """
    cfg = Config(data_dir=str(tmp_path / "data"))
    s = Store(cfg)
    try:
        for i, w in enumerate([1.0, 0.9, 0.75, 0.6, 0.45, 0.3]):
            s.add_node(f"Quantumstable concept {i}",
                       content=f"quantumstable payload {i}",
                       node_id=f"st{i}", weight=w)
        first = [r["id"] for r in hybrid_search(s, "quantumstable", top_k=6)]
        second = [r["id"] for r in hybrid_search(s, "quantumstable", top_k=6)]
    finally:
        s.close()
    assert first == second, "ranking must be deterministic across calls"
    assert len(first) == 6, f"all live matches must be returned: {first}"
    assert len(set(first)) == 6, f"duplicate results: {first}"


# -- R3.5: the fence never lies by omission -------------------------------


@pytest.mark.red_now
def test_r3_5_cli_note_when_fenced_candidates_would_rank(tmp_path):
    """spec@1f0cdd71 R3.5; strat@e58068c2 oracle row R3.5 (red_now).

    I3 defect report: the former assertion required the defective literal
    ``3 archived/superseded`` even though this fixture contains only archived
    nodes.  The expectation below is intentionally category-accurate.

    2 live + 3 archived matches: the default CLI search returns fewer than
    top_k live results while fenced candidates would have ranked, so the
    output appends the one-line note naming the count and the escape hatch.
    With the same store, --include-archived shows the archived titles and
    (per the note's own definition) the note's trigger condition is gone.

    Red if: no note is emitted on a short default result (base), the count
    is wrong, or the note omits the escape-hatch pointer.
    """
    data_dir = tmp_path / "data"
    s = Store(Config(data_dir=str(data_dir)))
    for i in range(2):
        s.add_node(f"Quantumnote live {i}", content="quantumnote payload",
                   node_id=f"live{i}")
    for i in range(3):
        s.add_node(f"Quantumnote archived {i}",
                   content="quantumnote payload archived",
                   node_id=f"arch{i}")
        s.update_node(f"arch{i}", status="archived")
    s.close()

    r = _run_cli(tmp_path, "search", "quantumnote", data_dir=data_dir)
    assert r.returncode == 0, r.stderr
    assert "3 archived results fenced" in r.stdout, (
        f"missing/wrong fence note in: {r.stdout!r}")
    assert "--include-archived" in r.stdout, (
        "the note must name the escape hatch")
    assert "Quantumnote archived 0" not in r.stdout
    included = _run_cli(
        tmp_path, "search", "quantumnote", "--include-archived", data_dir=data_dir
    )
    assert included.returncode == 0, included.stderr
    delivered = sum(
        f"Quantumnote archived {i}" in included.stdout for i in range(3)
    )
    count = re.search(r"\b(\d+)\s+archived results fenced\b", r.stdout)
    assert count and int(count.group(1)) == delivered, (
        f"fence note count must equal additional archived items delivered: "
        f"note={r.stdout!r}, included={included.stdout!r}"
    )


@pytest.mark.red_now
def test_r3_5_cli_note_names_each_fenced_status_category(tmp_path):
    """I3 defect follow-up: mixed fenced statuses must both be named exactly.

    Two archived and one superseded candidate are fenced behind one live
    result.  Amendment 4 requires the total fenced count plus both named
    statuses; it does not require a particular separator or per-category
    breakdown.
    """
    data_dir = tmp_path / "mixed-data"
    s = Store(Config(data_dir=str(data_dir)))
    s.add_node("Quantummix live", content="quantummix payload", node_id="live")
    for i in range(2):
        node_id = f"arch{i}"
        s.add_node(f"Quantummix archived {i}", content="quantummix payload",
                   node_id=node_id)
        s.update_node(node_id, status="archived")
    old_id = s.add_node("Quantummix superseded", content="quantummix payload",
                        node_id="sup-old")
    s.supersede_node(old_id, "replacement without quantummix")
    s.close()

    r = _run_cli(tmp_path, "search", "quantummix", data_dir=data_dir)
    assert r.returncode == 0, r.stderr
    note_lines = [line.lower() for line in r.stdout.splitlines()
                  if "results fenced" in line.lower()]
    assert note_lines, f"mixed-status fence note is missing: {r.stdout!r}"
    note = " ".join(note_lines)
    assert "archived" in note and "superseded" in note, (
        f"mixed-status note must name both fenced categories: {note!r}"
    )
    count = re.search(r"\b(\d+)\s+[^\n]*results fenced\b", note)
    assert count and int(count.group(1)) == 3, (
        f"mixed-status note must report fenced-set size 3: {note!r}"
    )


def test_r3_5_no_note_when_top_k_satisfied_or_nothing_fenced(tmp_path):
    """spec@1f0cdd71 R3.5 (no note when top_k is satisfied by live
    results); strat@e58068c2 oracle row R3.5 second half (green-now: the
    note string does not exist on base, so its absence holds identically
    before and after the fix; the sibling test carries the base-failing
    note-present assertion). This is also the anti-gaming guard: a
    degenerate implementation that appends the note unconditionally
    satisfies the R3.5 red test but fails here.

    Two stores: (a) 60 live + 3 archived matches -- any sane default top_k
    is satisfied by live results, so no note; (b) all-live matches --
    nothing fenced, so no note.

    Red if: the note is emitted unconditionally whenever anything was
    fenced (a), or on every short result even with nothing fenced (b).
    """
    data_a = tmp_path / "data-a"
    s = Store(Config(data_dir=str(data_a)))
    for i in range(60):
        s.add_node(f"Quantumsat live {i}", content="quantumsat payload",
                   node_id=f"live{i}")
    for i in range(3):
        s.add_node(f"Quantumsat archived {i}",
                   content="quantumsat payload archived",
                   node_id=f"arch{i}")
        s.update_node(f"arch{i}", status="archived")
    s.close()
    r_a = _run_cli(tmp_path, "search", "quantumsat", data_dir=data_a)
    assert r_a.returncode == 0, r_a.stderr
    assert "results fenced" not in r_a.stdout, (
        "no note when top_k is satisfied by live results")

    data_b = tmp_path / "data-b"
    s = Store(Config(data_dir=str(data_b)))
    s.add_node("Quantumsolo live", content="quantumsolo payload",
               node_id="solo1")
    s.close()
    r_b = _run_cli(tmp_path, "search", "quantumsolo", data_dir=data_b)
    assert r_b.returncode == 0, r_b.stderr
    assert "results fenced" not in r_b.stdout, (
        "no note when nothing was fenced")


@pytest.mark.red_now
def test_r3_5_mcp_note_when_fenced_candidates_would_rank(fence_store,
                                                         tmp_path,
                                                         monkeypatch):
    """spec@1f0cdd71 R3.5 (MCP surface); strat@e58068c2 oracle row R3.5
    (red_now).

    The fence_store holds 1 live + 1 archived + 1 superseded match: the
    default MCP search is short of top_k with fenced candidates that would
    have ranked, so the tool result carries the fence note; nothing fenced
    on a fresh all-live store means no note.

    Red if: the MCP tool result omits the note on a short default result,
    or emits it when nothing was fenced.
    """
    try:
        import kindex.mcp_server as mcp_mod
    except (ImportError, SystemExit):
        pytest.skip("kindex.mcp_server unavailable (mcp extra missing or "
                    "incompatible) -- MCP coverage requires the [mcp] extra "
                    "in the authoritative environment")

    cfg = Config(data_dir=str(tmp_path / "data"))
    monkeypatch.setattr(mcp_mod, "_store", fence_store)
    monkeypatch.setattr(mcp_mod, "_config", cfg)
    out = mcp_mod.search("quantumfence")
    assert "results fenced" in out, (
        f"missing fence note in MCP result: {out!r}")
    assert "include_archived" in out, "the note must name the escape hatch"


@pytest.fixture
def multi_fence_store(tmp_path):
    """Several live survivors with distinct relevance, interleaved with
    archived and superseded matches, so R3.4 ordering is non-vacuous."""
    s = Store(Config(data_dir=str(tmp_path / "multi")))
    for n in range(1, 5):
        s.add_node(f"Live {n}", content="ordertoken " * n + "live", node_id=f"live{n}", weight=1.0)
        s.add_node(f"Archived {n}", content="ordertoken " * (n + 1) + "archived",
                   node_id=f"arch{n}", weight=1.0)
        s.update_node(f"arch{n}", status="archived")
    old_id = s.add_node("Superseded", content="ordertoken " * 3 + "superseded", node_id="sup1", weight=1.0)
    s.supersede_node(old_id, "Completely different replacement zebra text")
    yield s
    s.close()


def test_r3_4_fence_keeps_order_of_several_survivors(multi_fence_store):
    """R3.4 with more than one live survivor: the fenced result is exactly
    the fence-free ranking with archived and superseded nodes removed."""
    default_ids = [r["id"] for r in hybrid_search(multi_fence_store, "ordertoken")]
    unfenced_ids = [r["id"] for r in hybrid_search(multi_fence_store, "ordertoken",
                                                   include_archived=True)]
    live = {f"live{n}" for n in range(1, 5)}
    assert set(default_ids) == live
    assert {"arch1", "sup1"} <= set(unfenced_ids)
    assert [i for i in unfenced_ids if i in live] == default_ids, (
        "fencing must not re-order the surviving candidates (R3.4)")
