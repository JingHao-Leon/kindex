"""Known-secret masking: values from user-named env files become $NAME.

Every value here is synthetic and chosen so the format-based credential
redactor (privacy.py) would NOT catch it: what removes it is the mask.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from kindex import mcp_server, secret_mask
from kindex.config import Config, load_config
from kindex.privacy import redact, redact_text
from kindex.secret_mask import (
    DEFAULT_KEY_PATTERNS,
    SecretMask,
    build_table,
    count_matches,
    key_selected,
    mask_text,
    parse_env_text,
)
from kindex.secret_scan import scan
from kindex.store import Store

SRC = str(Path(__file__).resolve().parents[1] / "src")
OPENAI = "zq7FakeOpenAiValue3Xy"
GITHUB = "fakeGithubValue81234"
SHORT = "short7x"


@pytest.fixture(autouse=True)
def _mask_off_after():
    yield
    secret_mask.configure(None)


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "fake.profile"
    path.write_text(
        "# synthetic fixture\n"
        f"export OPENAI_API_KEY=\"{OPENAI}\"\n"
        f"GITHUB_TOKEN={GITHUB}  # personal\n"
        "export PATH=\"/usr/local/bin:$PATH\"\n"
        "EDITOR=vim-something-long\n"
    )
    return path


@pytest.fixture
def masked(profile):
    return secret_mask.configure({"files": [str(profile)]})


# ── parsing ───────────────────────────────────────────────────────────

def test_parse_export_quotes_and_comments():
    text = (
        "export A_TOKEN=\"double-quoted-value\"\n"
        "B_TOKEN='single-quoted-value'\n"
        "C_TOKEN=unquoted-value-1 # trailing comment\n"
        "D_TOKEN=\"quoted # not a comment\"  # real comment\n"
        "E_TOKEN=has#hash-no-space\n"
        "  export   F_TOKEN=indented-value\n"
    )
    assert parse_env_text(text) == [
        ("A_TOKEN", "double-quoted-value"),
        ("B_TOKEN", "single-quoted-value"),
        ("C_TOKEN", "unquoted-value-1"),
        ("D_TOKEN", "quoted # not a comment"),
        ("E_TOKEN", "has#hash-no-space"),
        ("F_TOKEN", "indented-value"),
    ]


def test_parse_skips_expansions_short_values_and_non_assignments():
    text = (
        "A_TOKEN=\"$OTHER_secret_value\"\n"
        "B_TOKEN=prefix${HOME}suffix\n"
        "C_TOKEN=`cat /some/file`\n"
        "D_TOKEN='literal$dollar-value'\n"
        f"E_TOKEN={SHORT}\n"
        "F_TOKEN=\"unterminated-quote-value\n"
        "9BAD_TOKEN=invalid-name-value\n"
        "alias G_TOKEN=not-an-assignment\n"
        "H_TOKEN=exactly8\n"
    )
    assert parse_env_text(text) == [("H_TOKEN", "exactly8")]
    assert parse_env_text(text, min_length=9) == []
    assert ("E_TOKEN", SHORT) in parse_env_text(text, min_length=4)


def test_key_selectors_default_full_partial_and_glob():
    assert "CLIENT_ID" in DEFAULT_KEY_PATTERNS and "API" in DEFAULT_KEY_PATTERNS
    for name in ("GITHUB_TOKEN", "openai_api_key", "DB_PASSWORD", "SENTRY_DSN",
                 "SLACK_WEBHOOK_URL", "TWILIO_SID", "GOOGLE_CLIENT_ID"):
        assert key_selected(name), name
    for name in ("PATH", "HOME", "EDITOR", "LANG"):
        assert not key_selected(name), name
    # Full name, case-insensitive.
    assert key_selected("GITHUB_TOKEN", ["github_token"])
    assert not key_selected("OPENAI_API_KEY", ["github_token"])
    # Partial name: case-insensitive substring.
    assert key_selected("MY_OPENAI_KEY", ["openai"])
    # Glob.
    assert key_selected("AWS_SECRET_ACCESS_KEY", ["AWS_*_KEY"])
    assert not key_selected("AWS_REGION", ["AWS_*_KEY"])
    # An explicit list replaces the defaults.
    assert not key_selected("GITHUB_TOKEN", ["OPENAI"])
    assert key_selected("EDITOR", ["EDIT"])


# ── replacement ───────────────────────────────────────────────────────

def test_longest_value_first_and_first_name_wins():
    table = build_table([
        ("SHORT_KEY", "abcdefgh"),
        ("LONG_KEY", "abcdefgh-and-more"),
        ("FIRST_TOKEN", "shared-value-123"),
        ("SECOND_TOKEN", "shared-value-123"),
    ])
    text = "x abcdefgh-and-more y abcdefgh z shared-value-123"
    assert mask_text(text, table) == "x $LONG_KEY y $SHORT_KEY z $FIRST_TOKEN"
    assert count_matches(text, table) == {"LONG_KEY": 1, "SHORT_KEY": 1, "FIRST_TOKEN": 1}


def test_every_occurrence_and_json_escaped_spelling():
    table = build_table([("Q_TOKEN", 'quote"in-value')])
    assert mask_text('a quote"in-value b quote"in-value', table) == "a $Q_TOKEN b $Q_TOKEN"
    stored = json.dumps({"note": 'quote"in-value'})
    assert 'quote' not in mask_text(stored, table)
    assert json.loads(mask_text(stored, table)) == {"note": "$Q_TOKEN"}


def test_mask_rereads_changed_file_and_warns_once_for_missing(tmp_path, caplog):
    path = tmp_path / "env"
    path.write_text("A_TOKEN=first-value-aa\n")
    mask = SecretMask([path])
    assert mask.mask("first-value-aa") == "$A_TOKEN"
    path.write_text("A_TOKEN=second-value-bbb\n")  # size differs: re-read
    assert mask.mask("first-value-aa second-value-bbb") == "first-value-aa $A_TOKEN"

    missing = SecretMask([tmp_path / "absent"])
    with caplog.at_level(logging.WARNING, logger="kindex.secret_mask"):
        assert missing.mask("anything at all") == "anything at all"
        missing.mask("again")
    warnings = [r for r in caplog.records if "secret_mask" in r.getMessage()]
    assert len(warnings) == 1


def test_redact_text_masks_before_credential_rules(masked):
    assert redact_text(f"use {OPENAI} here") == "use $OPENAI_API_KEY here"
    # An assignment of a masked value keeps its safe placeholder.
    assert redact_text(f"password={GITHUB}") == "password=$GITHUB_TOKEN"
    assert redact({"k": [f"x{GITHUB}y"]}) == {"k": ["x$GITHUB_TOKENy"]}
    assert "vim-something-long" in redact_text("EDITOR=vim-something-long")
    secret_mask.configure(None)
    assert OPENAI in redact_text(f"use {OPENAI} here")


# ── write paths ───────────────────────────────────────────────────────

def _raw(store: Store, node_id: str) -> str:
    row = store.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return json.dumps(dict(row))


def test_store_writes_are_masked(tmp_path, masked):
    store = Store(Config(data_dir=str(tmp_path / "data")))
    try:
        nid = store.add_node(f"Title {OPENAI}", f"content {GITHUB}",
                             prov_why=f"why {OPENAI}", prov_source=f"src/{GITHUB}",
                             intent=f"intent {OPENAI}", extra={"note": GITHUB},
                             aka=[OPENAI])
        raw = _raw(store, nid)
        assert OPENAI not in raw and GITHUB not in raw
        assert "$OPENAI_API_KEY" in raw and "$GITHUB_TOKEN" in raw
        store.update_node(nid, content=f"new {OPENAI}")
        store.edit_node(nid, append=f"more {GITHUB}")
        new_id = store.supersede_node(nid, f"replacement {OPENAI}", reason=f"r {GITHUB}")["id"]
        store.add_edge(nid, new_id, provenance=f"edge {OPENAI}")
        for table in ("nodes", "edges", "activity_log"):
            dump = json.dumps([dict(r) for r in store.conn.execute(f"SELECT * FROM {table}")])
            assert OPENAI not in dump and GITHUB not in dump, table
        assert store.fts_search(OPENAI) == []
    finally:
        store.close()


def test_mcp_add_and_show_are_masked(tmp_path, masked, monkeypatch):
    store = Store(Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(mcp_server, "_get_store", lambda: (store, store.config))
    monkeypatch.setattr(mcp_server, "_store", store)
    monkeypatch.setattr(mcp_server, "_config", store.config)
    try:
        out = mcp_server.add(f"Deploy note: the key is {OPENAI} for staging")
        assert OPENAI not in out
        dump = json.dumps([dict(r) for r in store.conn.execute("SELECT * FROM nodes")])
        assert OPENAI not in dump and "$OPENAI_API_KEY" in dump
    finally:
        store.close()


def test_read_paths_mask_rows_written_before_the_feature(tmp_path, masked, monkeypatch):
    store = Store(Config(data_dir=str(tmp_path / "data")))
    nid = store.add_node("Historical masking canary")
    # A row from before the feature: bypass the write path on purpose.
    store.conn.execute("UPDATE nodes SET content = ? WHERE id = ?",
                       (f"old note with {OPENAI} inside", nid))
    store.conn.commit()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: (store, store.config))
    try:
        for call in (lambda: mcp_server.show(nid),
                     lambda: mcp_server.search("Historical masking canary"),
                     lambda: mcp_server.context("Historical masking canary"),
                     lambda: mcp_server.prime("Historical masking canary")):
            output = call()
            assert OPENAI not in output
        assert "$OPENAI_API_KEY" in mcp_server.show(nid)
        # Stored bytes are untouched until `kin mask scan --apply`.
        assert OPENAI in _raw(store, nid)
    finally:
        store.close()


# ── scan / apply ──────────────────────────────────────────────────────

def _seed_history(store: Store) -> str:
    nid = store.add_node("Scan canary", "clean")
    store.conn.execute("UPDATE nodes SET content = ?, extra = ? WHERE id = ?",
                       (f"a {OPENAI} b {OPENAI} c {GITHUB}",
                        json.dumps({"note": GITHUB}), nid))
    store.conn.execute("INSERT INTO activity_log (action, target_title) VALUES ('x', ?)",
                       (f"title {OPENAI}",))
    store.conn.commit()
    return nid


def test_scan_reports_names_never_values_and_apply_rewrites(tmp_path, masked):
    store = Store(Config(data_dir=str(tmp_path / "data")))
    try:
        nid = _seed_history(store)
        report = scan(store, masked)
        assert report.rows == 2
        assert report.by_key == {"OPENAI_API_KEY": 3, "GITHUB_TOKEN": 2}
        assert report.by_location == {"activity_log.target_title": 1,
                                      "nodes.content": 1, "nodes.extra": 1}
        dumped = report.model_dump_json()
        assert OPENAI not in dumped and GITHUB not in dumped
        assert not report.applied and OPENAI in _raw(store, nid)

        applied = scan(store, masked, apply=True)
        assert applied.applied and len(applied.snapshots) == 1
        assert Path(applied.snapshots[0]).exists()
        assert OPENAI not in _raw(store, nid) and GITHUB not in _raw(store, nid)
        assert store.get_node(nid)["extra"] == {"note": "$GITHUB_TOKEN"}
        assert store.fts_search(OPENAI) == []
        assert scan(store, masked).rows == 0
        actions = [r["action"] for r in store.recent_activity(10)]
        assert "secret_mask_apply" in actions and "db_snapshot" in actions
    finally:
        store.close()


# ── config authority and CLI ──────────────────────────────────────────

@pytest.fixture
def world(tmp_path, profile):
    home = tmp_path / "home"
    (home / ".config" / "kindex").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(p for p in (SRC, os.environ.get("PYTHONPATH", "")) if p),
        "PYTHONDONTWRITEBYTECODE": "1",
        "KIN_NO_SCHEDULER_WRITES": "1",
    }
    return {"home": home, "repo": repo, "env": env, "profile": profile,
            "data": tmp_path / "data"}


def _user_config(world, body: str) -> None:
    (world["home"] / ".config" / "kindex" / "kin.yaml").write_text(body)


def _probe(world) -> str:
    code = ("from kindex.config import load_config; from kindex import secret_mask; "
            "c = load_config(); m = secret_mask.active(); "
            "print(m is not None, 'secret_mask' in c._ignored_project_keys)")
    result = subprocess.run([sys.executable, "-c", code], cwd=world["repo"], env=world["env"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_project_config_cannot_enable_or_disable_masking(world):
    (world["repo"] / ".kin").mkdir()
    project = world["repo"] / ".kin" / "config"
    project.write_text(f"secret_mask:\n  files: [{world['profile']}]\n")
    assert _probe(world) == "False True"

    _user_config(world, f"secret_mask:\n  files: [{world['profile']}]\n")
    project.write_text("secret_mask: null\n")
    assert _probe(world) == "True True"
    project.write_text("secret_mask:\n  files: []\n  keys: [NOTHING_MATCHES]\n")
    assert _probe(world) == "True True"


def _kin(world, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kindex.cli", *args,
                           "--data-dir", str(world["data"])],
                          cwd=world["repo"], env=world["env"], text=True,
                          capture_output=True, timeout=60)


def test_cli_add_and_mask_scan(world):
    _user_config(world, f"secret_mask:\n  files: [{world['profile']}]\n")
    added = _kin(world, "add", f"Staging uses {OPENAI} as its key")
    assert added.returncode == 0, added.stderr
    assert OPENAI not in added.stdout + added.stderr
    db = world["data"] / "kindex.db"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE nodes SET intent = ? WHERE title LIKE 'Staging%'",
                     (f"legacy {GITHUB}",))
    dump = lambda: json.dumps(sqlite3.connect(db).execute("SELECT * FROM nodes").fetchall())
    assert OPENAI not in dump() and GITHUB in dump()

    found = _kin(world, "mask", "scan")
    assert found.returncode == 0, found.stderr
    assert "$GITHUB_TOKEN: 1" in found.stdout and GITHUB not in found.stdout
    assert "--apply" in found.stdout

    applied = _kin(world, "mask", "scan", "--apply", "--json")
    assert applied.returncode == 0, applied.stderr
    report = json.loads(applied.stdout)
    assert report["applied"] and report["by_key"] == {"GITHUB_TOKEN": 1}
    assert GITHUB not in dump()


def test_cli_mask_scan_refuses_when_off(world):
    result = _kin(world, "mask", "scan")
    assert result.returncode == 1
    assert "Secret masking is off" in result.stderr


def test_scan_covers_archives_and_reports_identifiers(tmp_path, masked):
    from kindex.archive import _open_archive

    store = Store(Config(data_dir=str(tmp_path / "data")))
    # A new identifier holding a known secret is refused, like any credential.
    with pytest.raises(ValueError):
        store.add_node("Plain", node_id=f"id-{GITHUB}")
    nid = store.add_node("Plain")
    store.conn.execute("UPDATE nodes SET id = ? WHERE id = ?", (f"id-{GITHUB}", nid))
    store.conn.commit()
    (tmp_path / "data" / "archive").mkdir()
    archive = _open_archive(tmp_path / "data" / "archive" / "current.db")
    archive.execute("INSERT INTO archived_nodes (id, title, content, archived_at) "
                    "VALUES ('old', 't', ?, 'now')", (f"archived {OPENAI}",))
    archive.commit()
    archive.close()
    try:
        report = scan(store, masked)
        assert report.by_location == {"archive/current.db:archived_nodes.content": 1}
        assert report.unmaskable == {"nodes.id": 1}
        applied = scan(store, masked, apply=True)
        assert len(applied.snapshots) == 1  # live had nothing to rewrite
        with sqlite3.connect(tmp_path / "data" / "archive" / "current.db") as conn:
            assert conn.execute("SELECT content FROM archived_nodes").fetchone()[0] == \
                "archived $OPENAI_API_KEY"
    finally:
        store.close()


def test_kindex_lite_applies_the_user_mask(tmp_path, profile):
    from kindex.config import bound_root
    from kindex.mcp_lite import create_server

    root = tmp_path / "root"
    (root / ".config" / "kindex").mkdir(parents=True)
    # Under a bound root, ~ is the root.
    (root / "fake.profile").write_text(profile.read_text())
    (root / ".config" / "kindex" / "kin.yaml").write_text(
        "secret_mask:\n  files: [~/fake.profile]\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    with bound_root(root):
        create_server(repo, no_kinbase=True)
    assert secret_mask.active() is not None
    assert redact_text(OPENAI) == "$OPENAI_API_KEY"
