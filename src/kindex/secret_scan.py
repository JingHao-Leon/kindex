"""`kin mask scan`: find, and optionally mask, known secrets already stored.

New writes are masked on the way in (``privacy.redact_text``). Rows written
before ``secret_mask`` was configured keep their bytes until this maintenance
pass rewrites them. The scan walks every ordinary table of the live database
and of the node archives under ``<data_dir>/archive``, so a table added later
is covered without a list to maintain. Reports name key names and locations,
never values.

``--apply`` is a maintenance rewrite, not an edit: it takes a recovery
snapshot of each database first (fail-closed, like automated merges), rewrites
the matching cells in one transaction with ``secure_delete`` on, rebuilds the
FTS index so old tokens are gone, requeues changed nodes for embedding, logs a
``secret_mask_apply`` activity entry with counts per key name, and truncates
the WAL. Primary-key and foreign-key columns are reported, never rewritten:
changing an identifier would break every reference to it.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .secret_mask import SecretMask

if TYPE_CHECKING:
    from .store import Store

REASON = "secret-mask-apply"


class MaskScanReport(BaseModel):
    """What a scan found (and, with apply, rewrote). Never holds a value."""

    names: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    by_key: dict[str, int] = Field(default_factory=dict)
    by_location: dict[str, int] = Field(default_factory=dict)
    unmaskable: dict[str, int] = Field(default_factory=dict)
    rows: int = 0
    applied: bool = False
    snapshots: list[str] = Field(default_factory=list)
    skipped_tables: list[str] = Field(default_factory=list)


class _Hit(BaseModel):
    table: str
    rowid: int
    columns: dict[str, str]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    virtual = [name for name, sql in rows
               if (sql or "").lstrip().upper().startswith("CREATE VIRTUAL TABLE")]
    # A virtual table's shadow tables are derived (FTS is rebuilt on apply).
    return [name for name, _ in rows
            if not name.startswith("sqlite_") and name not in virtual
            and not any(name.startswith(v + "_") for v in virtual)]


def _scan(conn: sqlite3.Connection, mask: SecretMask, label: str,
          report: MaskScanReport) -> list[_Hit]:
    """Collect masked replacements for one database, tallying the report."""
    hits: list[_Hit] = []
    by_key: Counter[str] = Counter(report.by_key)
    by_location: Counter[str] = Counter(report.by_location)
    unmaskable: Counter[str] = Counter(report.unmaskable)
    prefix = "" if label == "live" else f"{label}:"
    for table in _tables(conn):
        info = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        columns = [row[1] for row in info]
        fixed = {row[1] for row in info if row[5]}
        fixed |= {row[3] for row in conn.execute(
            f"PRAGMA foreign_key_list({_quote(table)})").fetchall()}
        select = ", ".join(_quote(c) for c in columns)
        try:
            cursor = conn.execute(f"SELECT rowid, {select} FROM {_quote(table)}")
        except sqlite3.OperationalError:
            report.skipped_tables.append(prefix + table)  # WITHOUT ROWID
            continue
        for row in cursor:
            changed: dict[str, str] = {}
            for column, value in zip(columns, row[1:]):
                if not isinstance(value, str):
                    continue
                counts = mask.counts(value)
                if not counts:
                    continue
                location = f"{prefix}{table}.{column}"
                if column in fixed:
                    unmaskable[location] += 1
                    continue
                by_key.update(counts)
                by_location[location] += 1
                changed[column] = mask.mask(value)
            if changed:
                hits.append(_Hit(table=table, rowid=row[0], columns=changed))
    report.by_key = dict(by_key.most_common())
    report.by_location = dict(sorted(by_location.items()))
    report.unmaskable = dict(sorted(unmaskable.items()))
    report.rows += len(hits)
    return hits


def _rewrite(conn: sqlite3.Connection, hits: list[_Hit]) -> None:
    for hit in hits:
        sets = ", ".join(f"{_quote(c)} = ?" for c in hit.columns)
        conn.execute(f"UPDATE {_quote(hit.table)} SET {sets} WHERE rowid = ?",
                     (*hit.columns.values(), hit.rowid))


def _archives(store: "Store") -> list[Path]:
    directory = store.config.data_path / "archive"
    return sorted(directory.glob("*.db")) if directory.is_dir() else []


def scan(store: "Store", mask: SecretMask, *, apply: bool = False) -> MaskScanReport:
    """Count (and with ``apply`` rewrite) stored occurrences of known secrets."""
    report = MaskScanReport(names=mask.names(), files=[str(f) for f in mask.files])
    archives = _archives(store)
    report.databases = [str(store.db_path), *map(str, archives)]
    live_hits = _scan(store.conn, mask, "live", report)
    archive_hits: list[Path] = []
    for path in archives:
        with closing(sqlite3.connect(str(path))) as conn:
            found = _scan(conn, mask, f"archive/{path.name}", report)
        if found:
            archive_hits.append(path)
    if not apply or report.rows == 0:
        return report
    _apply_live(store, mask, report, bool(live_hits))
    for path in archive_hits:
        _apply_archive(path, mask, report)
    report.applied = True
    return report


def _apply_live(store: "Store", mask: SecretMask, report: MaskScanReport,
                found: bool) -> None:
    if not found:
        return
    from .snapshots import snapshot_db

    # Fail-closed: no restore point, no rewrite.
    report.snapshots.append(str(snapshot_db(store, REASON)))
    conn = store.conn
    conn.commit()
    conn.execute("PRAGMA secure_delete = ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Rescan under the write lock so the rewrite matches what is stored now.
        hits = _scan(conn, mask, "live", MaskScanReport())
        _rewrite(conn, hits)
        node_ids = [_node_id(conn, h.rowid) for h in hits if h.table == "nodes"]
        if node_ids:
            conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        store._log_in_transaction(
            conn, "secret_mask_apply", details={
                "rows": len(hits), "keys": report.by_key,
                "snapshot": report.snapshots[-1]})
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    from .vectors import enqueue_embeddings

    enqueue_embeddings(store, [n for n in node_ids if n])
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass  # a concurrent reader holds the WAL; the next checkpoint overwrites it


def _node_id(conn: sqlite3.Connection, rowid: int) -> str | None:
    row = conn.execute("SELECT id FROM nodes WHERE rowid = ?", (rowid,)).fetchone()
    return row[0] if row else None


def _apply_archive(path: Path, mask: SecretMask, report: MaskScanReport) -> None:
    from .snapshots import snapshot_connection

    conn = sqlite3.connect(str(path))
    try:
        report.snapshots.append(str(snapshot_connection(path, conn, REASON)))
        conn.execute("PRAGMA secure_delete = ON")
        conn.execute("BEGIN IMMEDIATE")
        try:
            _rewrite(conn, _scan(conn, mask, f"archive/{path.name}", MaskScanReport()))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()


def format_report(report: MaskScanReport) -> str:
    """Human summary: key names, counts and locations only."""
    lines = [f"Secret masking: {len(report.names)} key name(s) from "
             f"{len(report.files)} file(s)"]
    if not report.rows and not report.unmaskable:
        lines.append("No stored occurrences of known secret values.")
        return "\n".join(lines)
    verb = "Masked" if report.applied else "Found"
    lines.append(f"{verb} {report.rows} row(s) containing known secret values:")
    lines.extend(f"  ${name}: {count} occurrence(s)" for name, count in report.by_key.items())
    if report.by_location:
        lines.append("By location (rows):")
        lines.extend(f"  {loc}: {count}" for loc, count in report.by_location.items())
    if report.unmaskable:
        lines.append("Identifiers containing a known secret (not rewritten; rename by hand):")
        lines.extend(f"  {loc}: {count}" for loc, count in report.unmaskable.items())
    if report.skipped_tables:
        lines.append("Skipped tables: " + ", ".join(report.skipped_tables))
    if report.applied:
        lines.append("Snapshots taken before the rewrite (they still hold the old "
                     "values; delete them once you have checked the result):")
        lines.extend(f"  {path}" for path in report.snapshots)
    elif report.rows:
        lines.append("Run `kin mask scan --apply` to rewrite them as $NAME.")
    return "\n".join(lines)
