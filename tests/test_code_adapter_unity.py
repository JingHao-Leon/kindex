"""Tests for opt-in Unity asset ingestion (issue #17)."""
from __future__ import annotations

import pytest

from kindex.adapters import code
from kindex.config import Config
from kindex.store import Store

_GUID = "0123456789abcdef0123456789abcdef"
_UNITY_YAML = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &123\nGameObject:\n"


def _make_unity_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "Assets").mkdir(parents=True)
    (repo / "Scripts").mkdir()
    (repo / "Assets" / "Player.prefab").write_text(_UNITY_YAML)
    (repo / "Assets" / "Player.prefab.meta").write_text(
        f"fileFormatVersion: 2\nguid: {_GUID}\n"
    )
    (repo / "Scripts" / "main.cs").write_text("class Main {}\n")
    (repo / "Scripts" / "main.cs.meta").write_text(
        "fileFormatVersion: 2\nguid: fedcba9876543210fedcba9876543210\n"
    )
    return repo


@pytest.fixture
def local_only(monkeypatch):
    monkeypatch.setattr(code, "_run_ctags", lambda *_args: [])
    monkeypatch.setattr(code, "_check_cscope", lambda: False)
    monkeypatch.setattr(code, "_check_treesitter", lambda _lang: None)


@pytest.fixture
def store(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "kindex"))
    s = Store(cfg)
    yield s
    s.close()


def _modules(store):
    return {
        node["title"]: node
        for node in store.all_nodes(node_type="artifact", limit=100)
    }


def test_unity_files_excluded_by_default(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    result = code.ingest_code(store, repo)
    assert result.errors == []
    assert set(_modules(store)) == {"Scripts/main.cs"}


def test_unity_opt_in_creates_asset_modules_with_guids(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []

    modules = _modules(store)
    assert set(modules) == {"Assets/Player.prefab", "Scripts/main.cs"}
    assert not any(title.endswith(".meta") for title in modules)

    prefab = modules["Assets/Player.prefab"]["extra"]
    assert prefab["language"] == "Unity Prefab"
    assert prefab["serialization"] == "text"
    assert prefab["unity_guid"] == _GUID

    # Code files are also referenced by GUID in Unity, so scripts get one too
    script = modules["Scripts/main.cs"]["extra"]
    assert script["unity_guid"] == "fedcba9876543210fedcba9876543210"
    assert "serialization" not in script


def test_unity_binary_asset_is_sniffed_not_parsed(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Assets").mkdir(parents=True)
    (repo / "Assets" / "Terrain.asset").write_bytes(b"\x00\x01\x02binarystuff")

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []

    terrain = _modules(store)["Assets/Terrain.asset"]["extra"]
    assert terrain["serialization"] == "binary"
    assert terrain["line_count"] == 0


def test_unity_excludes_library_and_temp(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Library").mkdir(parents=True)
    (repo / "Temp").mkdir()
    (repo / "Assets").mkdir()
    (repo / "Library" / "x.asset").write_text(_UNITY_YAML)
    (repo / "Temp" / "y.asset").write_text(_UNITY_YAML)
    (repo / "Assets" / "keep.asset").write_text(_UNITY_YAML)

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []
    assert set(_modules(store)) == {"Assets/keep.asset"}


def test_include_extensions_generic(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Water.shader").write_text("Shader \"Custom/Water\" {}\n")

    result = code.ingest_code(
        store, repo, extra_extensions={".shader": "Unity Shader"},
    )
    assert result.errors == []
    water = _modules(store)["Water.shader"]["extra"]
    assert water["language"] == "Unity Shader"


def test_adapter_reads_code_ingest_config(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    cfg = Config(
        data_dir=str(tmp_path / "kindex"),
        code_ingest={"unity": True},
    )

    result = code.adapter.ingest(store, directory=str(repo), _config=cfg)
    assert result.errors == []
    assert "Assets/Player.prefab" in _modules(store)


def test_unity_excludes_nested_library_but_not_asset_folders(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Games" / "ProjA" / "Library").mkdir(parents=True)
    (repo / "Games" / "ProjA" / "Assets").mkdir()
    (repo / "Assets" / "MyLibrary").mkdir(parents=True)
    (repo / "Games" / "ProjA" / "Library" / "x.asset").write_text(_UNITY_YAML)
    (repo / "Games" / "ProjA" / "Assets" / "keep.asset").write_text(_UNITY_YAML)
    (repo / "Assets" / "MyLibrary" / "thing.asset").write_text(_UNITY_YAML)

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []
    assert set(_modules(store)) == {
        "Games/ProjA/Assets/keep.asset",
        "Assets/MyLibrary/thing.asset",
    }


def test_target_directory_kin_config_enables_unity(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    kin_dir = repo / ".kin"
    kin_dir.mkdir()
    (kin_dir / "config").write_text("code_ingest:\n  unity: true\n")

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert "Assets/Player.prefab" in _modules(store)


def test_unity_guid_refresh_on_unchanged_asset(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    first = code.ingest_code(store, repo, unity=True)
    assert first.errors == []

    new_guid = "ffffffffffffffffffffffffffffffff"
    (repo / "Assets" / "Player.prefab.meta").write_text(
        f"fileFormatVersion: 2\nguid: {new_guid}\n"
    )

    second = code.ingest_code(store, repo, unity=True)
    assert second.errors == []
    prefab = _modules(store)["Assets/Player.prefab"]["extra"]
    assert prefab["unity_guid"] == new_guid


def test_adapter_flag_overrides_config_off(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    cfg = Config(
        data_dir=str(tmp_path / "kindex"),
        code_ingest={"unity": True},
    )

    result = code.adapter.ingest(
        store, directory=str(repo), _config=cfg, unity=False,
    )
    assert result.errors == []
    assert "Assets/Player.prefab" not in _modules(store)


def _make_vendored_unity_repo(tmp_path):
    """A Unity repo with a vendored Licensed/ tree, a material, a prefab,
    and files the built-in excludes already drop (tests/, Library/)."""
    repo = _make_unity_repo(tmp_path)
    (repo / "Assets" / "Licensed" / "Vendor").mkdir(parents=True)
    (repo / "Assets" / "Licensed" / "Vendor" / "Rock.asset").write_text(_UNITY_YAML)
    (repo / "Assets" / "Licensed" / "Vendor" / "Rock.cs").write_text("class Rock {}\n")
    (repo / "Assets" / "Stone.mat").write_text(_UNITY_YAML)
    (repo / "Assets" / "Keep.asset").write_text(_UNITY_YAML)
    (repo / "Library").mkdir()
    (repo / "Library" / "cache.asset").write_text(_UNITY_YAML)
    (repo / "tests").mkdir()
    (repo / "tests" / "helper.cs").write_text("class Helper {}\n")
    return repo


_VENDORED_EXCLUDES = "code_ingest:\n  unity: true\n  exclude:\n" \
    "    - 'Assets/Licensed/*'\n    - '*.mat'\n    - '*.prefab'\n"


def test_kin_config_exclude_drops_licensed_materials_and_prefabs(
        tmp_path, local_only, store):
    repo = _make_vendored_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(_VENDORED_EXCLUDES)

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    # Additive: the config patterns drop Licensed/, .mat and .prefab, and the
    # built-in (tests/) and Unity (Library/) excludes still apply.
    assert set(_modules(store)) == {"Assets/Keep.asset", "Scripts/main.cs"}


def test_loaded_config_exclude_reaches_the_adapter(tmp_path, local_only, store):
    repo = _make_vendored_unity_repo(tmp_path)
    cfg = Config(
        data_dir=str(tmp_path / "kindex"),
        code_ingest={"unity": True, "exclude": ["*.mat", "*/Licensed/*"]},
    )

    result = code.adapter.ingest(store, directory=str(repo), _config=cfg)
    assert result.errors == []
    assert set(_modules(store)) == {
        "Assets/Keep.asset", "Assets/Player.prefab", "Scripts/main.cs",
    }


def _git_init(path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(path)], check=True)


@pytest.mark.parametrize("subdir", ["", "client"])
def test_config_exclude_is_relative_to_the_repo_root(
        tmp_path, local_only, store, subdir):
    # The same pattern must mean the same files whether `kin ingest code`
    # targets the repo root or a subdirectory of it.
    repo = tmp_path / "mono"
    client = repo / "client"
    (client / "Assets" / "Licensed").mkdir(parents=True)
    (client / "Assets" / "Licensed" / "Rock.asset").write_text(_UNITY_YAML)
    (client / "Assets" / "Keep.asset").write_text(_UNITY_YAML)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        "code_ingest:\n  unity: true\n  exclude:\n    - 'client/Assets/Licensed/*'\n"
    )
    _git_init(repo)

    target = repo / subdir if subdir else repo
    result = code.adapter.ingest(store, directory=str(target))
    assert result.errors == []
    assert set(_modules(store)) == {"client/Assets/Keep.asset"}


def test_target_config_excludes_include_inherited_patterns(
        tmp_path, local_only, store):
    # A child .kin/config with its own code_ingest section must not hide the
    # excludes it inherits: the lists union, as they do in load_config.
    org = tmp_path / "org" / ".kin"
    org.mkdir(parents=True)
    (org / "config").write_text(
        "code_ingest:\n  exclude:\n    - 'Assets/Licensed/*'\n"
    )
    repo = _make_vendored_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        f"inherits:\n  - {org / 'config'}\n"
        "code_ingest:\n  unity: true\n  exclude:\n    - '*.mat'\n"
    )

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert set(_modules(store)) == {
        "Assets/Keep.asset", "Assets/Player.prefab", "Scripts/main.cs",
    }


@pytest.mark.parametrize("parent_exclude, child_exclude", [
    ("'Assets/Licensed/*'", "\n    - '*.mat'"),
    ("\n    - 'Assets/Licensed/*'", "'*.mat'"),
])
def test_inherited_excludes_union_across_scalar_and_list_forms(
        tmp_path, local_only, store, parent_exclude, child_exclude):
    # A bare-string exclude is one pattern in any layer; it must union with
    # a list in another layer instead of replacing it.
    org = tmp_path / "org" / ".kin"
    org.mkdir(parents=True)
    (org / "config").write_text(f"code_ingest:\n  exclude: {parent_exclude}\n")
    repo = _make_vendored_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        f"inherits:\n  - {org / 'config'}\n"
        f"code_ingest:\n  unity: true\n  exclude: {child_exclude}\n"
    )

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert set(_modules(store)) == {
        "Assets/Keep.asset", "Assets/Player.prefab", "Scripts/main.cs",
    }


def test_empty_config_exclude_keeps_unity_and_builtin_excludes(
        tmp_path, local_only, store):
    # `exclude:` with no value is YAML null. It must read as no extra
    # patterns, not invalidate the section and silently switch unity off.
    repo = _make_vendored_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text("code_ingest:\n  unity: true\n  exclude:\n")

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert set(_modules(store)) == {
        "Assets/Keep.asset", "Assets/Licensed/Vendor/Rock.asset",
        "Assets/Licensed/Vendor/Rock.cs", "Assets/Player.prefab",
        "Assets/Stone.mat", "Scripts/main.cs",
    }


def test_empty_child_exclude_keeps_inherited_patterns(tmp_path, local_only, store):
    # An empty child `exclude:` adds nothing; it must not erase the parent's.
    org = tmp_path / "org" / ".kin"
    org.mkdir(parents=True)
    (org / "config").write_text("code_ingest:\n  exclude:\n    - '*.mat'\n")
    repo = _make_vendored_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        f"inherits:\n  - {org / 'config'}\n"
        "code_ingest:\n  unity: true\n  exclude:\n"
    )

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert "Assets/Stone.mat" not in _modules(store)
    assert "Assets/Keep.asset" in _modules(store)


def test_target_config_inherits_unity_opt_in(tmp_path, local_only, store):
    # The target's .kin/config is read through its inherits chain, as
    # load_config reads it, so a parent's unity opt-in reaches the child.
    org = tmp_path / "org" / ".kin"
    org.mkdir(parents=True)
    (org / "config").write_text("code_ingest:\n  unity: true\n")
    repo = _make_unity_repo(tmp_path)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(f"inherits:\n  - {org / 'config'}\n")

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert "Assets/Player.prefab" in _modules(store)
