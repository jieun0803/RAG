import os

import pytest

from src.env_loader import (
    UPSTAGE_KEY,
    EnvironmentLoader,
    get_api_key,
    load_environment,
    set_api_key,
)


def test_parse_file_skips_comments_and_empty(tmp_path):
    p = tmp_path / "e.env"
    p.write_text(
        "\n# c\n\nFOO=bar\n  BAZ  =  quux  \nKEY=no=equals? only first split\n",
        encoding="utf-8",
    )
    d = EnvironmentLoader.parse_file(p)
    assert d["FOO"] == "bar"
    assert d["BAZ"] == "quux"
    assert d["KEY"] == "no=equals? only first split"


def test_parse_file_strips_quotes(tmp_path):
    p = tmp_path / "e.env"
    p.write_text('X="hello"\nY=\'world\'\n', encoding="utf-8")
    d = EnvironmentLoader.parse_file(p)
    assert d["X"] == "hello"
    assert d["Y"] == "world"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv(UPSTAGE_KEY, raising=False)
    with pytest.raises(ValueError, match="not set"):
        get_api_key()


def test_get_api_key_placeholder(monkeypatch):
    monkeypatch.setenv(UPSTAGE_KEY, "your_upstage_api_key_here")
    with pytest.raises(ValueError, match="placeholder"):
        get_api_key()


def test_set_and_get_api_key(monkeypatch):
    monkeypatch.delenv(UPSTAGE_KEY, raising=False)
    set_api_key("sk-test-real-looking-key")
    assert get_api_key() == "sk-test-real-looking-key"


def test_load_environment_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv(UPSTAGE_KEY, raising=False)
    cfg = tmp_path / "config.env"
    cfg.write_text(f"{UPSTAGE_KEY}=sk-from-file\n", encoding="utf-8")
    load_environment(cfg, strict=True)
    assert get_api_key() == "sk-from-file"


def test_env_precedence_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv(UPSTAGE_KEY, "sk-from-env")
    cfg = tmp_path / "config.env"
    cfg.write_text(f"{UPSTAGE_KEY}=sk-from-file\n", encoding="utf-8")
    load_environment(cfg, strict=True)
    assert get_api_key() == "sk-from-env"
