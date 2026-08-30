"""Regression test for Windows consoles that use a legacy text encoding."""

from pathlib import Path


def test_main_source_is_cp1252_console_safe():
    source = (Path(__file__).parent.parent / "src" / "main.py").read_text(encoding="utf-8")
    source.encode("cp1252")
