"""The terminal demo: lists its steps and runs the quick ones end to end."""
import subprocess
import sys

from tests.conftest import ROOT


def run_demo(*args):
    return subprocess.run([sys.executable, str(ROOT / "demo.py"), "--no-pause", "--no-colour", *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=600, cwd=ROOT)


def test_demo_lists_every_step():
    out = run_demo("--list")
    assert out.returncode == 0
    for word in ("Unit 1", "Unit 2", "Unit 3", "Unit 5 (1/4)", "Unit 5 (4/4)", "Summary"):
        assert word in out.stdout


def test_demo_runs_the_quick_steps():
    out = run_demo("--step", "0", "--step", "1", "--step", "11")
    assert out.returncode == 0, out.stderr
    assert "Traceback" not in out.stdout + out.stderr
    assert "55 features" in out.stdout and "lesion fraction" in out.stdout
    assert "graphs in total" in out.stdout


def test_demo_rejects_unknown_step():
    assert run_demo("--step", "99").returncode == 2


def test_terminal_line_chart_without_colour(capsys, monkeypatch):
    """Charts must also work when the output is not a terminal (piped, logged) and in ASCII mode."""
    import demo
    monkeypatch.setattr(demo, "USE_COLOUR", False)
    for ascii_mode in (False, True):
        monkeypatch.setattr(demo.ARGS, "ascii", ascii_mode)
        demo.line_chart({"train": [1, 2, 3, 4], "valid": [1, 1.5, 2, 2.2]}, "title", "x", "y")
        out = capsys.readouterr().out
        assert "\x1b[" not in out and "title" in out
