import os
import sys

import pytest

from opera_agent import keys as K


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in list(K.KNOWN) + ["OPERA_KEYS_FILE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)                                      # no stray ./api_keys.txt
    monkeypatch.setattr(K, "DEFAULT_LOCATIONS", (tmp_path / "api_keys.txt", tmp_path / "home" / "api_keys.txt"))


def write(path, text, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def test_parse_label_and_env_styles(tmp_path):
    f = write(tmp_path / "k.txt", """
# comment
GEMINI API KEY: AQ.fake-gemini
OPENROUTER API KEY:  sk-or-v1-fake   # trailing comment
export OPERA_YOLO_MODEL="/models/best.pt"
GEMINI_MODEL=gemini-3.1-pro
""")
    assert K.parse_keys_file(f) == {"GEMINI_API_KEY": "AQ.fake-gemini", "OPENROUTER_API_KEY": "sk-or-v1-fake",
                                    "OPERA_YOLO_MODEL": "/models/best.pt", "GEMINI_MODEL": "gemini-3.1-pro"}


def test_template_parses_to_nothing():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert K.parse_keys_file(os.path.join(root, "api_keys.example.txt")) == {}


def test_unknown_label_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="GEMNI"):
        K.parse_keys_file(write(tmp_path / "k.txt", "GEMNI KEY: x\n"))


def test_load_sets_env_but_env_wins(tmp_path, monkeypatch):
    f = write(tmp_path / "k.txt", "GEMINI API KEY: from-file\nOPENROUTER API KEY: or-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    msgs = []
    assert K.load_keys(f, log=msgs.append) == ["GEMINI_API_KEY"]
    assert os.environ["GEMINI_API_KEY"] == "from-file" and os.environ["OPENROUTER_API_KEY"] == "from-env"
    assert "from-file" not in " ".join(msgs)                        # values are never printed


def test_search_order(tmp_path, monkeypatch):
    write(tmp_path / "home" / "api_keys.txt", "GEMINI API KEY: home\n")
    assert K.find_keys_file() == tmp_path / "home" / "api_keys.txt"
    write(tmp_path / "api_keys.txt", "GEMINI API KEY: cwd\n")
    assert K.find_keys_file() == tmp_path / "api_keys.txt"
    monkeypatch.setenv("OPERA_KEYS_FILE", str(write(tmp_path / "job.txt", "GEMINI API KEY: job\n")))
    assert K.find_keys_file() == tmp_path / "job.txt"
    assert K.find_keys_file(tmp_path / "home" / "api_keys.txt") == tmp_path / "home" / "api_keys.txt"
    with pytest.raises(FileNotFoundError):
        K.find_keys_file(tmp_path / "missing.txt")
    assert K.load_keys(None, log=lambda *_: None) == ["GEMINI_API_KEY"] and os.environ["GEMINI_API_KEY"] == "job"


def test_no_file_is_fine():
    assert K.load_keys(None, log=lambda *_: None) == []


@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_warns_when_others_can_read(tmp_path):
    msgs = []
    K.load_keys(write(tmp_path / "k.txt", "GEMINI API KEY: x\n", mode=0o644), log=msgs.append)
    assert any("chmod 600" in m for m in msgs)


def test_keys_file_arg_is_removed_for_any_subcommand():
    argv = ["opera_agent", "process", "v", "o", "--keys-file", "/k.txt", "--interval", "2"]
    assert K.pop_keys_file_arg(argv) == "/k.txt" and argv == ["opera_agent", "process", "v", "o", "--interval", "2"]
    argv = ["opera_rag", "--keys-file=/k.txt", "ask", "q"]
    assert K.pop_keys_file_arg(argv) == "/k.txt" and argv == ["opera_rag", "ask", "q"]


def test_cli_reads_yolo_model_from_keys_file(tmp_path, monkeypatch, capsys):
    f = write(tmp_path / "k.txt", "OPERA_YOLO_MODEL=/models/best.pt\n")
    monkeypatch.setattr(sys, "argv", ["opera_agent", "classify", "-h", "--keys-file", str(f)])
    from opera_agent.__main__ import main
    with pytest.raises(SystemExit):
        main()
    assert os.environ["OPERA_YOLO_MODEL"] == "/models/best.pt"
