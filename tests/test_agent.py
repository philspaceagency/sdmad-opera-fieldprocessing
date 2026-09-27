import json

import pytest

from opera_agent import llm
from opera_agent.agent import OperaAgent
from opera_agent.llm import OpenAICompatBackend, TransientLLMError


class Scripted:
    """Fake backend: raises `fail` errors first, then plays back scripted replies."""
    def __init__(self, name, replies=(), fail=0):
        self.name, self.replies, self.fail, self.seen = name, list(replies), fail, []

    def complete(self, system, messages, tools, max_tokens=4000):
        self.seen.append([dict(m) for m in messages])
        if self.fail:
            self.fail -= 1
            raise TransientLLMError(f"{self.name}: 429 RESOURCE_EXHAUSTED")
        return self.replies.pop(0)


def call(name, args, i="c1"):
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    return {"text": "", "tool_calls": [{"id": i, "name": name, "args": args}], "message": msg}


def answer(text):
    return {"text": text, "tool_calls": [], "message": {"role": "assistant", "content": text}}


def agent(primary, fallback=None, **kw):
    a = OperaAgent(backend=primary, fallback_backends=[fallback] if fallback else [], rag=False, verbose=False, **kw)
    a._sleep = lambda s: None
    return a


def test_tool_loop(tmp_path):
    (tmp_path / "13MAY2025").mkdir()
    g = Scripted("gemini", [call("list_folder", {"path": str(tmp_path)}), answer("found 13MAY2025")])
    a = agent(g)
    assert a.run("find the survey") == "found 13MAY2025"
    tool_msg = g.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert "13MAY2025" in tool_msg["content"]


def test_retries_then_succeeds_on_primary():
    g = Scripted("gemini", [answer("ok")], fail=2)
    q = Scripted("qwen", [answer("from qwen")])
    assert agent(g, q, retries=2).run("hi") == "ok"
    assert q.seen == []


class BusyAfterFirst(Scripted):
    """Answers once, then keeps returning 503."""
    def complete(self, system, messages, tools, max_tokens=4000):
        if self.seen:
            self.seen.append(None)
            raise TransientLLMError(f"{self.name}: 503 UNAVAILABLE")
        return super().complete(system, messages, tools, max_tokens)


def test_falls_back_to_qwen_mid_conversation(tmp_path):
    g = BusyAfterFirst("gemini", [call("list_folder", {"path": str(tmp_path)})])
    q = Scripted("qwen", [answer("done on qwen")])
    assert agent(g, q, retries=1).run("find the survey") == "done on qwen"
    assert len(g.seen) == 3                      # first answer, then 1 try + 1 retry
    history = q.seen[0]                          # qwen got the whole conversation, including the tool result
    assert [m["role"] for m in history] == ["user", "assistant", "tool"]


def test_next_request_tries_primary_again():
    g = Scripted("gemini", [answer("gemini back")], fail=3)
    q = Scripted("qwen", [answer("qwen 1")])
    a = agent(g, q, retries=2)
    assert a.run("first") == "qwen 1"
    assert a.run("second") == "gemini back"


def test_no_fallback_reports_unavailable():
    a = agent(Scripted("gemini", fail=10), None, retries=1)
    assert "No model available" in a.run("hi")


# ---------------------------------------------------------------- backends
def test_openrouter_request_and_parse(monkeypatch):
    b = OpenAICompatBackend(api_key="test-key")
    sent = {}

    def fake_post(payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "inspect_survey",
                                                          "arguments": '{"videos_dir": "/d"}'}}]}}]}
    monkeypatch.setattr(b, "_post", fake_post)
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "_gemini": object(),
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_folder", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "list_folder", "content": "{}", "_gemini_id": None}]
    res = b.complete("sys", msgs, [{"name": "list_folder", "description": "d", "parameters": {"type": "object"}}])
    assert sent["model"] == llm.DEFAULT_QWEN
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert all(not k.startswith("_") for m in sent["messages"] for k in m)       # private keys stripped
    assert "name" not in sent["messages"][3]
    assert sent["tools"][0]["type"] == "function"
    assert res["tool_calls"] == [{"id": "t1", "name": "inspect_survey", "args": {"videos_dir": "/d"}}]


def test_openrouter_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatBackend()


def test_openrouter_upstream_rate_limit_is_transient(monkeypatch):
    b = OpenAICompatBackend(api_key="k")
    monkeypatch.setattr(b, "_post", lambda p: {"error": {"code": 429, "message": "rate limited"}})
    with pytest.raises(TransientLLMError):
        b.complete("s", [{"role": "user", "content": "x"}], [])


def test_gemini_history_conversion():
    pytest.importorskip("google.genai")
    from google.genai import types
    b = llm.GeminiBackend.__new__(llm.GeminiBackend)
    own = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="a", args={}))])
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "_gemini": own, "tool_calls": []},
            {"role": "tool", "tool_call_id": "c1", "name": "a", "content": '{"x": 1}', "_gemini_id": None},
            {"role": "assistant", "content": "qwen text", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "b", "arguments": '{"p": 2}'}}]},
            {"role": "tool", "tool_call_id": "t1", "name": "b", "content": '{"y": 2}'},
            {"role": "tool", "tool_call_id": "t2", "name": "b", "content": '{"y": 3}'}]
    c = b._contents(msgs)
    assert [x.role for x in c] == ["user", "model", "user", "model", "user"]
    assert c[1] is own                                            # Gemini's own turn resent untouched
    assert c[2].parts[0].function_response.response == {"result": {"x": 1}}
    assert c[3].parts[1].function_call.args == {"p": 2}
    assert len(c[4].parts) == 2                                   # consecutive tool results grouped


# ---------------------------------------------------------------- model chain
class Retired(Scripted):
    def complete(self, system, messages, tools, max_tokens=4000):
        self.seen.append(None)
        raise llm.ModelUnavailableError(f"{self.name}: 404 NOT_FOUND model is no longer available")


def test_retired_model_is_skipped_without_retries():
    old = Retired("gemini-2.5-flash")
    lite = Scripted("gemini-3.5-flash-lite", fail=1, replies=[answer("lite answered")])
    q = Scripted("qwen", [answer("qwen")])
    a = OperaAgent(backend=old, fallback_backends=[lite, q], rag=False, verbose=False, retries=2)
    a._sleep = lambda s: None
    assert a.run("hi") == "lite answered"
    assert len(old.seen) == 1 and q.seen == []            # 404: no retries, straight to the next model


def test_default_chain(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    a = OperaAgent(rag=False, verbose=False)
    assert [a.backend.name] + [b.name for b in a.fallbacks] == [
        "gemini:gemini-3.5-flash", "gemini:gemini-3.5-flash-lite", f"openrouter:{llm.DEFAULT_QWEN}"]
    a = OperaAgent(rag=False, verbose=False, llm_model="gemini-3.1-pro", gemini_fallbacks=[], fallback=None)
    assert a.backend.name == "gemini:gemini-3.1-pro" and a.fallbacks == []
    monkeypatch.delenv("OPENROUTER_API_KEY")
    a = OperaAgent(rag=False, verbose=False, llm_model="gemini-3.5-flash-lite")
    assert a.fallbacks == []                              # the lite fallback isn't added twice
