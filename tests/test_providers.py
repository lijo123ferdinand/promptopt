import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from promptopt import AnthropicProvider, LMError, OpenAICompatProvider
from promptopt.providers import make_provider


class FakeServer:
    """A local OpenAI-compatible endpoint that replays scripted responses."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                status, payload, headers = outer.script.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()


def ok(text, usage=True):
    payload = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 11, "completion_tokens": 3}
    return (200, payload, {})


def test_openai_compat_success_and_request_shape():
    srv = FakeServer([ok("positive")])
    try:
        p = OpenAICompatProvider("llama3.2", base_url=srv.url, api_key="test-key")
        c = p.complete("hello", max_tokens=7)
        assert (c.text, c.prompt_tokens, c.completion_tokens) == ("positive", 11, 3)
        r = srv.requests[0]
        assert r["path"] == "/v1/chat/completions" and r["auth"] == "Bearer test-key"
        assert r["body"] == {"model": "llama3.2", "messages": [{"role": "user", "content": "hello"}],
                             "max_tokens": 7, "temperature": 0.0}
    finally:
        srv.close()


def test_openai_compat_retries_then_succeeds_and_estimates_usage():
    srv = FakeServer([(503, {"error": "busy"}, {}), (429, {"error": "slow"}, {"Retry-After": "0"}), ok("x" * 40, usage=False)])
    sleeps = []
    try:
        p = OpenAICompatProvider("m", base_url=srv.url, api_key="", sleep=sleeps.append)
        c = p.complete("abcd" * 10)
        assert c.completion_tokens == 10 and c.prompt_tokens == 10
        assert len(sleeps) == 2 and len(srv.requests) == 3 and srv.requests[0]["auth"] is None
    finally:
        srv.close()


def test_openai_compat_does_not_retry_client_errors():
    srv = FakeServer([(400, {"error": "bad"}, {})])
    try:
        p = OpenAICompatProvider("m", base_url=srv.url, api_key="", sleep=lambda s: None)
        with pytest.raises(LMError) as e:
            p.complete("x")
        assert e.value.status == 400 and not e.value.retryable and len(srv.requests) == 1
    finally:
        srv.close()


def test_openai_compat_connection_error_is_retryable():
    p = OpenAICompatProvider("m", base_url="http://127.0.0.1:9/v1", api_key="", max_retries=1, sleep=lambda s: None, timeout=2)
    with pytest.raises(LMError) as e:
        p.complete("x")
    assert e.value.retryable


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        resp = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="neg"),
                     SimpleNamespace(type="text", text="ative")],
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )

        def create(**kw):
            self.calls.append(kw)
            return resp

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_anthropic_provider_with_fallback_beta():
    fake = FakeAnthropic()
    p = AnthropicProvider(client=fake)
    c = p.complete("prompt", max_tokens=10)
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.model) == ("negative", 20, 5, "claude-sonnet-5-5")
    kw = fake.calls[0]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw and kw["max_tokens"] >= 10


def test_anthropic_provider_plain_and_refusal():
    fake = FakeAnthropic()
    AnthropicProvider(client=fake, refusal_fallback=False).complete("p")
    assert "betas" not in fake.calls[0]
    with pytest.raises(LMError, match="refusal"):
        AnthropicProvider(client=FakeAnthropic("refusal")).complete("p")


def test_make_provider():
    assert make_provider("mock").model == "mock-1"
    with pytest.raises(ValueError):
        make_provider("openai")
    with pytest.raises(ValueError):
        make_provider("nope")
