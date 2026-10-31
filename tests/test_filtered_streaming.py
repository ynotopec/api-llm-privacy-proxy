import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import app


class EmailSanitizer:
    def __init__(self):
        self.values = []

    async def sanitize_payload(self, value):
        self.values.append(value)
        stats = app.RedactionStats()

        def walk(item):
            if isinstance(item, str) and "alice@example.com" in item:
                stats.add("private_email", 3)
                return item.replace("alice@example.com", "[PRIVATE_EMAIL_1]")
            if isinstance(item, list):
                return [walk(child) for child in item]
            if isinstance(item, dict):
                return {key: walk(child) for key, child in item.items()}
            return item

        return walk(value), stats


def events(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    return [json.loads(frame) for frame in frames[:-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_usage", [True, False])
async def test_chat_stream_filters_complete_output_and_preserves_tool_calls(monkeypatch, include_usage):
    sanitizer = EmailSanitizer()
    monkeypatch.setattr(app, "sanitizer", sanitizer)
    email_fragments = ["alice", "@exam", "ple.com"]

    async def forward(req, full_path, sanitized_payload, stream=False):
        assert stream is False
        assert sanitized_payload["stream"] is False
        assert "stream_options" not in sanitized_payload
        assert sanitized_payload["model"] == "test"
        assert sanitized_payload["messages"][0]["content"] == "Contact [PRIVATE_EMAIL_1]"
        return app.JSONResponse({
            "id": "chatcmpl-test", "created": 123, "model": "test", "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {
                    "role": "assistant", "content": "Écrire à " + "".join(email_fragments),
                    "reasoning_content": "Contact alice@example.com",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {
                        "name": "send", "arguments": '{"to":"alice@example.com"}',
                    }}],
                }, "finish_reason": "tool_calls"},
                {"index": 1, "message": {"role": "assistant", "content": "Autre réponse"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
        }, headers={"etag": "original-body"})

    monkeypatch.setattr(app, "forward_request", forward)
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "test-anonym", "stream": True, "stream_options": {"include_usage": include_usage},
            "messages": [{"role": "user", "content": "Contact alice@example.com"}],
        })
    assert response.status_code == 200
    assert "alice@example.com" not in response.text
    assert "etag" not in response.headers
    assert "content-length" not in response.headers
    assert response.headers["x-privacy-stream-mode"] == "buffered"
    assert response.headers["x-privacy-filtered-output-spans"] == "3"
    assert len(sanitizer.values) == 2
    chunks = events(response)
    assert len(chunks) == (3 if include_usage else 2)
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["model"] == "test-anonym"
    assert chunks[0]["created"] == 123
    delta = chunks[0]["choices"][0]["delta"]
    assert delta["content"] == "Écrire à [PRIVATE_EMAIL_1]"
    assert delta["reasoning_content"] == "Contact [PRIVATE_EMAIL_1]"
    assert delta["tool_calls"][0]["index"] == 0
    assert delta["tool_calls"][0]["id"] == "call-1"
    assert json.loads(delta["tool_calls"][0]["function"]["arguments"]) == {"to": "[PRIVATE_EMAIL_1]"}
    assert chunks[0]["choices"][1]["delta"]["content"] == "Autre réponse"
    assert chunks[1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    if include_usage:
        assert chunks[2]["choices"] == []
        assert chunks[2]["usage"]["total_tokens"] == 13
    else:
        assert all("usage" not in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_legacy_completion_stream_is_filtered(monkeypatch):
    monkeypatch.setattr(app, "sanitizer", EmailSanitizer())

    async def forward(req, full_path, sanitized_payload, stream=False):
        assert full_path == "completions"
        assert stream is False
        return app.JSONResponse({"model": "test", "object": "text_completion", "id": "cmpl-test", "choices": [
            {"index": 0, "text": "alice@example.com", "finish_reason": "stop", "logprobs": None},
        ]})

    monkeypatch.setattr(app, "forward_request", forward)
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
        response = await client.post("/v1/completions", json={"model": "test", "stream": True, "prompt": "hello"})
    assert response.status_code == 200
    chunks = events(response)
    assert chunks[0]["choices"][0]["text"] == "[PRIVATE_EMAIL_1]"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["upstream_error", "invalid_json", "unexpected_sse", "invalid_choices", "filter_failure"])
async def test_stream_errors_never_emit_unfiltered_completion(monkeypatch, case):
    sanitizer = EmailSanitizer()
    monkeypatch.setattr(app, "sanitizer", sanitizer)

    async def forward(req, full_path, sanitized_payload, stream=False):
        if case == "upstream_error":
            return app.JSONResponse({"error": {"message": "alice@example.com"}}, status_code=429)
        if case == "invalid_json":
            return app.Response(content="alice@example.com", media_type="application/json")
        if case == "unexpected_sse":
            return app.Response(content="data: alice@example.com\n\n", media_type="text/event-stream")
        if case == "invalid_choices":
            return app.JSONResponse({"choices": "alice@example.com"})
        return app.JSONResponse({"choices": [{"message": {"content": "alice@example.com"}}]})

    if case == "filter_failure":
        original = sanitizer.sanitize_payload

        async def fail_output(value):
            if "choices" in value:
                raise app.HTTPException(status_code=503, detail="privacy_filter_busy")
            return await original(value)

        monkeypatch.setattr(sanitizer, "sanitize_payload", fail_output)
    monkeypatch.setattr(app, "forward_request", forward)
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test", "stream": True})
    assert response.status_code == (429 if case == "upstream_error" else 503 if case == "filter_failure" else 502)
    assert "alice@example.com" not in response.text
    assert not response.headers["content-type"].startswith("text/event-stream")


@pytest.mark.asyncio
async def test_filtered_stream_through_real_forwarder_and_shared_http_client(monkeypatch):
    monkeypatch.setattr(app, "sanitizer", EmailSanitizer())

    def upstream(request):
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(200, json={"model": "test", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "alice@example.com"}, "finish_reason": "stop"},
        ]})

    async with AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(app.app.state, "upstream_client", upstream_client, raising=False)
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json={"model": "test-anonym", "stream": True})
        assert response.status_code == 200
        assert events(response)[0]["choices"][0]["delta"]["content"] == "[PRIVATE_EMAIL_1]"
        assert not upstream_client.is_closed
