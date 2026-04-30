import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.responses import JSONResponse

import app as privacy_app


class FakeSanitizer:
    async def sanitize_payload(self, payload):
        stats = privacy_app.RedactionStats()

        def walk(v):
            if isinstance(v, str):
                changed = v
                if "alice@example.com" in changed:
                    changed = changed.replace("alice@example.com", "[PRIVATE_EMAIL_1]")
                    stats.add("private_email", 3)
                if "Alice Smith" in changed:
                    changed = changed.replace("Alice Smith", "[PRIVATE_PERSON_1]")
                    stats.add("private_person", 2)
                return changed
            if isinstance(v, list):
                return [walk(x) for x in v]
            if isinstance(v, dict):
                return {k: walk(x) for k, x in v.items()}
            return v

        return walk(payload), stats


@pytest.mark.asyncio
async def test_proxy_filters_openai_payload(monkeypatch):
    privacy_app.settings.inbound_api_keys = ["test-token"]
    privacy_app.sanitizer = FakeSanitizer()

    async def fake_forward_request(req, full_path, sanitized_payload, stream=False):
        assert full_path == "chat/completions"
        assert stream is False
        assert sanitized_payload["stream"] is False
        assert sanitized_payload["messages"][0]["content"] == (
            "Hello, my name is [PRIVATE_PERSON_1] and my email is [PRIVATE_EMAIL_1]"
        )
        return JSONResponse(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok"
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        )

    monkeypatch.setattr(privacy_app, "forward_request", fake_forward_request)

    transport = ASGITransport(app=privacy_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "ai-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, my name is Alice Smith and my email is alice@example.com"
                    }
                ],
                "stream": False,
            },
        )

    assert res.status_code == 200
    assert res.headers["x-privacy-filtered-tokens"] == "5"
    assert res.headers["x-privacy-filtered-spans"] == "2"


@pytest.mark.asyncio
async def test_proxy_forwards_streaming_without_buffering(monkeypatch):
    privacy_app.settings.inbound_api_keys = ["test-token"]
    privacy_app.sanitizer = FakeSanitizer()

    async def fake_forward_request(req, full_path, sanitized_payload, stream=False):
        assert full_path == "chat/completions"
        assert stream is True
        return JSONResponse({"ok": True})

    monkeypatch.setattr(privacy_app, "forward_request", fake_forward_request)

    transport = ASGITransport(app=privacy_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "ai-chat",
                "messages": [{"role": "user", "content": "Bonjour Alice Smith"}],
                "stream": True,
            },
        )

    assert res.status_code == 200
