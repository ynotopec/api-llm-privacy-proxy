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
        assert sanitized_payload["model"] == "gpt-4o"
        assert sanitized_payload["thinking"] == {"level": "high", "budget_tokens": 1024}
        assert sanitized_payload["metadata"]["model"] == "audit-model-anonym"
        assert sanitized_payload["messages"][0]["content"] == (
            "Hello, my name is [PRIVATE_PERSON_1] and my email is [PRIVATE_EMAIL_1]"
        )
        return JSONResponse(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
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
                "model": "gpt-4o-anonym",
                "thinking": {"level": "high", "budget_tokens": 1024},
                "metadata": {"model": "audit-model-anonym"},
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
    assert int(res.headers["content-length"]) == len(res.content)
    assert res.json()["model"] == "gpt-4o-anonym"


@pytest.mark.asyncio
async def test_proxy_forwards_streaming_without_buffering(monkeypatch):
    privacy_app.settings.inbound_api_keys = ["test-token"]
    privacy_app.sanitizer = FakeSanitizer()

    async def fake_forward_request(req, full_path, sanitized_payload, stream=False):
        assert full_path == "chat/completions"
        assert stream is True
        assert sanitized_payload["model"] == "gpt-4o"
        return JSONResponse({"ok": True})

    monkeypatch.setattr(privacy_app, "forward_request", fake_forward_request)

    transport = ASGITransport(app=privacy_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-4o-anonym",
                "messages": [{"role": "user", "content": "Bonjour Alice Smith"}],
                "stream": True,
            },
        )

    assert res.status_code == 200


@pytest.mark.asyncio
async def test_proxy_suffixes_model_list(monkeypatch):
    privacy_app.settings.inbound_api_keys = ["test-token"]

    async def fake_forward_request(req, full_path, sanitized_payload, stream=False):
        assert full_path == "models"
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": "gpt-4o", "object": "model"},
                    {"id": "embedding-model-anonym", "object": "model"},
                ],
            }
        )

    monkeypatch.setattr(privacy_app, "forward_request", fake_forward_request)

    transport = ASGITransport(app=privacy_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-token"},
        )

    assert res.status_code == 200
    assert int(res.headers["content-length"]) == len(res.content)
    assert [model["id"] for model in res.json()["data"]] == [
        "gpt-4o-anonym",
        "embedding-model-anonym",
    ]


@pytest.mark.asyncio
async def test_sanitizer_preserves_user_config_keys(monkeypatch):
    sanitizer = privacy_app.PrivacySanitizer()

    async def fake_sanitize_text(text, ctx, stats):
        if "Alice Smith" in text:
            stats.add("private_person", 2)
            return text.replace("Alice Smith", "[PRIVATE_PERSON_1]")
        return text

    monkeypatch.setattr(sanitizer, "sanitize_text", fake_sanitize_text)

    sanitized, stats = await sanitizer.sanitize_payload(
        {
            "model": "gpt-4o-anonym",
            "thinking": {"level": "high", "note": "Alice Smith"},
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "Bonjour Alice Smith"}],
        }
    )

    assert sanitized["thinking"] == {"level": "high", "note": "Alice Smith"}
    assert sanitized["reasoning_effort"] == "high"
    assert sanitized["messages"][0]["content"] == "Bonjour [PRIVATE_PERSON_1]"
    assert stats.spans == 1
