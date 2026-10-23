import asyncio
import gzip
import threading
import time
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient, MockTransport
from starlette.requests import ClientDisconnect, Request

import app


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return [1, 2]


def ready_sanitizer(classifier):
    sanitizer = app.PrivacySanitizer()
    sanitizer.classifier = classifier
    sanitizer.tokenizer = Tokenizer()
    sanitizer._touch()
    return sanitizer


async def wait_for_event(event):
    assert await asyncio.to_thread(event.wait, 3), "worker did not reach the expected stage"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["load", "inference"])
async def test_slow_model_work_does_not_block_health(monkeypatch, stage):
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    loads = []
    sanitizer = app.PrivacySanitizer()

    def block():
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(3)

    def classify(texts, **kwargs):
        if stage == "inference":
            block()
        return [[] for text in texts]

    def load():
        loads.append(True)
        if stage == "load":
            block()
        sanitizer.tokenizer = Tokenizer()
        sanitizer.classifier = classify

    original_ensure = sanitizer.ensure_loaded

    def ensure():
        if sanitizer.classifier is None:
            load()
        original_ensure()

    monkeypatch.setattr(sanitizer, "ensure_loaded", ensure)
    task = asyncio.create_task(sanitizer.sanitize_payload({"content": "hello"}))
    try:
        await wait_for_event(entered)
        second = asyncio.create_task(sanitizer.sanitize_payload({"content": "world"}))
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            health = await asyncio.wait_for(client.get("/health"), 1)
        assert health.status_code == 200
        assert not task.done()
        release.set()
        await asyncio.gather(task, second)
        assert len(loads) == 1
    finally:
        release.set()
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_cancellation_keeps_admission_occupied_and_model_serialized(monkeypatch):
    monkeypatch.setattr(app.settings, "inference_max_pending", 2)
    entered, release = threading.Event(), threading.Event()
    active = 0
    peak = 0
    threads = set()

    def classify(texts, **kwargs):
        nonlocal active, peak
        threads.add(threading.get_ident())
        active += 1
        peak = max(peak, active)
        entered.set()
        assert release.wait(3)
        active -= 1
        return [[] for text in texts]

    sanitizer = ready_sanitizer(classify)
    first = asyncio.create_task(sanitizer.sanitize_payload("first"))
    try:
        await wait_for_event(entered)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert sanitizer._pending == 1
        second = asyncio.create_task(sanitizer.sanitize_payload("second"))
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as exc:
            await sanitizer.sanitize_payload("overloaded")
        assert exc.value.status_code == 503
        assert exc.value.headers == {"Retry-After": "1"}
        release.set()
        assert (await second)[0] == "second"
        assert peak == 1
        assert len(threads) == 1
        assert sanitizer._pending == 0
    finally:
        release.set()
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_batching_deduplicates_inference_but_preserves_stats_and_placeholders(monkeypatch):
    monkeypatch.setattr(app.settings, "inference_batch_size", 2)
    batches = []

    def classify(texts, **kwargs):
        batches.append(list(texts))
        assert kwargs == {"batch_size": len(texts), "num_workers": 0}
        return [[{"start": 0, "end": 5, "entity_group": "person", "score": 1}] for text in texts]

    sanitizer = ready_sanitizer(classify)
    try:
        result, stats = await sanitizer.sanitize_payload({
            "model": "untouched", "thinking": {"private": "untouched"},
            "messages": ["Alice one", "Alice one", "Alice two", "Bobxx three"],
        })
        assert batches == [["Alice one", "Alice two"], ["Bobxx three"]]
        assert result["messages"] == ["[PERSON_1] one", "[PERSON_1] one", "[PERSON_1] two", "[PERSON_2] three"]
        assert result["thinking"] == {"private": "untouched"}
        assert result["model"] == "untouched"
        assert (stats.spans, stats.tokens) == (4, 8)
        assert sanitizer.tokenizer.encoded == ["Alice", "Bobxx"]
        # No cross-request cache or placeholder numbering.
        result, _ = await sanitizer.sanitize_payload("Bobxx three")
        assert result == "[PERSON_1] three"
        assert batches[-1] == ["Bobxx three"]
    finally:
        await sanitizer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("padding", [True, False])
async def test_batches_respect_character_budget_and_missing_padding(monkeypatch, padding):
    monkeypatch.setattr(app.settings, "inference_batch_size", 4)
    monkeypatch.setattr(app.settings, "inference_batch_max_chars", 6)
    batches = []

    def classify(texts, **kwargs):
        batches.append(list(texts))
        return [[] for text in texts]

    sanitizer = ready_sanitizer(classify)
    if not padding:
        sanitizer.tokenizer.pad_token_id = None
    try:
        result, _ = await sanitizer.sanitize_payload(["aa", "bb", "cccc", "long-string"])
        assert result == ["aa", "bb", "cccc", "long-string"]
        expected = [["aa", "bb"], ["cccc"], ["long-string"]] if padding else [[x] for x in result]
        assert batches == expected
    finally:
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_oversized_text_rejected_before_model_load(monkeypatch):
    monkeypatch.setattr(app.settings, "max_string_chars", 5)
    sanitizer = app.PrivacySanitizer()
    try:
        with pytest.raises(HTTPException) as exc:
            await sanitizer.sanitize_payload({"messages": ["small", "too long"]})
        assert exc.value.status_code == 413
        assert sanitizer.classifier is None
    finally:
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_idle_watchdog_unloads_without_waiting_for_a_new_request(monkeypatch):
    monkeypatch.setattr(app.settings, "model_idle_unload_seconds", 0.05)
    sanitizer = ready_sanitizer(lambda texts, **kwargs: [[] for text in texts])
    unloaded = threading.Event()
    original = sanitizer._unload

    def unload():
        original()
        unloaded.set()

    monkeypatch.setattr(sanitizer, "_unload", unload)
    await sanitizer.sanitize_payload("warm")
    watchdog = asyncio.create_task(sanitizer.idle_watchdog())
    try:
        await wait_for_event(unloaded)
        assert sanitizer.classifier is None
        assert sanitizer.tokenizer is None
    finally:
        watchdog.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_idle_watchdog_never_unloads_during_inference(monkeypatch):
    monkeypatch.setattr(app.settings, "model_idle_unload_seconds", 0.05)
    monkeypatch.setattr(app.settings, "inference_max_pending", 1)
    entered, release = threading.Event(), threading.Event()

    def classify(texts, **kwargs):
        entered.set()
        assert release.wait(3)
        return [[] for text in texts]

    sanitizer = ready_sanitizer(classify)
    task = asyncio.create_task(sanitizer.sanitize_payload("working"))
    watchdog = asyncio.create_task(sanitizer.idle_watchdog())
    try:
        await wait_for_event(entered)
        await asyncio.sleep(0.15)
        assert sanitizer.classifier is classify
        assert not watchdog.done()
        release.set()
        await task
        assert time.monotonic() - sanitizer._last_used_at < 0.05
    finally:
        release.set()
        watchdog.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog
        await sanitizer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [HTTPException(status_code=503), RuntimeError("unload failed")])
async def test_idle_watchdog_retries_after_a_forced_maintenance_failure(monkeypatch, failure, caplog):
    monkeypatch.setattr(app.settings, "model_idle_unload_seconds", 0.05)
    sanitizer = ready_sanitizer(lambda texts, **kwargs: [[] for text in texts])
    await sanitizer.sanitize_payload("warm")
    original_run = sanitizer._run
    unloaded = threading.Event()
    attempts = []

    async def fail_once(func, *args):
        attempts.append(True)
        if len(attempts) == 1:
            # Force a rejection: normal idle admission has no intervening await.
            raise failure
        result = await original_run(func, *args)
        if sanitizer.classifier is None:
            unloaded.set()
        return result

    monkeypatch.setattr(sanitizer, "_run", fail_once)
    watchdog = asyncio.create_task(sanitizer.idle_watchdog())
    try:
        await wait_for_event(unloaded)
        assert len(attempts) >= 2
        assert not watchdog.done()
        if isinstance(failure, RuntimeError):
            assert "Privacy model idle maintenance failed; will retry" in caplog.text
    finally:
        watchdog.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog
        await sanitizer.aclose()


@pytest.mark.asyncio
async def test_executor_shutdown_does_not_block_health(monkeypatch):
    sanitizer = ready_sanitizer(lambda texts, **kwargs: [[] for text in texts])
    await sanitizer.sanitize_payload("warm")
    entered, release = threading.Event(), threading.Event()
    original_shutdown = sanitizer._executor.shutdown
    loop_thread = threading.get_ident()

    def slow_shutdown(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(3)
        original_shutdown(*args, **kwargs)

    monkeypatch.setattr(sanitizer._executor, "shutdown", slow_shutdown)
    close = asyncio.create_task(sanitizer.aclose())
    try:
        await wait_for_event(entered)
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            response = await asyncio.wait_for(client.get("/health"), 1)
        assert response.status_code == 200
        assert not close.done()
    finally:
        release.set()
        await close
    assert sanitizer._executor is None


@pytest.mark.asyncio
async def test_invalid_model_and_unsupported_stream_rejected_before_inference():
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
        invalid = await client.post("/v1/chat/completions", json={"model": "-anonym", "messages": ["hello"]})
        stream = await client.post("/v1/responses", json={"model": "valid", "stream": True})
    assert invalid.status_code == 400
    assert stream.status_code == 400
    assert stream.json()["detail"] == "filtered_streaming_supported_for_chat_and_text_completions_only"
    assert app.sanitizer._executor is None


@pytest.mark.asyncio
async def test_shared_http_client_lifecycle_and_no_cross_user_cookies(monkeypatch):
    created = []
    received = []

    def upstream(request):
        received.append(request)
        return httpx.Response(200, json={"object": "model", "id": "test"}, headers={"set-cookie": "session=secret"})

    def make_client(**kwargs):
        client = AsyncClient(transport=MockTransport(upstream), **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(app.httpx, "AsyncClient", make_client)
    async with app.lifespan(app.app):
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            first = await client.get("/v1/models?a=1&a=2", headers={"Cookie": "user=one"})
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            second = await client.get("/v1/models")
        assert first.json()["id"] == "test-anonym"
        assert second.status_code == 200
        assert len(created) == 1
        assert not created[0].cookies
        assert received[0].headers["cookie"] == "user=one"
        assert "cookie" not in received[1].headers
        assert received[0].url.params.get_list("a") == ["1", "2"]
    assert created[0].is_closed
    assert app.app.state.upstream_client is None


class EventStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.closed = False

    async def __aiter__(self):
        yield self.content
        yield b""

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_stream_closes_when_send_fails_before_body_is_started():
    source = EventStream(b"data: hello\n\n")

    def upstream(request):
        return httpx.Response(200, stream=source, headers={"content-type": "text/event-stream"})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client disconnected before response headers")

    async with AsyncClient(transport=MockTransport(upstream)) as client:
        app.app.state.upstream_client = client
        scope = {"type": "http", "app": app.app, "method": "POST", "headers": [], "query_string": b"",
                 "asgi": {"spec_version": "2.4"}}
        response = await app.forward_request(Request(scope), "chat/completions", {}, stream=True)
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert source.closed
        assert not client.is_closed
    app.app.state.upstream_client = None


@pytest.mark.asyncio
async def test_stream_decoding_and_early_close_keep_shared_client_open():
    content = b'data: {"content":"hello"}\n\n'
    source = EventStream(gzip.compress(content))

    def upstream(request):
        return httpx.Response(200, stream=source, headers={"content-encoding": "gzip", "content-type": "text/event-stream"})

    async with AsyncClient(transport=MockTransport(upstream)) as client:
        app.app.state.upstream_client = client
        request = Request({"type": "http", "app": app.app, "method": "POST", "headers": [], "query_string": b""})
        response = await app.forward_request(request, "chat/completions", {"stream": True}, stream=True)
        assert "content-encoding" not in response.headers
        assert "content-length" not in response.headers
        assert await anext(response.body_iterator) == content
        await response.body_iterator.aclose()
        assert source.closed
        assert not client.is_closed
    app.app.state.upstream_client = None


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_output", [True, False])
async def test_response_json_decoded_once_and_filter_timing_excludes_upstream(monkeypatch, filter_output):
    monkeypatch.setattr(app.settings, "filter_output", filter_output)
    payload = b'{"model":"test","choices":[{"message":{"content":"hello"}}]}'
    decoded = []
    original_loads = app.json.loads

    def loads(value, *args, **kwargs):
        if value == payload:
            decoded.append(True)
        return original_loads(value, *args, **kwargs)

    class Sanitizer:
        async def sanitize_payload(self, value):
            return value, app.RedactionStats()

    async def forward(*args, **kwargs):
        return app.Response(content=payload, media_type="application/json")

    clock = iter([0.0, 0.01, 1.0, 10.0, 20.0, 20.02])
    monkeypatch.setattr(app, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(app.json, "loads", loads)
    monkeypatch.setattr(app, "sanitizer", Sanitizer())
    monkeypatch.setattr(app, "forward_request", forward)
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test", "messages": []})
    assert response.status_code == 200
    assert decoded == [True]
    assert response.json()["model"] == "test-anonym"
    assert float(response.headers["x-upstream-latency-ms"]) == 9000
    assert float(response.headers["x-privacy-filter-input-latency-ms"]) == 10
    assert float(response.headers["x-privacy-filter-latency-ms"]) == (30 if filter_output else 10)


@pytest.mark.asyncio
async def test_real_transformers_batch_pipeline_without_network():
    import torch
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast, pipeline

    raw = RawTokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "Alice": 2, "Bob": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", model_max_length=32,
    )
    model = BertForTokenClassification(BertConfig(
        vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, id2label={0: "O", 1: "B-PER"}, label2id={"O": 0, "B-PER": 1},
    ))
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 10.0]))
    model.eval()
    modes = []
    model.register_forward_pre_hook(lambda *args: modes.append(torch.is_inference_mode_enabled()))
    sanitizer = ready_sanitizer(pipeline(
        "token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1,
    ))
    sanitizer.tokenizer = tokenizer
    try:
        result, stats = await sanitizer.sanitize_payload(["Alice", "Bob", "Alice"])
        assert result == ["[PER_1]", "[PER_2]", "[PER_1]"]
        assert (stats.spans, stats.tokens) == (3, 3)
        assert modes == [True]
    finally:
        await sanitizer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,status", [(httpx.ConnectError, 502), (httpx.ReadTimeout, 504), (httpx.PoolTimeout, 503)])
async def test_upstream_failures_are_reported_without_closing_the_pool(failure, status):
    def upstream(request):
        raise failure("private backend details", request=request)

    async with AsyncClient(transport=MockTransport(upstream)) as client:
        app.app.state.upstream_client = client
        request = Request({"type": "http", "app": app.app, "method": "POST", "headers": [], "query_string": b""})
        with pytest.raises(HTTPException) as exc:
            await app.forward_request(request, "chat/completions", {}, stream=True)
        assert exc.value.status_code == status
        assert "private backend details" not in exc.value.detail
        assert not client.is_closed
    app.app.state.upstream_client = None
