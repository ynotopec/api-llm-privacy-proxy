from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": body.get("model"),
        "received_by_upstream": body,
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


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model"},
        ],
    }
