from transformers import AutoModel
import torch


# model for extracting code embedding
model = AutoModel.from_pretrained(
    "Salesforce/SFR-Embedding-Code-2B_R", trust_remote_code=True
)


@torch.no_grad()
def infer_fn(src: str):
    return model.encode_corpus([src], max_length=32768).cpu().numpy()


# serve the model
def serve_model(port):
    import uvicorn
    from fastapi import FastAPI, Response
    from pydantic import BaseModel

    app = FastAPI()

    class CodeRequest(BaseModel):
        code: str

    @app.post("/")
    async def get_code_embeddings_endpoint(request: CodeRequest):
        code = request.code
        embedding = infer_fn(code)
        return Response(content=embedding.tobytes())

    uvicorn.run(app, host="localhost", port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve the model")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve the model on"
    )
    args = parser.parse_args()

    port = args.port
    print(f"Serving model on port {port}")
    serve_model(port)

# example inference
# curl -X POST "http://localhost:8000/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"code\":\"def add(a, b):\\n    return a + b\"}"
