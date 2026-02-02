import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embedding(model, tokenizer, text: str, max_length: int = 8192) -> Tensor:
    # Tokenize the input texts
    batch_dict = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).to(device)
model.eval()


@torch.no_grad()
def infer_fn(src: str):
    return get_embedding(model, tokenizer, src).squeeze().detach().cpu().numpy()


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
