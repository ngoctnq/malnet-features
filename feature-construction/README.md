# Semantic Feature Extraction for Control-Flow Graphs of Android Malwares

This pipeline takes a directory of Android packages, processes each `.apk` file, and writes a processed `.pkl`. Processing is implemented in `create_graph.py`, which includes meta-feature extraction and LLM-based code embedding. The LLM embeddings are computed by sending REST requests to an LLM embedding server started with `llm_inference_server_<llm_name>.py`.

## Environment setup
Install dependencies from `requirements.txt`. Tested on Python 3.12.

```sh
pip install -r requirements.txt
```

## How to run

### 1) Start the LLM embedding server
Pick a free port and start the server:

```sh
# run the appropriate embedder
python llm_inference_server_cxe.py --port <port>
# python llm_inference_server_qwen.py --port <port>
# python llm_inference_server_unix.py --port <port>
```

Leave the server running while APKs are being processed, then stop it afterward.

### 2) Process APKs
Run the graph feature construction:

```sh
python create_graph.py \
	--apk_dir <apk_dir> \
	--n_jobs <num_parallel_jobs> \
	--port <llm_server_port>
```
Ensure the `--port` in `create_graph.py` matches the server port. The save directory will be the original directory with a `-features` suffix, e.g. `/path/to/malnet/0A1B2C.apk` will write the output to `/path/to/malnet-features/0A1B2C.pkl`.

**Important!** The script will not run on processed files for ease of resuming. If you want to run feature construction again (e.g. with a different LLM embedder), move all processed `.pkl` files appropriately beforehand.

### Example

```sh
python llm_inference_server.py --port 8080
python create_graph.py --apk_dir ./apks --n_jobs 8 --port 8080
```

## Data splits
`splits` folder contains the split of our two new datasets, MalNet-Tiny-Common and MalNet-Tiny-Distinct. They are in the same format as the original [MalNet-Tiny split](https://github.com/safreita1/malnet-graph/tree/master/split_info_tiny/type), which should be used when constructing our features for MalNet-Tiny.

## Credit
This code is based on the original data processing code for [MalNet](https://github.com/safreita1/malnet-graph/blob/master/create_dataset.py).