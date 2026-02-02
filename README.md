# Evaluating Out-of-Distribution Robustness in Graph-Based Android Malware Classification: A New Principled Benchmark

This repository contains the accompanied code to reproduce the results in our paper. Please refer to the individual `README.md` in each subfolder for further details.

## `dataloader`: Precomputed dataset

To improve usability, we provide a repackaged precomputed download for the dataset splits described in the paper and uploaded to HuggingFace at this link: https://huggingface.co/datasets/ngoctnq/malnet-features. Please consult the repo's README for more information on how to use the dataset.

## `feature-construction`: Dataset construction
To reconstruct the dataset, please download all the necessary APKs from the [original repo](https://androzoo.uni.lu/), with [permission from the original owners](https://androzoo.uni.lu/access).
- `splits` subfolder contains the split of our two new datasets, in the same format as [MalNet](https://github.com/safreita1/malnet-graph/tree/master/split_info_tiny/type).
- Run `llm_inference_server.py` to run a server with a HuggingFace instance of the code embedding extractor.
- Run `create_graph.py` to generate the attributed FCGs. This script will invoke REST requests to the LLM server to generate function embeddings.

## `training`: Model training and adaptation
Move the above constructed data into the appropriate directory structure, then run model training/evaluation.
- Follow the instructions in `Exphormers.md` to set up environment.
- Put the data in the `datasets` subfolder according to `README.md`.
- Create the desired training configuration and run:
```bash
python main.py --cfg <yaml_file>
``` 