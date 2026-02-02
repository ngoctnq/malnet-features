# Quantifying the Generalization Gap: A New Benchmark for Out-of-Distribution Graph-Based Android Malware Classification

This repository contains the accompanied code to reproduce the results in our paper. Please refer to the individual `README.md` in each subfolder for further details.

## `feature-construction`: Dataset construction
To reconstruct the dataset, please download all the necessary APKs from the [original repo](https://androzoo.uni.lu/), with [permission from the original owners](https://androzoo.uni.lu/access).
- `splits` subfolder contains the split of our two new datasets, in the same format as [MalNet](https://github.com/safreita1/malnet-graph/tree/master/split_info_tiny/type).
- Run `llm_inference_server.py` to run a server with a HuggingFace instance of the code embedding extractor.
- Run `create_graph.py` to generate the attributed FCGs. This script will invoke REST requests to the LLM server to generate function embeddings.
## `training`: Model training and adaptation
Move processed data (whether independently constructed or downloaded from our precomputed upload) into the appropriate directory structure, then run model training/evaluation.
- Follow the instructions in `Exphormers.md` to set up environment.
- Put the data in the `datasets` subfolder according to `README.md`.
- Create the desired training configuration and run:
```bash
python main.py --cfg <yaml_file>
``` 

## Citation
```
@misc{tran2026quantifyinggeneralizationgapnew,
      title={Quantifying the Generalization Gap: A New Benchmark for Out-of-Distribution Graph-Based Android Malware Classification}, 
      author={Ngoc N. Tran and Anwar Said and Waseem Abbas and Tyler Derr and Xenofon D. Koutsoukos},
      year={2026},
      eprint={2508.06734},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2508.06734}, 
}
```