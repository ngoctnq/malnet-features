The configurations will be structured as follows:

```
./configs/<task>/<model>/<dataset>_<model>_<collator>+<ablation>+<base_feature>[[<dataset>]_<method>]
```

where:
- `task` is the experiment type:
    - `cls` is upstream classification
    - `cls_mntx` is local training for MNT-X variants
    - `eval_mncf` is evaluation on MNC(-F) of models trained on MNT
    - `ft` is finetuning on MNT-X for models trained on MNT
- `model` is the architecture being tested
    - One of: GCN, GAT, GIN, GPS, Exphormer (`xfm`)
    - Model is duplicated because checkpoints are saved based on the config filename.
- `dataset` is the dataset being trained on, one of the following:
    - `mnt`: original MalNet-Tiny
    - `mntf`: MalNet-Tiny + Features
    - `mncf`: MalNet-Tiny-Common + Features
    - `mndf`: MalNet-Tiny-Distinct + Features
- `collator` is how MNT-F collates features of different dimensions
    - One of the following: `prune`/`trim`/`zero`/`none`
- `ablation` is the features included in MNT-F:
    - `base` denotes the base EMBER features
    - `llm` denotes the use of LLM features
    - `all` denotes both.
- `base_feature` is the original features tacked on
    - `ldp` denotes tacking LDP features
    - `none` is just none.
- (Optional) `method`: Step-2 adaptation approach
    - `eval` is just the evaluation.
    - `ft` is the model (vanilla) finetuned on the new dataset.