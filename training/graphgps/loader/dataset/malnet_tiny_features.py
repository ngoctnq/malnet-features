import pickle
from typing import Optional, Callable, List

import os
import glob
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_isolated_nodes
from tqdm import tqdm


class MalNetTinyFeatures(InMemoryDataset):
    # 70/10/20 train, val, test split by type
    def __init__(
        self,
        collator,
        ablation,
        root: str,
        remove_isolated: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        variant: str = "",
        llm_name: Optional[str] = None,
    ):
        '''
        Args:
            collator: trim/prune/zero/none
            ablation: base/llm/all
            variant: common/distinct
            llm_name: None/unix/qwen
                None: use CodeXEmbed (default)
        '''
        
        # had to add this before super() to make processed_dir work
        self.collator = collator
        self.ablation = ablation
        self.remove_isolated = remove_isolated
        self.variant = variant
        self.llm_name = llm_name
        assert not (collator == 'prune' and remove_isolated), \
            "Cannot use prune with remove_isolated=True, as sometimes we get empty graphs."        

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        if self.variant == '' or self.variant.lower() == 'common':
            folders = ["addisplay", "adware", "benign", "downloader", "trojan"]
        elif self.variant.lower() == 'distinct':
            folders = ["clicker++trojan", "malware", "riskware", "spr", "spyware"]
        else:
            raise ValueError(f"Invalid variant: {self.variant}")
        return [osp.join("malnet-graphs-tiny", folder) for folder in folders]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "split_dict.pt", "keep_idxs_list.pt", "mask_list.pt", "hash_list.pt"]

    @property
    def processed_dir(self) -> str:
        feature_set = self.collator + '+' + self.ablation
        if self.llm_name is not None:
            feature_set += '+' + self.llm_name
        if not self.remove_isolated:
            feature_set += "_keepisolated"
        return osp.join(super().processed_dir, feature_set)

    def process(self):
        data_list = []
        split_dict = {"train": [], "valid": [], "test": []}
        keep_idxs_list = []
        mask_list = []
        hash_list = []

        parse = (  # noqa: E731
            lambda f: set(
                [x.split("/")[-1] for x in f.read().split("\n")[:-1]]
            )
        )  # -1 for empty line at EOF
        split_dir = osp.join(self.raw_dir, "split_info_tiny", "type")
        with open(osp.join(split_dir, "train.txt"), "r") as f:
            train_names = parse(f)
            assert len(train_names) == 3500
        with open(osp.join(split_dir, "val.txt"), "r") as f:
            val_names = parse(f)
            assert len(val_names) == 500
        with open(osp.join(split_dir, "test.txt"), "r") as f:
            test_names = parse(f)
            assert len(test_names) == 1000

        for y, raw_path in enumerate(self.raw_paths):
            filenames = glob.glob(osp.join(raw_path, "*/*.pkl"))

            for filename in tqdm(filenames):
                hash_list.append(osp.splitext(osp.basename(filename))[0])
                edge_index, node_features, keep_idxs = process_file(filename, self.collator, self.ablation, self.llm_name)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                node_features = torch.tensor(node_features, dtype=torch.float)

                # if all edges are pruned, make it an empty graph
                if self.remove_isolated:
                    if edge_index.numel() == 0:
                        # print(filename)
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        num_nodes = 0
                        mask = []
                    else:
                        # Remove isolated nodes, including those with only a self-loop
                        edge_index, _, mask = remove_isolated_nodes(
                            edge_index, num_nodes=node_features.shape[0]
                        )
                        num_nodes = int(edge_index.max()) + 1
                else:
                    mask = torch.ones(node_features.shape[0], dtype=torch.bool)
                    num_nodes = node_features.shape[0]

                data = Data(
                    x=node_features[mask],
                    edge_index=edge_index,
                    y=torch.tensor(y),
                    num_nodes=num_nodes,
                )
                data_list.append(data)

                ind = len(data_list) - 1
                graph_id = osp.splitext(osp.basename(filename))[0]
                if graph_id in train_names:
                    split_dict["train"].append(ind)
                elif graph_id in val_names:
                    split_dict["valid"].append(ind)
                elif graph_id in test_names:
                    split_dict["test"].append(ind)
                else:
                    raise ValueError(f'No split assignment for "{graph_id}".')

                keep_idxs_list.append(keep_idxs)
                mask_list.append(mask)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])
        torch.save(keep_idxs_list, self.processed_paths[2])
        torch.save(mask_list, self.processed_paths[3])
        torch.save(hash_list, self.processed_paths[4])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])


def process_file(file, collator, ablation, llm_name=None):
    # number of features for base variant without LLM embeddings
    NUM_FEATURES = 940

    llm_name = 'codexembed' if llm_name is None else llm_name.lower()

    if llm_name == 'codexembed':
        llm_dim = 2304  # CodeXEmbed
    elif llm_name == 'unix':
        llm_dim = 768   # UniXCoder
    elif llm_name == 'qwen':
        llm_dim = 1024  # Qwen-3-Embedding-0.6B
    else:
        raise ValueError(f"Invalid llm_name: {llm_name}")
    
    try:
        pkld = pickle.load(open(file, "rb"))
    except Exception as e:
        print(f"Error loading {file}!")
        raise e

    pkld["node_features"] = list(pkld["node_features"])
    if ablation != 'base':
        pkld['llm_features'] = []
        for k, v in enumerate(pkld["node_features"]):
            # source code is only available if all features are present
            if v.shape[0] > NUM_FEATURES:
                node_feature = v[:NUM_FEATURES]
                llm_embedding = v[NUM_FEATURES:]
            else:
                node_feature = v
                llm_embedding = np.zeros((0,))
            if llm_embedding.shape[0] != 0:
                assert llm_embedding.shape[0] == llm_dim, f"LLM embedding dimension mismatch: {llm_embedding.shape[0]} vs {llm_dim}"
            
            pkld["node_features"][k] = node_feature
            pkld['llm_features'].append(llm_embedding)

    keep_idxs = None # for returning which nodes are kept after collator processing
    if collator == "trim":
        num_features = min(x.shape[0] for x in pkld["node_features"])
        for k, v in enumerate(pkld["node_features"]):
            pkld["node_features"][k] = v[:num_features]

    elif collator == "none":
        for k in range(len(pkld["node_features"])):
            pkld["node_features"][k] = np.zeros((0,))

    else:
        if ablation not in ["base", "llm", "all"]:
            print(f"Invalid ablation: {ablation}")
            exit(1)

        # common processing for both Prune and Zero
        num_meta_features = max(x.shape[0] for x in pkld["node_features"])
        assert num_meta_features == NUM_FEATURES, \
            f"Meta feature dimension mismatch: {num_meta_features} vs {NUM_FEATURES}"
        if 'llm_features' in pkld:
            num_llm_features = llm_dim
            assert all(v.shape[0] == llm_dim for v in pkld["llm_features"]), \
                f"LLM feature dimension mismatch: {num_llm_features} vs {llm_dim}"
        else:
            num_llm_features = 0

        if collator == "prune":
            if ablation == "base":
                keep_idxs = [
                    idx
                    for idx in range(len(pkld["node_features"]))
                    if pkld["node_features"][idx].shape[0] == num_meta_features
                ]
                pkld["node_features"] = [pkld["node_features"][idx] for idx in keep_idxs]
            else:
                assert num_llm_features > 0, f"LLM features not found for ablation='{ablation}'"
                if ablation == "llm":
                    keep_idxs = [
                        idx
                        for idx in range(len(pkld["llm_features"]))
                        if pkld["llm_features"][idx].shape[0] == num_llm_features
                    ]
                    pkld["node_features"] = [pkld["llm_features"][idx] for idx in keep_idxs]
                else:  # all
                    keep_idxs = [
                        idx
                        for idx in range(len(pkld["node_features"]))
                        if pkld["node_features"][idx].shape[0] == num_meta_features and \
                            pkld["llm_features"][idx].shape[0] == num_llm_features
                    ]
                    pkld["node_features"] = [
                        np.concatenate([pkld["node_features"][idx], pkld["llm_features"][idx]], axis=0)
                        for idx in keep_idxs
                    ]
                
            idx_map = {k: v for v, k in enumerate(keep_idxs)}
            pkld["edge_index"] = [
                (idx_map[k], idx_map[v])
                for k, v in pkld["edge_index"]
                if k in idx_map and v in idx_map
            ]

        elif collator == "zero":
            if ablation != 'llm':
                for k, v in enumerate(pkld["node_features"]):
                    if v.shape[0] != num_meta_features:
                        assert v.shape[0] < num_meta_features, "Invalid node feature dimension"
                        pkld["node_features"][k] = np.pad(
                            v, (0, num_meta_features - v.shape[0]), mode="constant"
                        )
            if ablation != 'base':
                assert num_llm_features > 0, f"LLM features not found for ablation='{ablation}'"
                for k, v in enumerate(pkld["llm_features"]):
                    if v.shape[0] != num_llm_features:
                        assert v.shape[0] < num_llm_features, "Invalid LLM feature dimension"
                        pkld["llm_features"][k] = np.pad(
                            v, (0, num_llm_features - v.shape[0]), mode="constant"
                        )

            if ablation == "base":
                # do nothing, already padded
                pass
            elif ablation == "llm":
                pkld["node_features"] = pkld["llm_features"]
            else:  # all
                pkld["node_features"] = [
                    np.concatenate([meta_f, llm_f], axis=0)
                    for meta_f, llm_f in zip(pkld["node_features"], pkld["llm_features"])
                ]
        else:
            print("Invalid cfg")
            exit(1)

    # sanity check
    for edge in pkld["edge_index"]:
        if edge[0] >= len(pkld["node_features"]) or edge[1] >= len(pkld["node_features"]):
            print("Invalid edge index")
            exit(1)

    return list(pkld["edge_index"]), np.stack(pkld["node_features"], axis=0), keep_idxs
