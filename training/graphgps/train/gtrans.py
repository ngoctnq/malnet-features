import copy
import logging
import time
import numpy as np

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train, register_config
from tqdm import tqdm
from yacs.config import CfgNode as CN

from types import MethodType

import sys
sys.path.append('GTrans')
from GTrans.gtransform_both import GraphAgent as GTrans

import warnings
warnings.filterwarnings("ignore")

global model_features
model_features = None

@torch.jit.script # type: ignore
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@register_train('gtrans') # type: ignore
def gtrans(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run GTrans adaptation.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    # ignore train and val set
    # for i in range(0, num_splits):
    for i in [2]:
        adapt_func(loggers[i], loaders[i], model, split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    perf[0] = copy.deepcopy(perf[2])  # Copy test performance to train logger
    perf[1] = copy.deepcopy(perf[2])  # Copy test performance to val logger
    # val_perf = perf[1]

    # best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        # best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
        #                      cfg.metric_agg)()
        # if m in perf[0][best_epoch]:
        #     best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        # else:
        #     # Note: For some datasets it is too expensive to compute
        #     # the main metric on the training set.
        #     best_train = f"train_{m}: {0:.4f}"
        # best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> GTrans Inference | "
        # f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        # f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()

@torch.no_grad()
def adapt_func(logger, loader, model, split='train'):
    """
    Args:
        logger: Logger for the current split
        loader: DataLoader for the current split
        model: GNN model
        split: Name of the split (train, val, test)
    """
    feature_hook = get_model_features_hook(model)
    agent = GTrans(model, loader, cfg)
    finetune = getattr(cfg.train, 'finetune', False)

    time_start = time.time()
    for batch in tqdm(loader):
        batch.split = split
        batch.to(torch.device(cfg.device))
        assert cfg.gnn.head != 'inductive_edge'

        assert cfg.train.batch_size == 1, \
            "GTrans requires batch size of 1, please set cfg.train.batch_size=1"
        if 'Exphormer' not in cfg.gt.layer_type:
            batch = batch[0]

        with torch.enable_grad():
            if finetune:
                acc, output, labels = agent.finetune(batch)
            else:
                acc, output, labels = agent.learn_graph(batch)

        pred, true = output, labels
        extra_stats = {}

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)

        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()

    feature_hook.remove()


# hook to get model features
def get_model_features_hook(model):
    def hook(module, input, output):
        global model_features
        model_features = input
    hook_obj = model.post_mp.layer_post_mp.model[1].register_forward_hook(hook)

    # monkey patch a get_embed method to the model
    def get_embed(self, batch):
        self(batch)
        global model_features
        return model_features
    model.get_embed = MethodType(get_embed, model)

    return hook_obj


@register_config('gtrans') # type: ignore
def gtrans_cfg(cfg):
    """GTrans-specific config options."""
    cfg.gtrans = CN()
    cfg.gtrans.finetune = False
    cfg.gtrans.lr_adj = 0.1
    cfg.gtrans.epochs = 20
    cfg.gtrans.lr_feat = 0.001
    cfg.gtrans.ratio = 0.1
    cfg.gtrans.existing_space = 1
    cfg.gtrans.loop_feat = 4
    cfg.gtrans.loop_adj = 0
    cfg.gtrans.tent = 0
    cfg.gtrans.debug = 1
    cfg.gtrans.loss = 'LC'
    cfg.gtrans.strategy = 'dropedge'
    cfg.gtrans.margin = -1
