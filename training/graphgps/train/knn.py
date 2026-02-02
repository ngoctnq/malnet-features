import logging
import time
import numpy as np

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train, register_config

from sklearn.neighbors import KNeighborsClassifier


@register_train('knn_probe') # type: ignore
@torch.no_grad()
def adapt_full(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run T3A inference.

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

    # hard patch model
    model.post_mp.layer_post_mp.model[1] = torch.nn.Identity()
    num_classes = loaders[0].dataset.num_classes
    classifier = KNeighborsClassifier(n_neighbors=cfg.train.knn_probe_k)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # generate features
    features, labels = [], []
    for batch in loaders[0]:
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        assert cfg.gnn.head != 'inductive_edge'
        feature, label = model(batch)
        features.append(feature.detach().cpu())
        labels.append(label.detach().cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # train the classifier
    classifier.fit(features, labels)

    for i in range(0, num_splits):
        eval_func(loggers[i], loaders[i], model, classifier, split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    val_perf = perf[1]

    best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                             cfg.metric_agg)()
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> k-NN Probe Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


def eval_func(logger, loader, featurizer, classifier, split='train'):
    """
    Args:
        logger: Logger for the current split
        loader: DataLoader for the current split
        model: GNN model
        split: Name of the split (train, val, test)
    """
    featurizer.eval()

    time_start = time.time()

    features, labels = [], []

    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        assert cfg.gnn.head != 'inductive_edge'

        feature, label = featurizer(batch)
        features.append(feature.detach().cpu())
        labels.append(label.detach().cpu())

    features = torch.cat(features, dim=0)
    _true = torch.cat(labels, dim=0)
    _pred = torch.tensor(classifier.predict_proba(features))
    
    logger.update_stats(true=_true,
                        pred=_pred,
                        loss=0,
                        lr=0, time_used=time.time() - time_start,
                        params=cfg.params,
                        dataset_name=cfg.dataset.name)


@register_config('knn_probe')
def knn_probe_config(cfg):
    cfg.train.knn_probe_k = 50
    return cfg
