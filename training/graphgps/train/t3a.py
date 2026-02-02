'''
Adapted from https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py
'''

import copy
import logging
import time
import numpy as np

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train


@torch.jit.script # type: ignore
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@register_train('t3a') # type: ignore
def t3a(loggers, loaders, model, optimizer=None, scheduler=None):
    return adapt_full(loggers, loaders, model, t3a_adapt_func)


@register_train('tent') # type: ignore
def tent(loggers, loaders, model, optimizer=None, scheduler=None):
    return adapt_full(loggers, loaders, model, tent_adapt_func)


def adapt_full(loggers, loaders, model, adapt_func):
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

    for i in range(0, num_splits):
        adapt_func(loggers[i], loaders[i], model, split=split_names[i])
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
        f"> {adapt_func.__name__.split('_')[0]} Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()

@torch.no_grad()
def t3a_adapt_func(logger, loader, model, split='train'):
    """
    Args:
        logger: Logger for the current split
        loader: DataLoader for the current split
        model: GNN model
        split: Name of the split (train, val, test)
    """
    model.eval()

    # hard patch model
    last_layer = model.post_mp.layer_post_mp.model[1]
    model.post_mp.layer_post_mp.model[1] = torch.nn.Identity()
    num_classes = loader.dataset.num_classes
    t3a_model = T3A(num_classes, last_layer.model)

    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        assert cfg.gnn.head != 'inductive_edge'
        pred, true = model(batch)
        extra_stats = {}

        pred = t3a_model(pred, adapt=True)

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

    # restore model
    model.post_mp.layer_post_mp.model[1] = last_layer


class T3A(torch.nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, classifier):
        super().__init__()
        self.classifier = classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float() # type: ignore

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = getattr(cfg.train, 'filter_K', -1)
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, z, adapt=False):
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float() # type: ignore
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1) # type: ignore
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0) # type: ignore

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


@torch.no_grad()
def tent_adapt_func(logger, loader, model, split='train'):
    """
    Args:
        logger: Logger for the current split
        loader: DataLoader for the current split
        model: GNN model
        split: Name of the split (train, val, test)
    """
    model.eval()
    tent_model = TentFull(model)

    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        assert cfg.gnn.head != 'inductive_edge'
        pred, true = tent_model(batch, adapt=True)
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


class TentFull(torch.nn.Module):
    def __init__(self, model):
        '''
        Hyperparameters:
        - alpha: learning rate multiplier for the optimizer
        - gamma: number of steps to adapt the model
        '''
        super().__init__()
        self.steps = getattr(cfg.train, 'tent_gamma', 10)  # number of steps to adapt the model
        self.alpha = getattr(cfg.train, 'tent_alpha', 5e-5)  # learning rate multiplier
        self.weight_decay = getattr(cfg.train, 'weight_decay', 0.0)  # weight decay for optimizer
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.model = configure_model(model)
        self.optimizer = torch.optim.Adam(
            collect_params(model), 
            lr=self.alpha,
            weight_decay=self.weight_decay
        )
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def forward(self, x, adapt=False):
        if adapt:
            self.optimizer.load_state_dict(self.optimizer_state)
            copy_x = x.x.clone()

            for _ in range(self.steps):
                # self.model.eval()
                x.x = copy_x  # reset x to original input
                outputs = self.forward_and_adapt(x)
                # self.model.train()
        else:
            outputs = self.model(x)
        return outputs # type: ignore

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        self.optimizer.zero_grad()
        outputs = self.model(x)
        # adapt
        loss = softmax_entropy(outputs[0]).mean(0)
        loss.backward()
        self.optimizer.step()
        return outputs
        

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False  # type: ignore
            m.running_mean = None   # type: ignore
            m.running_var = None    # type: ignore
    return model


def collect_params(model):
    params = []
    for _, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
    return params