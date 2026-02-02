from types import MethodType
import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .custom_train import custom_train


@register_train('AdapterGNN') # type: ignore
def adapterGNN(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    # enable grad for normalization layers and head
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            m.requires_grad_(True)
        elif 'post_mp' in name:
            m.requires_grad_(True)
        else:
            m.requires_grad_(False)

    for i in range(len(model.layers)):
        old_layer_name = model.layers[i].__class__.__name__
        
        if old_layer_name == 'MultiLayer':
            for idx, sublayer in enumerate(model.layers[i].models):
                if sublayer.__class__.__name__ == 'LocalModel':
                    sublayer.forward = MethodType(local_model_forward, sublayer)
                    new_layer = GINConv2(sublayer).to(cfg.device)
                    new_layer.forward = MethodType(gps_local_forward, new_layer)
                    model.layers[i].models[idx] = new_layer

        else:
            if old_layer_name == 'GPSLayer':
                old_layer = model.layers[i]
                old_layer.forward = MethodType(gps_forward, old_layer)
            else:
                old_layer = model.layers[i].model

                if old_layer_name == 'GINConv':
                    old_layer.forward = MethodType(gin_forward, old_layer)

                elif old_layer_name == 'GCNConv':
                    old_layer.forward = MethodType(gcn_forward, old_layer)
                
                else:
                    raise NotImplementedError(
                        f"AdapterGNN does not support {model.layers[i].__class__.__name__} layer."
                    )
            
            new_layer = GINConv2(old_layer).to(cfg.device)
            if old_layer_name == 'GPSLayer':
                new_layer.forward = MethodType(gps_batch_forward, new_layer)
            model.layers[i] = new_layer

    custom_train(loggers, loaders, model, optimizer, scheduler)


class GINConv2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        bottleneck_dim = 15
        emb_dim = cfg.gt.dim_hidden

        self.msgs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim)
            ),
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim)
            )
        ])
        
        for module in self.msgs:
            torch.nn.init.zeros_(module[2].weight.data) # type: ignore
            torch.nn.init.zeros_(module[2].bias.data) # type: ignore

        self.msgs.requires_grad_(True)

        gating = 0.01
        self.msg_gating = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=True) # type: ignore
        self.msg_gating.data += gating
        self.register_parameter('msg_gating', self.msg_gating)
        

    def forward(self, batch):
        post, pre, x = self.model(batch.x, batch.edge_index)
        x = self.msgs[0](x)
        pre = self.msgs[1](pre)
        batch.x = x * self.msg_gating[0] + pre * self.msg_gating[1] + post
        return batch
    

def gin_forward(self, x, edge_index, size=None): # type: ignore
    """"""
    if isinstance(x, torch.Tensor):
        x = (x, x)

    # propagate_type: (x: OptPairTensor)
    out = self.propagate(edge_index, x=x, size=size)

    x_r = x[1]
    if x_r is not None:
        out += (1 + self.eps) * x_r

    return self.nn(out), x[0], out


def gcn_forward(self, x, edge_index, edge_weight=None): # type: ignore
    if self.normalize:
        if isinstance(edge_index, torch.Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops) # type: ignore
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

    pre = x
    x = self.lin(x)

    # propagate_type: (x: Tensor, edge_weight: OptTensor)
    out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                            size=None)

    if self.bias is not None:
        out += self.bias

    return out, pre, x
        

def gps_batch_forward(self, batch):
    post, pre, x = self.model(batch)
    x = self.msgs[0](x)
    pre = self.msgs[1](pre)
    batch.x = x * self.msg_gating[0] + pre * self.msg_gating[1] + post.x
    return batch        


def gps_local_forward(self, batch):
    post, pre, x = self.model(batch)
    x = self.msgs[0](x)
    pre = self.msgs[1](pre)
    x = x * self.msg_gating[0] + pre * self.msg_gating[1] + post
    return x


def gps_forward(self, batch):
    h = batch.x
    h_in1 = h  # for first residual connection

    h_out_list = []
    # Local MPNN with edge attributes.
    if self.local_model is not None:
        if self.local_gnn_type == 'CustomGatedGCN':
            es_data = None
            if self.equivstable_pe:
                es_data = batch.pe_EquivStableLapPE
            local_out = self.local_model(Batch(
                batch=batch, # type: ignore
                x=h, # type: ignore
                edge_index=batch.edge_index, # type: ignore
                edge_attr=batch.edge_attr, # type: ignore
                pe_EquivStableLapPE=es_data) # type: ignore
            )
            # GatedGCN does residual connection and dropout internally.
            h_local = local_out.x
            batch.edge_attr = local_out.edge_attr
        else:
            if self.equivstable_pe:
                h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                            batch.pe_EquivStableLapPE)
            else:
                h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

    # Multi-head attention.
    if self.self_attn is not None:
        h_dense, mask = to_dense_batch(h, batch.batch)
        if self.global_model_type == 'Transformer':
            h_attn = self._sa_block(h_dense, None, ~mask)[mask]
        elif self.global_model_type == 'Performer':
            h_attn = self.self_attn(h_dense, mask=mask)[mask]
        elif self.global_model_type == 'BigBird':
            h_attn = self.self_attn(h_dense, attention_mask=mask)
        else:
            raise RuntimeError(f"Unexpected {self.global_model_type}")

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

    # Combine local and global outputs.
    # h = torch.cat(h_out_list, dim=-1)
    h = sum(h_out_list)

    # Feed Forward block.
    h = h + self._ff_block(h)
    if self.layer_norm:
        h = self.norm2(h, batch.batch)
    if self.batch_norm:
        h = self.norm2(h)

    batch.x = h
    # post, pre, x
    return batch, h_in1, h_local # type: ignore


def local_model_forward(self, batch):
    h = batch.x

    h_in1 = h  # for first residual connection

    edge_index = getattr(batch, self.edge_type)
    edge_attr = getattr(batch, self.edge_attr_type)
    if edge_index is None:
        raise ValueError(f'edge type {self.edge_type} is not stored in the data!')

    if self.local_gnn_type == 'CustomGatedGCN':
        es_data = None
        if self.equivstable_pe:
            es_data = batch.pe_EquivStableLapPE
        local_out = self.local_model(Batch(
            batch=batch, # type: ignore
            x=h, # type: ignore
            edge_index=edge_index, # type: ignore
            edge_attr=edge_attr, # type: ignore
            pe_EquivStableLapPE=es_data # type: ignore
        ))
        # GatedGCN does residual connection and dropout internally.
        h_local = local_out.x
        setattr(batch, self.edge_attr_type, local_out.edge_attr)
    else:
        if self.equivstable_pe:
            h_local = self.local_model(h, edge_index, edge_attr,
                                        batch.pe_EquivStableLapPE)
        elif self.local_gnn_type == 'GCN':
            h_local = self.local_model(h, edge_index)
        else:
            h_local = self.local_model(h, edge_index, edge_attr)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.

    if self.layer_norm:
        h_local = self.norm1_local(h_local, batch.batch)
    if self.batch_norm:
        h_local = self.norm1_local(h_local)

    # post, pre, x
    # h_in1 is duplicated because CustomGatedGCN does not have an intermediate output
    return h_local, h_in1, h_in1
