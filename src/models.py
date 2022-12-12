import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.datasets import TUDataset

class AttentionHead(nn.Module):
    """
    this class represents a single attention head
    """
    def __init__(self, in_dim, attn_dim, proj_dim):
        super().__init__()

        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.to_qkv = nn.Linear(in_dim, 3*attn_dim)
        self.proj = nn.Linear(attn_dim, proj_dim)
        self.scaler = attn_dim ** -0.5

    def scaled_dot_product(self, q, k, v, d_k, mask=None):
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / (d_k ** 0.5)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, adj_mask=None):
        B, T, _ = x.shape
        qkv = self.to_qkv(x)

        qkv = qkv.reshape(B, T, 3*self.attn_dim)
        qkv = qkv.permute(0, 1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = self.scaled_dot_product(q, k, v, self.attn_dim, adj_mask)
        values = values.permute(0, 2, 1)
        values = values.reshape(B, T, self.attn_dim)
        out = self.proj(values)

        return out, attention

class MHSA(nn.Module):
    def __init__(self, in_dim, attn_dim, proj_dim, n_global_heads=1, n_local_heads=1):
        super().__init__()
        self.global_heads = [AttentionHead(in_dim, attn_dim, proj_dim) for _ in range(n_global_heads)]
        self.local_heads = [AttentionHead(in_dim, attn_dim, proj_dim) for _ in range(n_local_heads)]
        self.concat_proj = nn.Linear((n_global_heads+n_local_heads)*attn_dim, in_dim) # W_O [hd_v, d_in]
    
    def forward(self, x, adj=None):
        global_attn_op = []
        local_attn_op = []

        for gh in self.global_heads:
            attn, weights = gh(x)
            global_attn_op.append(attn)

        if adj:
            for lh in self.local_heads:
                attn, weights = lh(x)
                local_attn_op.append(attn, adj)

            global_attn_op.extend(local_attn_op)

        concat = torch.concat(global_attn_op, dim=-1)
        out = self.concat_proj(concat)

        return out

class TFEncoder(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

class EdgeGNN(tgnn.MessagePassing):
    def __init__(self, edge_in_dim, edge_out_dim):
        super().__init__(aggr="sum")

        self.edge_embed = tgnn.Linear(edge_in_dim, edge_out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.edge_embed(edge_attr)

class GraphTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TFEncoder(5)
        self.gnn = EdgeGNN(edge_in_dim, edge_out_dim)

    def forward(self, x, adj, edge_index, edge_attr):
        pass

tud = TUDataset(root="../data/", use_edge_attr=True, use_node_attr=True, name="MUTAG")
graph = tud[0]
print (graph)

gnn = EdgeGNN(graph.edge_attr.size(1), 10)
print (graph.x.shape, graph.edge_attr.shape)
y = gnn(graph.x, graph.edge_index, graph.edge_attr)
print (y.shape, graph.edge_attr.shape)