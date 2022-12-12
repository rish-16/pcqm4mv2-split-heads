import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.datasets import TUDataset

import matplotlib.pyplot as plt

class AttentionHead(nn.Module):
    """
    this class represents a single attention head
    """
    def __init__(self, in_dim, attn_dim, proj_dim):
        super().__init__()

        self.in_dim = in_dim
        self.attn_dim = attn_dim

    def scaled_dot_product(self, q, k, v, d_k, mask=None):
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / (d_k ** 0.5)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

class LocalHead(AttentionHead):
    def __init__(self, in_dim, attn_dim, proj_dim):
        super().__init__(in_dim, attn_dim, proj_dim)

        self.to_qkv = nn.Linear(in_dim, 3*attn_dim)
        self.proj = nn.Linear(attn_dim, proj_dim)

    def forward(self, x, adj):
        B, T, _ = x.shape
        qkv = self.to_qkv(x) # XWq, XWk, XWv

        qkv = qkv.reshape(B, T, 3*self.attn_dim)
        qkv = qkv.permute(0, 1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = self.scaled_dot_product(q, k, v, self.attn_dim, mask=adj)
        values = values.permute(0, 2, 1)
        values = values.reshape(B, T, self.attn_dim)
        out = self.proj(values)        

        return out, attention

class GlobalHead(AttentionHead):
    def __init__(self, in_dim, attn_dim, proj_dim):
        super().__init__(in_dim, attn_dim, proj_dim)
    
        self.to_qkv = nn.Linear(in_dim, 3*attn_dim)
        self.proj = nn.Linear(attn_dim, proj_dim)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.to_qkv(x)

        qkv = qkv.reshape(B, T, 3*self.attn_dim)
        qkv = qkv.permute(0, 1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = self.scaled_dot_product(q, k, v, self.attn_dim)
        values = values.permute(0, 2, 1)
        values = values.reshape(B, T, self.attn_dim)
        out = self.proj(values)

        return out, attention

class MHSA(nn.Module):
    def __init__(self, in_dim, attn_dim, proj_dim, n_global_heads=1, n_local_heads=1):
        super().__init__()
        self.global_heads = [GlobalHead(in_dim, attn_dim, proj_dim) for _ in range(n_global_heads)]
        self.local_heads = [LocalHead(in_dim, attn_dim, proj_dim) for _ in range(n_local_heads)]
        self.concat_proj = nn.Linear((n_global_heads+n_local_heads)*in_dim, in_dim) # W_O [hd_v, d_in]
    
    def forward(self, x, adj=None):
        global_attn_op = []
        local_attn_op = []

        local_attn_weights = []
        global_attn_weights = []

        for gh in self.global_heads:
            out, attn = gh(x)
            global_attn_weights.append(attn)
            global_attn_op.append(out)

        for lh in self.local_heads:
            out, attn = lh(x, adj)
            local_attn_weights.append(attn)
            local_attn_op.append(out)

        global_attn_op.extend(local_attn_op)
        global_attn_weights.extend(local_attn_weights)

        concat = torch.concat(global_attn_op, dim=-1)
        print (concat.shape)
        out = self.concat_proj(concat)

        return out, global_attn_weights

class TFEncoder(nn.Module):
    def __init__(self, n_layers):
        

# tud = TUDataset(root="../data/", use_edge_attr=True, use_node_attr=True, name="PROTEINS")
# graph = tud[0]
# print (graph)        

# mhsa = MHSA(graph.x.size(1), 64, graph.x.size(1), n_global_heads=4, n_local_heads=4)

# adj = tg.utils.to_dense_adj(graph.edge_index)
# y, all_weights = mhsa(graph.x.unsqueeze(0), adj)

# print (y.shape, type(all_weights), len(all_weights))

# fig = plt.figure()

# for i in range(0, len(all_weights) // 2):
#     fig.add_subplot(3, len(all_weights)//2, i+1)
#     plt.imshow(all_weights[i].squeeze(0).detach().numpy(), cmap="winter_r")
#     plt.title(f"global {i}")

# for i in range(len(all_weights) // 2, len(all_weights)):
#     fig.add_subplot(3, len(all_weights)//2, i+1)
#     plt.imshow(all_weights[i].squeeze(0).detach().numpy(), cmap="winter_r")
#     plt.title(f"local {i - len(all_weights)//2}")

# for i in range(0, len(all_weights)//2):
#     fig.add_subplot(3, len(all_weights)//2, len(all_weights)+i+1)
#     plt.imshow(adj.squeeze(0).detach().numpy(), cmap="winter_r")
#     plt.title(f"adj")
#     break

# fig.add_subplot(1, 3, 1)
# plt.imshow(torch.bmm(attn_l, adj).squeeze(0).detach().numpy(), cmap="ocean")
# plt.title("local attn")

# fig.add_subplot(132)
# plt.imshow(attn_g.squeeze(0).detach().numpy(), cmap="ocean")
# plt.title("global attn")

# fig.add_subplot(133)
# plt.imshow(adj.squeeze(0).detach().numpy(), cmap="ocean")
# plt.title("adj")

plt.show()


"""
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 
'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 
'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 
'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 
'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 
'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 
'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 
'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 
'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 
'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 
'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 
'viridis_r', 'winter', 'winter_r'
"""