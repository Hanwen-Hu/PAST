"""Components of the IEST model including embedding, cross-gating and GCN layers."""

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Layer to embed node IDs and time features."""
    def __init__(self, node_num: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        assert hidden_dim % 8 == 0
        self.node_num = node_num
        self.node_embedding = nn.Parameter(torch.randn(node_num, hidden_dim))
        self.week_embedding = nn.Embedding(7, hidden_dim // 8 * 3)
        self.hour_embedding = nn.Embedding(24, hidden_dim // 8 * 3)
        self.minute_embedding = nn.Embedding(4, hidden_dim // 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param t: batch, length, 1, 4
        :return: batch, length, node, hidden_dim
        """
        batch, length, _, _ = t.shape
        x_s = self.node_embedding.reshape(1, 1, self.node_num, -1).expand(batch, length, -1, -1)
        x_w = self.week_embedding(t[:, :, :, 1])  # bath, length, 1, dim
        x_h = self.hour_embedding(t[:, :, :, 2])
        x_m = self.minute_embedding(t[:, :, :, 3])
        x_t = torch.cat([x_w, x_h, x_m], dim=-1).expand(-1, -1, self.node_num, -1)
        return self.dropout(x_s), self.dropout(x_t)


class CrossGatedLayer(nn.Module):
    """Layer to perform cross-gating between spatial and temporal features."""
    def __init__(self, hidden_dim: int, dropout: float=0.1) -> None:
        super().__init__()
        self.gate_layer_s = nn.Linear(hidden_dim, hidden_dim)
        self.gate_layer_t = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_layer_s = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_layer_t = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x_s: batch, length, node, hidden_dim
        :param x_t: batch, length, node, hidden_dim
        :return: batch, length, node, hidden_dim
        """
        res_s, res_t = x_s, x_t
        g_s = self.gate_layer_s(x_s)
        g_t = self.gate_layer_t(x_t)
        c_s = self.candidate_layer_s(x_s)
        c_t = self.candidate_layer_t(x_t)
        x_s = torch.relu(c_s) * torch.sigmoid(g_s) * torch.tanh(g_t)
        x_t = torch.relu(c_t) * torch.sigmoid(g_t) * torch.tanh(g_s)
        return self.dropout(x_s) + res_s, self.dropout(x_t) + res_t


class TemporalGCNLayer(nn.Module):
    """Layer to perform temporal graph convolution."""
    def __init__(self, seq_len: int, hidden_dim: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.hidden_weight = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.temporal_weight = nn.Parameter(torch.randn(1, seq_len, seq_len))
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.dropout = self._calc_drop_prob(alpha, dropout)     
    
    def _calc_drop_prob(self, alpha: float, dropout: float) -> torch.Tensor:
        row = torch.arange(self.seq_len).reshape(-1, 1)
        col = torch.arange(self.seq_len).reshape(1, -1)
        interval = torch.abs(row - col)
        beta = torch.log(dropout * self.seq_len * self.seq_len / torch.sum(torch.exp(-alpha * interval)))
        return torch.exp(-alpha * interval + beta)

    def _drop_adj(self, m: torch.Tensor) -> torch.Tensor:
        """
        :param m: batch * node, 1, length
        :return: batch * node, length, length + hidden_dim
        """
        batch_node = m.shape[0]
        weight = self.temporal_weight * m  # 缺失位置不可达 batch * node, length, length
        if self.training:
            drop_mask = torch.rand_like(self.dropout) > self.dropout
            weight = weight * drop_mask.unsqueeze(0)
        weight = torch.cat([self.hidden_weight.expand(batch_node, -1, -1), weight], dim=-1)
        return weight / (torch.sum(torch.abs(weight), dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch, length, node, hidden_dim
        :param m: batch, length, node, 1
        :param h: batch, node, hidden_dim
        :return: batch, length, node, hidden_dim
        """
        res_x = x
        batch, length, node, hidden_dim = x.shape
        x = x.permute(0, 2, 1, 3)  # batch, node, length, hidden_dim
        m = m.permute(0, 2, 1, 3)  # batch, node, length, 1
        adj = self._drop_adj(m.reshape(-1, 1, length))
        adj = adj.reshape(batch, node, length, -1)  # batch, node, length, length + hidden_len
        h = h.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)  # batch node, hidden_dim, hidden_dim
        x = torch.cat([h, x], dim=2)  # batch, node, length + hidden_dim, hidden_dim
        x = torch.einsum('bnlk, bnkd -> bnld', adj, x)  # batch, node, length, hidden_dim
        x = self.ff(x).permute(0, 2, 1, 3)
        m = m.permute(0, 2, 1, 3)
        return x + x * ~m + res_x * m


class SpatioGCNLayer(nn.Module):
    """Layer to perform spatial graph convolution."""
    def __init__(self, hidden_dim: int, adj: torch.Tensor, order: int, dropout: float) -> None:
        super().__init__()
        self.order = order
        row_degree = torch.sum(torch.abs(adj), dim=-1, keepdim=True) + 1e-6
        col_degree = torch.sum(torch.abs(adj), dim=-2, keepdim=True) + 1e-6
        self.adj = adj / torch.sqrt(row_degree * col_degree)  # node, node
        self.ff = nn.Sequential(nn.Linear(hidden_dim * (order + 1), hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch, length, node, hidden_dim
        :return: batch, length, node, hidden_dim
        """
        result = [x]
        for _ in range(self.order):
            result.append(torch.einsum('mn, blnd -> blmd', self.adj, result[-1]))
        return self.ff(torch.cat(result, dim=-1)) + x  # batch, length, node, dims


class GraphIntegratedLayer(nn.Module):
    """Layer to integrate temporal and spatial graph convolutions."""
    def __init__(self, seq_len: int, hidden_dim: int, adj: torch.Tensor, alpha: float, order: int, dropout: float) -> None:
        super().__init__()
        self.temporal_layer = TemporalGCNLayer(seq_len, hidden_dim, alpha, dropout)
        self.spatio_layer = SpatioGCNLayer(hidden_dim, adj, order, dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch, length, node, hidden_dim
        :param m: batch, length, node, 1
        :param h: batch, node, hidden_dim
        :return: batch, length, node, hidden_dim
        """
        x = self.temporal_layer(x, m, h)
        x = self.spatio_layer(x)
        return x
