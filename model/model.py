"""Model definition for the IEST architecture."""

from argparse import Namespace

import torch
import torch.nn as nn

from .layers import EmbeddingLayer, CrossGatedLayer, GraphIntegratedLayer


class Model(nn.Module):
    """Framework of IEST Model"""
    def __init__(self, args: Namespace, adj: torch.Tensor) -> None:
        super().__init__()
        # Layers for Cross-Gated Unit
        self.embed_layer = EmbeddingLayer(args.node_num, args.hidden_dim, args.dropout)
        self.gated_layers = nn.ModuleList([
            CrossGatedLayer(args.hidden_dim, args.dropout)
            for _ in range(args.layer_num)])
        self.out_layer_t = nn.Linear(2 * args.hidden_dim, 1)
        # Layers for Graph-Integrated Unit
        self.input_layer = nn.Linear(1, args.hidden_dim)
        self.fwd_layers = nn.ModuleList([
            GraphIntegratedLayer(args.seq_len, args.hidden_dim, adj, args.alpha, args.order, args.dropout)
            for _ in range(args.layer_num)])
        self.bwd_layers = nn.ModuleList([
            GraphIntegratedLayer(args.seq_len, args.hidden_dim, adj, args.alpha, args.order, args.dropout)
            for _ in range(args.layer_num)])
        self.out_layer_fwd = nn.Linear(args.hidden_dim, 1)
        self.out_layer_bwd = nn.Linear(args.hidden_dim, 1)

    def forward(self, x, m, t) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: batch, length, node, 1
        :param m: batch, length, node, 1
        :param t: batch, length, 1, 4
        :return: batch, length, node, 1
        """
        x_spatio, x_temporal = self.embed_layer(t)
        x_fwd, m_fwd = self.input_layer(x), m
        x_bwd, m_bwd = torch.flip(self.input_layer(x), dims=[1]), torch.flip(m, dims=[1])
        for gate, fwd, bwd in zip(self.gated_layers, self.fwd_layers, self.bwd_layers):
            x_spatio, x_temporal = gate(x_spatio, x_temporal)  # batch, length, node, hidden_dim
            h_fwd = (x_spatio + x_temporal)[:, 0]  # Select the first time step as initial hidden state
            h_bwd = (x_spatio + x_temporal)[:, -1]
            x_fwd = fwd(x_fwd, m_fwd, h_fwd)
            x_bwd = bwd(x_bwd, m_bwd, h_bwd)
            m_fwd = torch.ones_like(m).bool()
            m_bwd = torch.ones_like(m).bool()
        x_fwd = self.out_layer_fwd(x_fwd[:, :-1])
        x_bwd = self.out_layer_bwd(torch.flip(x_bwd, dims=[1])[:, 1:])
        x_hat = torch.cat([x_bwd[:, :1], (x_fwd[:, :-1] + x_bwd[:, 1:]) / 2, x_fwd[:, -1:]], dim=1)
        t_hat = self.out_layer_t(torch.cat([x_spatio, x_temporal], dim=-1))
        return x_hat, t_hat
