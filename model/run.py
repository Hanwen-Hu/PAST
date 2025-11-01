"""PAST Model Trainer and Tester"""

from argparse import Namespace
from math import sqrt

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_loader import TrafficData, TrafficDataConfig
from .model import Model


class PAST:
    """PAST Model Trainer and Tester"""
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.path = f'files/{args.dataset}_{(args.miss_rate * 10):.0f}_{args.miss_len}_{args.miss_span}.pth'
        print(f"Dataset: {args.dataset}\nRate: {args.miss_rate}\tLength: {args.miss_len}\tSpan: {args.miss_span}\t")

        train_config = TrafficDataConfig(args.dataset, args.seq_len, 'offline', args.miss_rate, args.miss_len, args.miss_span, args.device)
        test_config = TrafficDataConfig(args.dataset, args.seq_len, 'online', args.miss_rate, args.miss_len, args.miss_span, args.device)
        self.train_set = TrafficData(train_config)
        self.test_set = TrafficData(test_config)
        self.train_loader = DataLoader(self.train_set, args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, args.batch_size, shuffle=True)

        self.model = Model(args, self.train_set.adj).to(args.device)
        if args.load:
            self.model.load_state_dict(torch.load(self.path, map_location=args.device))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = lambda x, y: torch.mean(torch.abs(x - y))

        self.valid_mask = self._valid_mask(args.batch_size, args.seq_len, args.node_num)  # batch, length, node, 1

    def _valid_mask(self, batch, length, node) -> torch.Tensor:
        # batch, length, node, 1
        mask_index = torch.randint(0, length, (batch * node, 2), device=self.args.device)  # batch * node, 2
        mask_index = torch.sort(mask_index, dim=1)[0]
        mask = torch.ones(batch * node, length, device=self.args.device).bool()
        for i in range(batch * node):
            mask[i, mask_index[i, 0]:mask_index[i, 1]] = False
        return mask

    def _train_epoch(self):
        train_loss, valid_loss = 0, 0
        for x, _, m, t in tqdm(self.train_loader):
            # Validation Set
            x[~m] = 0
            if self.args.miss_len > 1:
                index = torch.randperm(m.shape[0] * m.shape[2])
                valid_m = torch.ones_like(self.valid_mask).bool()
                valid_m[index[:index.shape[0] // 2]] = self.valid_mask[:index.shape[0] // 2]
                valid_m = valid_m[:m.shape[0] * m.shape[2]].reshape(m.shape[0], m.shape[2], m.shape[1], 1).permute(0, 2, 1, 3)
            else:
                valid_m = torch.rand(m.shape, device=self.args.device) > 0.1
            valid_m[~m] = True
            mask = m * valid_m

            # Begin Training
            self.optimizer.zero_grad()
            x_hat, t_hat = self.model(x, mask, t)
            loss = self.mse_loss(x_hat[m], x[m]) + self.mse_loss((x_hat.detach() + t_hat)[m], x[m])
            loss.backward()
            self.optimizer.step()
            y_hat = x_hat + t_hat
            train_loss += self.mse_loss(y_hat[m], x[m]).item()
            valid_loss += self.mse_loss(y_hat[~valid_m], x[~valid_m]).item()
        return sqrt(train_loss / len(self.train_loader)), sqrt(valid_loss / len(self.train_loader))

    def _test_epoch(self, mode: str = 'offline'):
        rmse, mae = 0, 0
        loader = self.train_loader if mode == 'offline' else self.test_loader
        for x, y, m, t in tqdm(loader):
            x[~m] = 0
            x_hat, t_hat = self.model(x, m, t)
            y_hat = x_hat + t_hat
            rmse += self.mse_loss(y_hat[~m], y[~m]).item()
            mae += self.mae_loss(y_hat[~m], y[~m]).item()
        return sqrt(rmse / len(loader)), mae / len(loader)

    def train(self):
        """Train the PAST model."""
        self.model.train()
        best_valid, patience = float('inf'), 0
        for epoch in range(self.args.epochs):
            train_loss, valid_loss = self._train_epoch()
            if valid_loss < best_valid:
                best_valid = valid_loss
                patience = 0
                torch.save(self.model.state_dict(), self.path)
            else:
                patience += 1
            print(f'Epoch: {epoch}\tTrain RMSE: {train_loss:.4f}\tValid RMSE: {valid_loss:.4f}\tPatience:{patience}')
            if patience >= 5:
                break

    def test(self):
        """Test the PAST model."""
        self.model.load_state_dict(torch.load(self.path, map_location=self.args.device))
        self.model.eval()
        with torch.no_grad():
            offline_rmse, offline_mae = self._test_epoch('offline')
            online_rmse, online_mae = self._test_epoch('online')
        content = f'{self.args.dataset},{self.args.miss_rate},{self.args.miss_len},{self.args.miss_span},Result: {offline_rmse:.4f},{offline_mae:.4f},{online_rmse:.4f},{online_mae:.4f}\n'
        with open('result.txt', 'a', encoding='utf-8') as file:
            file.write(content)
        print(f'Dataset: {self.args.dataset}\nRate: {self.args.miss_rate}\tLength: {self.args.miss_len}\tSpan: {self.args.miss_span}')
        print(f'Offline RMSE: {offline_rmse:.4f}\tOffline MAE: {offline_mae:.4f}\tOnline MSE: {online_rmse:.4f}\tOnline MAE: {online_mae:.4f}')
