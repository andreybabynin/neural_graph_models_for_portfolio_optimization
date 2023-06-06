import torch
from torch_geometric.nn import RGATConv, RGCNConv
from src.optimizers import *
from src.loss_functions import *
from src.utils import *
import random

# class RGATModel_v1(torch.nn.Module):
#     def __init__(self, num_features, num_relations):
#         super().__init__()

#         self.rgat1 = RGATConv(
#             in_channels=num_features,
#             out_channels=10,
#             num_relations=num_relations,
#             edge_dim=1,
#         )
#         self.rgat2 = RGATConv(
#             in_channels=10, out_channels=1, num_relations=num_relations, edge_dim=1
#         )
#         self.dropout = torch.nn.Dropout(0.5)
#         self.norm1 = torch.nn.BatchNorm1d(10)
#         self.elu = torch.nn.ELU()

#     def forward(self, node_embs, edge_index, edge_attr, edge_types):
#         # remove batch dimension
#         node_embs = node_embs.squeeze(0)
#         edge_index = edge_index.squeeze(0)
#         edge_attr = edge_attr.squeeze(0)
#         edge_types = edge_types.squeeze(0)

#         x = self.rgat1(node_embs, edge_index, edge_types, edge_attr)
#         x = self.elu(x)
#         x = self.dropout(x)
#         x = self.norm1(x)
#         x = self.rgat2(x, edge_index, edge_types, edge_attr)

#         return x.squeeze(1)


# class RGCNModel_v1(torch.nn.Module):
#     def __init__(self, num_features, num_relations):
#         super().__init__()

#         self.rgcn1 = RGCNConv(
#             in_channels=num_features, out_channels=10, num_relations=num_relations
#         )
#         self.rgcn2 = RGCNConv(
#             in_channels=10, out_channels=1, num_relations=num_relations
#         )
#         self.dropout = torch.nn.Dropout(0.5)
#         self.norm1 = torch.nn.BatchNorm1d(10)
#         self.elu = torch.nn.ELU()

#     def forward(self, node_embs, edge_index, edge_types):
#         # remove batch dimension
#         node_embs = node_embs.squeeze(0)
#         edge_index = edge_index.squeeze(0)
#         edge_types = edge_types.squeeze(0)

#         x = self.rgcn1(node_embs, edge_index, edge_types)
#         x = self.elu(x)
#         x = self.dropout(x)
#         x = self.norm1(x)
#         x = self.rgcn2(x, edge_index, edge_types)

#         return x.squeeze(1)


# class RGATModel_v3(torch.nn.Module):
#     def __init__(self, num_features, num_relations, train_gamma=True, batch_size=7):
#         super().__init__()

#         self.rgat1 = RGATConv(
#             in_channels=num_features,
#             out_channels=10,
#             num_relations=num_relations,
#             edge_dim=1,
#             heads=1,
#             dropout=0.5,
#         )
#         self.rgat2 = RGATConv(
#             in_channels=10,
#             out_channels=1,
#             num_relations=num_relations,
#             edge_dim=1,
#             heads=1,
#             dropout=0.5,
#         )

#         self.gamma = torch.nn.Parameter(
#             torch.FloatTensor(1).uniform_(0.01, 0.1), requires_grad=train_gamma
#         )
#         self.dropout = torch.nn.Dropout(0.5)
#         self.norm1 = torch.nn.BatchNorm1d(10)
#         self.elu = torch.nn.ELU()
#         self.batch_size = batch_size

#         self.optim_layer = eval("base_optim")(17, self.batch_size, eval("p_var"))

#     def forward(
#         self, node_emb, edge_index, edge_types, edge_attr, future_return, cov_matrix
#     ):
#         x = self.rgat1(node_emb, edge_index, edge_types, edge_attr)
#         x = self.norm1(x)
#         x = self.elu(x)
#         x = self.dropout(x)

#         y_h = self.rgat2(x, edge_index, edge_types, edge_attr)

#         y_h = y_h.reshape(self.batch_size, -1)

#         resid = future_return.reshape(self.batch_size, -1) - y_h

#         y_h = y_h[-1]  # last prediciton

#         w = self.optim_layer(resid, y_h, self.gamma)

#         return w


class RGATModel_v3_1(torch.nn.Module):
    def __init__(
        self, num_features, num_relations, num_assets, sample_size=20, train_gamma=True
    ):
        super().__init__()

        self.rgat1 = RGATConv(
            in_channels=num_features,
            out_channels=10,
            num_relations=num_relations,
            edge_dim=1,
            heads=1,
            dropout=0.5,
        )
        self.rgat2 = RGATConv(
            in_channels=10,
            out_channels=1,
            num_relations=num_relations,
            edge_dim=1,
            heads=1,
            dropout=0.5,
        )

        self.gamma = torch.nn.Parameter(
            torch.FloatTensor(1).uniform_(0.01, 0.1), requires_grad=train_gamma
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.norm1 = torch.nn.BatchNorm1d(10)
        self.elu = torch.nn.ELU()
        self.sample_size = sample_size
        self.num_assets = num_assets

        self.optim_layer = eval("base_optim")(num_assets, sample_size, eval("p_var"))

    def pred_layer(
        self, node_emb, edge_index, edge_types, edge_attr, future_return, cov_matrix
    ):
        x = self.rgat1(node_emb, edge_index, edge_types, edge_attr)
        x = self.norm1(x)
        x = self.elu(x)
        x = self.dropout(x)

        y_h = self.rgat2(x, edge_index, edge_types, edge_attr)

        return y_h

    def forward(self, dataset, idx):
        resid = []

        for i in range(self.sample_size):
            batch = dataset[idx + i]
            node_emb = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            edge_type = batch.edge_type
            cov_matrix = batch.cov_matrix
            future_ret = batch.future_return

            y_h = self.pred_layer(
                node_emb, edge_index, edge_type, edge_attr, future_ret, cov_matrix
            )

            y_h = y_h.reshape(-1)

            resid_idx = future_ret.reshape(-1) - y_h
            resid.append(resid_idx)

        w = self.optim_layer(resid, y_h, self.gamma)

        return w


class RGATModel_v3_2(torch.nn.Module):
    def __init__(self, num_features, num_relations, num_assets, n_heads=1):
        super().__init__()

        self.rgat1 = RGATConv(
            in_channels=num_features,
            out_channels=15,
            num_relations=num_relations,
            edge_dim=1,
            heads=n_heads,
            dropout=0.5,
            concat=True,
            attention_mechanism="across-relation",
            attention_mode="additive-self-attention",
        )
        self.rgat2 = RGATConv(
            in_channels=15 * n_heads,
            out_channels=10,
            num_relations=num_relations,
            edge_dim=1,
            heads=n_heads,
            dropout=0.5,
            concat=False,
            attention_mechanism="across-relation",
            attention_mode="additive-self-attention",
        )
        self.linear1 = torch.nn.Linear(10, 1)

        self.dropout = torch.nn.Dropout(0.5)
        self.norm1 = torch.nn.BatchNorm1d(15 * n_heads)
        self.norm2 = torch.nn.BatchNorm1d(10)
        self.elu = torch.nn.ELU()
        self.num_assets = num_assets

    def forward(self, node_emb, edge_index, edge_types, edge_attr):
        x = self.rgat1(node_emb, edge_index, edge_types, edge_attr)

        x = self.norm1(x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.rgat2(x, edge_index, edge_types, edge_attr)

        x = self.norm2(x)
        x = self.elu(x)
        x = self.dropout(x)

        y_h = self.linear1(x)  # (#nodes, 1)

        return y_h


class GrossModel(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_relations,
        num_assets,
        sample_size,
        pred_window,
        n_heads=1,
        replay_prob=0.5,
        gamma=0.05,
        train_gamma=True,
        storage_size=100,
        optimizer="base_optim",
        prisk="p_var",
        min_weight=0,
        max_weight=0.25,
    ):
        super(GrossModel, self).__init__()

        self.pred_layer = RGATModel_v3_2(
            num_features, num_relations, num_assets, n_heads=n_heads
        )
        self.optim_layer = eval(optimizer)(
            num_assets,
            sample_size - 1,
            eval(prisk),
            max_weight=max_weight,
            min_weight=min_weight,
        )
        self.storage = LossStorage_v2(storage_size)
        self.sample_size = sample_size
        self.pred_window = pred_window
        self.replay_prob = replay_prob
        self.train_gamma = train_gamma

        if not train_gamma:
            self.gamma = torch.nn.Parameter(
                torch.FloatTensor([gamma]), requires_grad=False
            )
        else:
            self.gamma = torch.nn.Parameter(
                torch.FloatTensor([gamma]), requires_grad=True
            )

        self.mode = "Train"

    def select_batch(self, step, idx):
        if step == self.sample_size - 1:
            # take last prediction step
            idx_new = idx + self.sample_size + self.pred_window - 1
        else:
            # sample from storage
            if (
                len(self.storage) > self.sample_size + self.pred_window
                and random.random() < self.replay_prob
                and self.mode == "Train"
            ):
                idx_new = self.storage.sample()
            else:
                idx_new = idx + step

        return idx_new

    def stack_resid(self, step, future_ret, y_h):
        if step < self.sample_size - 1:
            resid_idx = (future_ret.reshape(-1) - y_h).reshape(1, -1)

            if step == 0:
                self.resid = resid_idx
            else:
                self.resid = torch.cat((self.resid, resid_idx), 0)

    def forward(self, dataset, idx):
        for i in range(self.sample_size):
            idx_new = self.select_batch(i, idx)

            batch = dataset[idx_new]
            node_emb = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            edge_type = batch.edge_type
            future_ret = batch.future_return

            y_h = self.pred_layer(node_emb, edge_index, edge_type, edge_attr)
            y_h = y_h.reshape(-1)

            # stack residuals for all samples but last one
            self.stack_resid(i, future_ret, y_h)

        w = self.optim_layer(self.resid, y_h, self.gamma)[0]

        return w, idx_new


class BareModel(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_relations,
        num_assets,
        pred_window,
        n_heads=1,
    ):
        super(BareModel, self).__init__()

        self.pred_layer = RGATModel_v3_2(
            num_features, num_relations, num_assets, n_heads=n_heads
        )
        self.pred_window = pred_window
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, dataset, idx):
        batch = dataset[idx]
        node_emb = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        edge_type = batch.edge_type

        y_h = self.pred_layer(node_emb, edge_index, edge_type, edge_attr)
        w = self.softmax(y_h)

        return w, idx
