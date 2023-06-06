from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch


class Dataset(Dataset):  # previously Dataset_V2
    """
    Args:

    nodes_embs: torch.Tensor, shape (#time_periods, #nodes, #embds_length)

    adj_matrix: torch.Tensor, shape (#time_periods, #relations, #nodes, #nodes)

    cov_matrix: torch.Tensor, shape (#time_periods, #nodes, #nodes)

    future_returns: torch.Tensor, shape (#time_periods, #nodes)

    Returns:

    nodes_embs: torch.Tensor, shape (#nodes, #embds_length)

    edge_index: torch.tensor, shape (2, #edges)

    edge_attr: torch.Tensor, shape (#edges)

    cov_matrix: torch.Tensor, shape (#nodes, #nodes)

    future_returns: torch.Tensor, shape (#nodes)
    """

    def __init__(self, nodes_embs, adj_matrix, cov_matrix, future_return):
        self.nodes_embs = torch.tensor(nodes_embs, dtype=torch.float32)
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)
        self.future_return = torch.tensor(future_return.values, dtype=torch.float32)

        self.convert_to_index()

        self.num_features = nodes_embs.shape[2]
        self.num_nodes = nodes_embs.shape[1]
        self.num_time_periods = nodes_embs.shape[0]
        self.num_relations = adj_matrix.shape[1]

    def convert_to_index(self):
        self.edge_time_index = self.adj_matrix.nonzero().t().contiguous()

    def __getitem__(self, idx):
        node_emb = self.nodes_embs[idx, :, :]

        edge_index = self.edge_time_index[2:, self.edge_time_index[0] == idx]

        edge_attr = self.adj_matrix[idx][self.adj_matrix[idx] != 0]

        edge_type = self.edge_time_index[1, self.edge_time_index[0] == idx]

        cov_matrix = self.cov_matrix[idx, :, :]

        future_return = self.future_return[idx, :]

        graph_data = Data(
            x=node_emb,
            edge_index=edge_index.type(torch.int64),
            edge_attr=edge_attr,
            edge_type=edge_type.type(torch.int64),
            cov_matrix=cov_matrix,
            future_return=future_return,
        )

        return graph_data

    def __len__(self):
        return self.num_time_periods


# class Dataset_v3(Dataset):
#     """
#     Args:

#     nodes_embs: torch.Tensor, shape (#time_periods, #nodes, #embds_length)

#     adj_matrix: torch.Tensor, shape (#time_periods, #relations, #nodes, #nodes)

#     cov_matrix: torch.Tensor, shape (#time_periods, #nodes, #nodes)

#     future_returns: torch.Tensor, shape (#time_periods, #nodes)

#     Returns:

#     nodes_embs: torch.Tensor, shape (#nodes, #embds_length)

#     edge_index: torch.tensor, shape (2, #edges)

#     edge_attr: torch.Tensor, shape (#edges)

#     cov_matrix: torch.Tensor, shape (#nodes, #nodes)

#     future_returns: torch.Tensor, shape (#nodes)
#     """

#     def __init__(self, nodes_embs, adj_matrix, cov_matrix, future_return, window=14):
#         self.nodes_embs = torch.tensor(nodes_embs, dtype=torch.float32)
#         self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
#         self.cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)
#         self.future_return = torch.tensor(future_return, dtype=torch.float32)

#         self.convert_to_index()

#         self.num_features = nodes_embs.shape[2]
#         self.num_nodes = nodes_embs.shape[1]
#         self.num_time_periods = nodes_embs.shape[0]
#         self.num_relations = adj_matrix.shape[1]
#         self.window = window

#     def convert_to_index(self):
#         self.edge_time_index = self.adj_matrix.nonzero().t().contiguous()

#     def __getitem__(self, idx):
#         node_emb = self.nodes_embs[idx : idx + self.window, :, :]

#         mask = (self.edge_time_index[0] >= idx) & (
#             self.edge_time_index[0] < idx + self.window
#         )

#         edge_index = self.edge_time_index[2:, mask]

#         edge_attr = self.adj_matrix[idx : idx + self.window][
#             self.adj_matrix[idx : idx + self.window] != 0
#         ]

#         edge_type = self.edge_time_index[1, mask]

#         cov_matrix = self.cov_matrix[idx : idx + self.window, :, :]

#         future_return = self.future_return[idx : idx + self.window, :]

#         # graph_data = Data(
#         #     x=node_emb,
#         #     edge_index=edge_index.type(torch.int64),
#         #     edge_attr=edge_attr,
#         #     edge_type=edge_type.type(torch.int64),
#         #     cov_matrix=cov_matrix,
#         #     future_return=future_return,
#         # )

#         return node_emb, edge_index, edge_attr, edge_type, cov_matrix, future_return

#     def __len__(self):
#         return self.num_time_periods - self.window
