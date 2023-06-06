import pandas as pd
import numpy as np
import torch

# import heapq
import random
from prettytable import PrettyTable

# class LossStorage:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.data = []

#     def add(self, loss, index):
#         # If we haven't reached capacity yet, just add the new element
#         if len(self.data) < self.capacity:
#             heapq.heappush(self.data, (loss, index))
#         else:
#             # If we're at capacity, remove the smallest element if the new element is larger
#             heapq.heappushpop(self.data, (loss, index))

#     def softmax(self, x):
#         # Compute the softmax of vector x in a numerically stable way
#         exps = np.exp(x)
#         return exps / np.sum(exps)

#     def sample(self):
#         # Normalize the losses to form a probability distribution
#         losses, indices = zip(*self.data)
#         probs = self.softmax(losses)

#         # Sample an index based on the probability distribution
#         index = random.choices(indices, weights=probs, k=1)[0]

#         return index

#     def __len__(self):
#         return len(self.data)


class LossStorage_v2:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}  # Change to a dictionary

    def add(self, loss, index):
        # If we haven't reached capacity yet, just add the new element
        if len(self.data) < self.capacity:
            self.data[index] = loss
        else:
            # If we're at capacity, remove the smallest element if the new element is larger
            min_index = min(self.data, key=self.data.get)
            if loss > self.data[min_index]:
                del self.data[min_index]
                self.data[index] = loss

    def softmax(self, x):
        # Compute the softmax of vector x in a numerically stable way
        exps = np.exp(x)
        return exps / np.sum(exps)

    def sample(self):
        # Normalize the losses to form a probability distribution
        indices, losses = zip(*self.data.items())
        probs = self.softmax(losses)

        # Sample an index based on the probability distribution
        index = random.choices(indices, weights=probs, k=1)[0]

        return index

    def __len__(self):
        return len(self.data)


def get_data(path, stocks):
    df_adj_close, df_close, df_high, df_low, df_volume = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    for i, n in enumerate(stocks):
        full_path = path + "\\" + n + ".csv"
        df_temp = pd.read_csv(full_path, index_col=False).rename(
            columns={
                "Adj Close": f"Adj_Close_{n}",
                "Close": f"Close_{n}",
                "High": f"High_{n}",
                "Low": f"Low_{n}",
                "Volume": f"Volume_{n}",
            }
        )

        print(n, "Start of history:", df_temp["Date"].min())

        if i == 0:
            df_adj_close = df_temp[["Date", f"Adj_Close_{n}"]]
            df_close = df_temp[["Date", f"Close_{n}"]]
            df_high = df_temp[["Date", f"High_{n}"]]
            df_low = df_temp[["Date", f"Low_{n}"]]
            df_volume = df_temp[["Date", f"Volume_{n}"]]

        else:
            df_adj_close = pd.merge(
                df_adj_close,
                df_temp[["Date", f"Adj_Close_{n}"]],
                how="inner",
                on="Date",
            )
            df_close = pd.merge(
                df_close, df_temp[["Date", f"Close_{n}"]], how="inner", on="Date"
            )
            df_high = pd.merge(
                df_high, df_temp[["Date", f"High_{n}"]], how="inner", on="Date"
            )
            df_low = pd.merge(
                df_low, df_temp[["Date", f"Low_{n}"]], how="inner", on="Date"
            )
            df_volume = pd.merge(
                df_volume, df_temp[["Date", f"Volume_{n}"]], how="inner", on="Date"
            )

    for df in [df_adj_close, df_close, df_high, df_low, df_volume]:
        df.index = pd.to_datetime(df.Date)

        df.drop("Date", axis=1, inplace=True)

    return df_adj_close, df_close, df_high, df_low, df_volume


def get_return(df_adj_close, period=1, future=False):
    df_return = df_adj_close.pct_change(period)

    if future:
        df_return = df_return.shift(-period)

    df_return.dropna(inplace=True)
    df_return.index = pd.to_datetime(df_return.index)

    return df_return


def common_index(*args):
    df_list = args

    index = set(df_list[0].index)
    for df in args[1:]:
        index = index.intersection(set(df.index))

    return sorted(index)


def concat_adj_matrices(*args):
    adj_list = []
    for adj in args:
        adj_list.append(np.expand_dims(adj, 1))

    return np.concatenate(adj_list, 1)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
