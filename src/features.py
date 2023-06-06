from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, VortexIndicator
from ta.volatility import UlcerIndex
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from src.utils import *
from sklearn.preprocessing import StandardScaler


def normalize(matrix):
    return (
        matrix / (np.nanmax(matrix) - np.nanmin(matrix)),
        np.nanmax(matrix),
        np.nanmin(matrix),
    )


def rsi_features(df_close, stocks, periods=[14, 28]):
    df_rsi = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for w in periods:
            df_rsi[f"{s}_RSI_{w}"] = RSIIndicator(
                df_close[f"Close_{s}"], window=w, fillna=False
            ).rsi()

    return df_rsi


def macd_features(
    df_close, stocks, periods_slow=[26, 36], periods_fast=[12, 18], periods_sign=[9, 12]
):
    df_macd = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for sl, f, i in zip(periods_slow, periods_fast, periods_sign):
            df_macd[f"{s}_MACD_{sl}"] = MACD(
                close=df_close[f"Close_{s}"],
                window_slow=sl,
                window_fast=f,
                window_sign=i,
                fillna=False,
            ).macd()

    return df_macd


def stochastic_features(df_close, df_high, df_low, stocks, periods=[14, 28]):
    df_so = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for w in periods:
            df_so[f"{s}_SO_{w}"] = StochasticOscillator(
                high=df_high[f"High_{s}"],
                low=df_low[f"Low_{s}"],
                close=df_close[f"Close_{s}"],
                window=w,
                fillna=False,
            ).stoch()

    return df_so


def wiliams_r_features(df_close, df_high, df_low, stocks, periods=[14, 28]):
    df_wo = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for w in periods:
            df_wo[f"{s}_WO_{w}"] = WilliamsRIndicator(
                high=df_high[f"High_{s}"],
                low=df_low[f"Low_{s}"],
                close=df_close[f"Close_{s}"],
                lbp=w,
                fillna=False,
            ).williams_r()

    return df_wo


def vortex_features(df_close, df_high, df_low, stocks, periods=[14, 28]):
    df_vortex = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for w in periods:
            df_vortex[f"{s}_ADX_{w}"] = VortexIndicator(
                high=df_high[f"High_{s}"],
                low=df_low[f"Low_{s}"],
                close=df_close[f"Close_{s}"],
                window=w,
                fillna=False,
            ).vortex_indicator_diff()

    return df_vortex


def ulcer_features(df_close, stocks, periods=[14, 28]):
    df_ulcer = pd.DataFrame(index=df_close.index)

    for s in stocks:
        for w in periods:
            df_ulcer[f"{s}_Ulcer_{w}"] = UlcerIndex(
                close=df_close[f"Close_{s}"], window=w, fillna=False
            ).ulcer_index()

    return df_ulcer


def get_features(df_close, df_high, df_low, stocks):
    df_rsi = rsi_features(df_close, stocks)
    df_macd = macd_features(df_close, stocks)
    df_so = stochastic_features(df_close, df_high, df_low, stocks)
    df_wo = wiliams_r_features(df_close, df_high, df_low, stocks)
    df_vortex = vortex_features(df_close, df_high, df_low, stocks)
    df_ulcer = ulcer_features(df_close, stocks)

    return pd.concat(
        [df_rsi, df_macd, df_so, df_wo, df_vortex, df_ulcer], axis=1, join="inner"
    )


def get_corr_n_cov(df_return, period=90):
    df_corr = df_return.rolling(period).corr().reset_index()
    df_corr.index = df_corr.Date
    df_corr.drop("Date", axis=1, inplace=True)
    df_corr = df_corr.dropna()

    df_cov = df_return.rolling(period).cov().reset_index()
    df_cov.index = pd.to_datetime(df_cov.Date)
    df_cov.drop("Date", axis=1, inplace=True)
    df_cov = df_cov.dropna()

    return df_corr, df_cov


# def get_index(path):
#     df_index = pd.read_parquet(path)
#     df_index.index = df_index.Date

#     return df_index


# def get_corr_index(df_return, df_index, period=90):
#     df = pd.merge(
#         df_return,
#         df_index["Close"].pct_change(),
#         how="inner",
#         left_index=True,
#         right_index=True,
#     )

#     df_corr = df.rolling(period).corr().reset_index()
#     df_corr.index = df_corr.Date
#     df_corr.drop("Date", axis=1, inplace=True)
#     df_corr = df_corr.dropna()

#     return df_corr


# def get_adj_matrix_index(df_corr, timestamps, n, threshold=0.1):
#     adj_matrix = np.zeros((len(timestamps), n, n))

#     for t in tqdm(range(len(timestamps))):
#         corr = df_corr.loc[timestamps[t]]

#         values = (
#             corr.iloc[:-1, -1].where(corr.iloc[:-1, -1].abs() > threshold).fillna(0)
#         )

#         for i in range(n):
#             for j in range(n):
#                 if i != j and values[i] != 0 and values[j] != 0:
#                     adj_matrix[t, i, j] = (values[i] + values[j]) / 2
#                 elif i != j:
#                     adj_matrix[t, i, j] = 0
#                 else:
#                     adj_matrix[t, i, j] = 1

#     return adj_matrix


# def get_corr_adj_matrix(df_corr, n_stocks, threshold=0.5):
#     temp = df_corr.iloc[:, 1:].abs().values

#     temp[temp < threshold] = 0

#     corr_adj_matrix = temp.reshape(-1, n_stocks, n_stocks)

#     return corr_adj_matrix


def get_cov_adj_matrix(df_cov, n_stocks, period=1):
    temp = df_cov.iloc[:, 1:].values

    cov_adj_matrix = temp.reshape(-1, n_stocks, n_stocks)

    return cov_adj_matrix * period


def get_nodes_matrix(df_features, stocks, num_features):
    index = df_features.index

    nodes_matrix = np.zeros((len(stocks), len(index), int(num_features)))

    for j, stock in enumerate(stocks):
        nodes_matrix[j] = df_features.loc[
            :, df_features.columns.str.contains(stock)
        ].values

    return np.swapaxes(nodes_matrix, 0, 1)


def get_quadrant(row, stock, short_period=28, long_period=90):
    if (
        row[f"Adj_Close_{stock}_{short_period}"] > 0
        and row[f"Adj_Close_{stock}_{long_period}"] > 0
    ):
        return 0

    elif (
        row[f"Adj_Close_{stock}_{short_period}"] > 0
        and row[f"Adj_Close_{stock}_{long_period}"] < 0
    ):
        return 1

    elif (
        row[f"Adj_Close_{stock}_{short_period}"] < 0
        and row[f"Adj_Close_{stock}_{long_period}"] > 0
    ):
        return 2

    else:
        return 3


def get_adj_matrix(df, n, stocks):
    matrix = np.zeros((len(df), n, n))

    for i in tqdm(range(len(df))):
        temp = np.zeros((n, n))

        for j in range(n):
            for k in range(j, n):
                if df.iloc[i][f"{stocks[j]}"] == df.iloc[i][f"{stocks[k]}"]:
                    temp[j][k] = 1

        matrix[i] = temp + temp.T - 2 * np.diag(np.diag(temp))

    return matrix


# def get_rsi_segment(row, stock, period=14):
#     if row[f"{stock}_RSI_{period}"] < 30:
#         return 2

#     elif row[f"{stock}_RSI_{period}"] > 70:
#         return 0

#     else:
#         return 1


def minimum_spanning_tree_matrix(df_corr):
    numStocks = df_corr.shape[1] - 1
    numPeriods = int(df_corr.shape[0] / numStocks)
    matrix = np.zeros((numPeriods, numStocks, numStocks))
    index = df_corr.index

    for i in tqdm(range(numPeriods)):
        corr = df_corr.loc[index[i]].iloc[:, 1:].values
        g_distance = np.nan_to_num(np.sqrt(0.5 * (1 - corr)), 0)
        G = nx.from_numpy_array(g_distance)
        T = nx.minimum_spanning_tree(G)
        matrix[i] = nx.to_numpy_array(T)

    return matrix


def maxim_planar_tree(df_corr):
    numStocks = df_corr.shape[1] - 1
    numPeriods = int(df_corr.shape[0] / numStocks)
    matrix = np.zeros((numPeriods, numStocks, numStocks))
    index = df_corr.index

    for i in tqdm(range(numPeriods)):
        corr = df_corr.loc[index[i]].iloc[:, 1:].values
        g_distance = np.nan_to_num(np.sqrt(0.5 * (1 - corr)), 0)
        G = nx.from_numpy_array(g_distance)
        T = nx.minimum_spanning_tree(G)

        possible_edges = list(nx.complement(T).edges())

        for edge in possible_edges:
            T.add_edge(*edge)
            if not nx.check_planarity(T)[0]:
                T.remove_edge(*edge)

        matrix[i] = (nx.to_numpy_array(T) > 0).astype(int) * g_distance

    return matrix


def rrg(df_adj_close, rs_ratio_window=50, rs_momentum_window=10):
    norm_prices = df_adj_close / df_adj_close.iloc[0]

    benchmark = norm_prices.mean(axis=1)

    relative_price_df = pd.DataFrame(
        index=norm_prices.index, columns=norm_prices.columns
    )
    for col in norm_prices.columns:
        relative_price_df[col] = norm_prices[col].values / benchmark.values

    cum = (
        relative_price_df.iloc[rs_momentum_window:].values
        / relative_price_df.iloc[:-rs_momentum_window].values
    )

    momentum_df = pd.DataFrame(
        cum,
        index=relative_price_df.index[rs_momentum_window:],
        columns=relative_price_df.columns,
    )

    momentum_df_mean = momentum_df.rolling(50).mean().iloc[rs_ratio_window - 1 :]
    momentum_df_std = momentum_df.rolling(50).std().iloc[rs_ratio_window - 1 :]
    momentum_df = momentum_df.iloc[rs_ratio_window - 1 :]

    rs_momentum = 100 + (momentum_df - momentum_df_mean) / momentum_df_std

    rolling_means = (
        relative_price_df.rolling(rs_ratio_window).mean().iloc[rs_ratio_window - 1 :]
    )
    rolling_std = (
        relative_price_df.rolling(rs_ratio_window).std().iloc[rs_ratio_window - 1 :]
    )
    relative_price_df = relative_price_df.iloc[rs_ratio_window - 1 :]

    rs_ratio = 100 + (relative_price_df - rolling_means) / rolling_std
    rs_ratio = rs_ratio.iloc[rs_momentum_window:]

    return rs_ratio, rs_momentum


def rrg_quadrants(df_adj_close):
    rs_ratio, rs_momentum = rrg(df_adj_close)
    df_rgg = pd.DataFrame(index=rs_ratio.index, columns=rs_ratio.columns)

    for stock in tqdm(rs_ratio.columns):
        for date in rs_ratio.index:
            if rs_ratio.loc[date, stock] > 100 and rs_momentum.loc[date, stock] > 100:
                df_rgg.loc[date, stock] = 1
            elif rs_ratio.loc[date, stock] > 100 and rs_momentum.loc[date, stock] < 100:
                df_rgg.loc[date, stock] = 2
            elif rs_ratio.loc[date, stock] < 100 and rs_momentum.loc[date, stock] < 100:
                df_rgg.loc[date, stock] = 3
            else:
                df_rgg.loc[date, stock] = 4
    return df_rgg


def features_pipeline(
    df_adj_close,
    df_close,
    df_high,
    df_low,
    df_volume,
    return_periods=[1, 3, 7, 14, 30, 40, 50, 60, 70, 80, 90],
    corr_period=90,
    forecast_period=5,
    stocks=[],
):
    num_stocks = len(stocks)
    df_features = get_features(df_close, df_high, df_low, stocks)

    cols = df_features.columns
    ind = df_features.index

    sc = StandardScaler()

    features = sc.fit_transform(df_features)

    # scaled features
    df_features = pd.DataFrame(features, columns=cols, index=ind)

    print("Features generated and scaled")

    # return features
    df_return = get_return(df_adj_close)

    df_past_returns = pd.DataFrame(index=df_return.index)

    for days in return_periods:
        df_temp = get_return(df_adj_close, period=days)

        df_past_returns = pd.merge(
            df_past_returns,
            df_temp,
            left_index=True,
            right_index=True,
            suffixes=("", f"_{days}"),
        )

    print("Return features generated")

    df_features = pd.merge(
        df_features, df_past_returns, left_index=True, right_index=True
    )

    num_features = df_features.shape[1] / num_stocks

    df_corr, df_cov = get_corr_n_cov(df_return, period=corr_period)

    df_vol_corr = df_volume.pct_change().rolling(90).corr().reset_index()
    df_vol_corr.index = df_vol_corr.Date
    df_vol_corr.drop("Date", axis=1, inplace=True)
    df_vol_corr = df_vol_corr.dropna()

    print("Correlation and covariance matrices generated")

    future_return = get_return(df_adj_close, forecast_period, future=True)

    # common index
    index = common_index(df_features, df_corr, future_return, df_vol_corr)

    df_past_returns = df_past_returns.loc[index]
    df_features = df_features.loc[index]
    df_corr = df_corr.loc[index]

    # get future covariance ??
    df_cov = df_cov.loc[index]

    future_return = future_return.loc[index]
    df_vol_corr = df_vol_corr.loc[index]
    df_return = df_return.loc[index]

    print("Common index length", len(index))

    df_rrg = rrg_quadrants(df_adj_close)
    df_rrg = df_rrg.loc[index]
    rrg_adj_matrix = get_adj_matrix(df_rrg, num_stocks, df_rrg.columns)

    df_corr_mst = minimum_spanning_tree_matrix(df_corr)
    df_vol_corr_mst = minimum_spanning_tree_matrix(df_vol_corr)

    df_corr_mfpt = maxim_planar_tree(df_corr)
    df_vol_corr_mfpt = maxim_planar_tree(df_vol_corr)

    cov_adj_matrix = get_cov_adj_matrix(df_cov, num_stocks)
    # print(cov_adj_matrix.shape)

    nodes_matrix = get_nodes_matrix(df_features, stocks, num_features)

    print("Adjacency matrices generated")

    combined_adj_matrix = concat_adj_matrices(
        rrg_adj_matrix, df_corr_mst, df_vol_corr_mst, df_corr_mfpt, df_vol_corr_mfpt
    )
    # combined_adj_matrix = concat_adj_matrices(df_corr_mst, df_vol_corr_mst, df_corr_mfpt, df_vol_corr_mfpt)

    print("Pipeline finished")

    return (
        nodes_matrix,
        combined_adj_matrix,
        cov_adj_matrix,
        future_return,
        df_return,
        df_features,
    )
