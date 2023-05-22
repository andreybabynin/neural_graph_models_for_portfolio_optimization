import neptune
import torch
from tqdm import tqdm
import numpy as np
from src.loss_functions import *


def train(
    model,
    optimizer,
    dataset,
    epochs=2,
    eval_func="max_return",
    exp_name="GrossModel_v1",
    neptune_token="",
    neptune_project="",
):
    model.train()
    model.mode = "Train"

    loss_list, gamma_list = [], []

    run = neptune.init_run(
        project=neptune_project, api_token=neptune_token, name=exp_name
    )
    params = {"learning_rate": 0.001, "optimizer": "Adam", "eval_func": eval_func}
    run["parameters"] = params

    for epoch in range(epochs):
        cum_losses, cum_gammas = 0, 0

        for i in range(len(dataset) - model.sample_size - model.pred_window):
            optimizer.zero_grad()

            w, idx = model(dataset, i)
            w = w.reshape(1, dataset.num_nodes)  # (1, #assets)

            future_ret = dataset[idx].future_return
            fut_cov = dataset[idx].cov_matrix

            loss = eval(eval_func)(w, future_ret.T, fut_cov)

            # add batch to PER
            model.storage.add(loss.detach().item(), idx)

            gamma_list.append(model.gamma.item())
            loss_list.append(loss.detach().item())

            cum_gammas += model.gamma.item()
            cum_losses += loss.detach().item()

            # Ensure that gamma > 0 after taking a descent step
            for name, param in model.named_parameters():
                if name == "gamma":
                    param.data.clamp_(0.0001)

            # track everything in neptune
            if i % 10 == 0 and i > 0:
                run["train/loss"].append(cum_losses / 10)
                run["train/gamma"].append(cum_gammas / 10)
                cum_losses, cum_gammas = 0, 0

            loss.backward()
            optimizer.step()

    run.stop()

    return loss_list, gamma_list


def evaluate(model, test_dataset):
    test_weights_matrix = np.zeros(
        (
            len(test_dataset) - model.sample_size - model.pred_window,
            1,
            test_dataset.num_nodes,
        )
    )
    model.mode = "Eval"
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset) - model.sample_size - model.pred_window)):
            w, _ = model(test_dataset, i)
            w = w.reshape(1, test_dataset.num_nodes)  # (1, #assets)

            test_weights_matrix[i] = w.detach().numpy()

    test_weights_matrix = test_weights_matrix.squeeze(1)

    return test_weights_matrix
