import torch
import numpy as np
from torch import nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.inverse_network import MLP
from src.models.inverse_training import Training
from src.data.solve_ivp import solve_nonautonomous_RK4
from src.data.irf import IRF
from src.visualization.plotting import (
    plot_displacement_prediction,
    plot_velocity_prediction,
    plot_IRF,
    plot_comparison,
    plot_losses,
    plot_mck,
)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def system_training(cfg, output_dir, logger):

    system_type = cfg.system.type
    input_dim = cfg.network.input_dim
    training_type = cfg.training.type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = f"{training_type}_{system_type}_{input_dim}inputs"
    model_dir = Path(f"{output_dir}/models")
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    best_model_path = f"{model_dir}/best_model.pth"
    figure_dir = Path(f"{output_dir}/figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)

    # solve the ODE to get the training data
    m = cfg.system.m
    c = cfg.system.c
    k = cfg.system.k
    omega_n = np.sqrt(k / m)
    damping_ratio = c / 2 / np.sqrt(m * k)
    setting_time = -np.log(0.04) / (damping_ratio * omega_n)
    print(
        f"omega_n = {float(omega_n):.3f}, damping_ratio = {float(damping_ratio):.3f}, setting_time = {float(setting_time):.3f}"
    )

    duration = cfg.system.duration
    fs = cfg.system.fs
    fe = cfg.system.fe
    amp = cfg.system.amp

    tspan = np.linspace(0, duration, duration * fs)
    if system_type == "auto":
        u = np.zeros_like(tspan)
        y0 = np.array([0.5, 0.0])
    elif system_type == "nonauto":
        u = amp * np.cos(fe * tspan)
        y0 = np.array([0.0, 0.0])
    else:
        raise ValueError("Invalid system type")

    y = solve_nonautonomous_RK4(y0, tspan, u, m, c, k)
    irf = IRF(m, c, k, tspan)
    plot_IRF(tspan, irf.cpu().numpy(), setting_time, experiment, figure_dir)
    plot_comparison(tspan, u, y, k, c, setting_time, experiment, figure_dir)

    # tspan = tspan[fs * 10 :]
    # u = u[fs * 10 :]
    # y = y[fs * 10 :, :]

    tspan = np.reshape(tspan, (-1, 1))
    u = np.reshape(u, (-1, 1))
    tspan_data = torch.tensor(tspan, dtype=torch.float32, device=device)
    u_data = torch.tensor(u, dtype=torch.float32, device=device)
    displacement_data = torch.tensor(y[:, 0:1], dtype=torch.float32, device=device)
    velocity_data = torch.tensor(y[:, 1:2], dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(
        torch.cat((tspan_data, u_data, displacement_data), -1)
    )
    train_dataset = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    y0 = torch.tensor(
        [displacement_data[0], velocity_data[0]],
        dtype=torch.float32,
        device=device,
    )

    model = MLP(
        device=device,
        cfg=cfg,
    ).to("cuda")
    mlp_training = Training(
        cfg=cfg,
        model=model,
    )
    training_logs = mlp_training.training(
        t0=u_data[0],
        u0=u_data[0],
        y0=y0,
        train_dataset=train_dataset,
        logger=logger,
        best_model_path=best_model_path,
    )
    loss_total_log = training_logs["loss_total"]
    loss_t0_log = training_logs["loss_t0"]
    loss_data_log = training_logs["loss_data"]
    loss_physics_log = training_logs["loss_physics"]
    m_log = training_logs["m"]
    c_log = training_logs["c"]
    k_log = training_logs["k"]
    plot_losses(loss_data_log, loss_physics_log, experiment, figure_dir)
    plot_mck(m_log, c_log, k_log, experiment, figure_dir)

    model.eval()
    if input_dim == 1:
        input_data = tspan_data
    y_pred, dy_pred, ddy_pred = model.forward_physics(input_data)
    # the predicted output is scaled back to the original scale
    y_pred = y_pred.detach().cpu().numpy()
    dy_pred = dy_pred.detach().cpu().numpy()
    displacement = displacement_data.detach().cpu().numpy()
    velocity = velocity_data.detach().cpu().numpy()

    plot_displacement_prediction(
        tspan,
        y_pred,
        displacement,
        setting_time,
        experiment,
        figure_dir,
    )
    plot_velocity_prediction(
        tspan, dy_pred, velocity, setting_time, experiment, figure_dir
    )
