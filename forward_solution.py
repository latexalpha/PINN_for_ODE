# import libraries
import torch
import numpy as np
from torch import nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.forward_network import MLP
from src.models.forward_training import Training
from src.data.solve_ivp import solve_nonautonomous_RK4
from src.data.irf import IRF
from src.visualization.plotting import (
    plot_displacement_prediction,
    plot_velocity_prediction,
    plot_IRF,
    plot_comparison,
    plot_losses,
)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def system_training(cfg, output_dir, logger):

    system_type = cfg.system.type
    input_dim = cfg.network.input_dim
    training_type = cfg.training.type
    experiment = f"{training_type}_{system_type}_{input_dim}inputs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(f"{output_dir}/models")
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    best_model_path = f"{model_dir}/best_model.pth"
    figure_dir = Path(f"{output_dir}/figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)

    ### solve the ODE
    m = cfg.system.m
    c = cfg.system.c
    k = cfg.system.k
    duration = cfg.system.duration
    fs = cfg.system.fs
    fe = cfg.system.fe
    amp = cfg.system.amp

    tspan = np.linspace(0, duration, duration * fs)
    omega_n = np.sqrt(k / m)
    damping_ratio = c / 2 / np.sqrt(m * k)
    setting_time = -np.log(0.04) / (damping_ratio * omega_n)
    print(
        f"omega_n = {float(omega_n):.3f}, "
        f"damping_ratio = {float(damping_ratio):.3f}, "
        f"setting_time = {float(setting_time):.3f}"
    )

    if system_type == "auto":
        u = np.zeros_like(tspan)
        y0 = np.array([0.1, 0.0])
    elif system_type == "nonauto":
        u = amp * np.cos(fe * tspan)
        y0 = np.array([0.0, 0.0])
    else:
        raise ValueError("Invalid system type")

    y = solve_nonautonomous_RK4(y0, tspan, u, m, c, k)
    irf = IRF(m, c, k, tspan)
    plot_IRF(tspan, irf.cpu().numpy(), setting_time, experiment, figure_dir)
    plot_comparison(tspan, u, y, k, c, setting_time, experiment, figure_dir)

    # convert to torch tensor and construct the dataset
    tspan = np.reshape(tspan, (-1, 1))
    u = np.reshape(u, (-1, 1))
    displacement = y[:, 0:1]
    velocity = y[:, 1:2]
    tspan_data = torch.tensor(tspan, dtype=torch.float32, device=device)
    u_data = torch.tensor(u, dtype=torch.float32, device=device)
    y0 = torch.tensor(y0, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(torch.cat((tspan_data, u_data), -1))
    train_dataset = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    model = MLP(
        device=device,
        cfg=cfg,
    ).to("cuda")
    mlp_training = Training(
        cfg=cfg,
        model=model,
    )
    Loss_total, Loss_t0, Loss_physics = mlp_training.training(
        t0=tspan_data[0],
        u0=u_data[0],
        y0=y0,
        train_dataset=train_dataset,
        logger=logger,
        best_model_path=best_model_path,
    )
    plot_losses(Loss_t0, Loss_physics, experiment, figure_dir)

    # model.load_state_dict(torch.load(best_model_path))
    model.eval()
    if input_dim == 1:
        input_data = tspan_data[:, 0:1]
    y_pred, dy_pred, ddy_pred = model.forward_physics(input_data)
    y_pred = y_pred.detach().cpu().numpy()
    dy_pred = dy_pred.detach().cpu().numpy()

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
