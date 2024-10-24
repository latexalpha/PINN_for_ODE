import torch
import torch.nn as nn


class Training:
    def __init__(self, cfg, model):

        self.model = model

        self.m = cfg.system.m
        self.c = cfg.system.c
        self.k = cfg.system.k

        self.system_type = cfg.system.type
        self.input_dim = cfg.network.input_dim
        self.physics_reg = cfg.training.physics_reg
        self.epochs = cfg.training.epochs

        self.initial_reg = cfg.training.initial_reg
        self.physics_reg = cfg.training.physics_reg

        learning_rate = cfg.training.learning_rate
        step_size = cfg.training.scheduler_step_size
        gamma = cfg.training.scheduler_gamma
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=step_size, gamma=gamma
        )
        self.measure = nn.MSELoss()

        self.Loss_total = []
        self.Loss_t0 = []
        self.Loss_physics = []
        self.loss_minimum = 1.0e10

    def train_step(self, t0, u0, y0, input_batch):
        self.opt.zero_grad()
        t0 = t0
        u0 = u0
        x0 = y0[0:1]
        v0 = y0[1:2]
        t0 = t0.unsqueeze(1)  # unsqueeze to make it 2D
        x0 = x0.unsqueeze(1)
        v0 = v0.unsqueeze(1)
        u0 = u0.unsqueeze(1)
        excitation = input_batch[:, 1:2]
        if self.input_dim == 1:
            input_t0 = t0
            input_batch = input_batch[:, 0:1]
        elif self.input_dim == 2:
            input_t0 = torch.cat((t0, u0), -1)
            input_batch = input_batch

        x_pred_t0, dxdt_t0, dxdt2_t0 = self.model.forward_physics(input_t0)
        initial_loss = self.measure(x_pred_t0, x0) + self.measure(dxdt_t0, v0)

        x_pred, dxdt, dxdt2 = self.model.forward_physics(input_batch)
        residual = self.m * dxdt2 + self.c * dxdt + self.k * x_pred - excitation
        physics_loss = self.measure(residual, 0.0 * residual)

        total_loss = self.initial_reg * initial_loss + self.physics_reg * physics_loss
        total_loss.backward()
        self.opt.step()

        return total_loss, initial_loss, physics_loss

    def training(self, t0, u0, y0, train_dataset, logger, best_model_path):
        # There are no data loss in the training process for the forward solution
        for epoch in range(self.epochs):
            for batch_input in train_dataset:
                batch_input = batch_input[0]
                batch_loss, batch_loss_t0, batch_loss_physics = self.train_step(
                    t0, u0, y0, batch_input
                )
                if batch_loss < self.loss_minimum:
                    self.loss_minimum = batch_loss
                    torch.save(self.model.state_dict(), best_model_path)

            self.Loss_total.append(batch_loss.cpu().detach().numpy())
            self.Loss_t0.append(batch_loss_t0.cpu().detach().numpy())
            self.Loss_physics.append(batch_loss_physics.cpu().detach().numpy())

            self.scheduler.step()
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch = {epoch}/{self.epochs}, Loss_total = {float(batch_loss):.10f}, "
                    f" Loss_t0 = {float(batch_loss_t0):.10f}, Loss_physics = {float(batch_loss_physics):.10f}"
                )
        return self.Loss_total, self.Loss_t0, self.Loss_physics
