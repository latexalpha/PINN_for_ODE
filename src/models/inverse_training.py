import torch
import torch.nn as nn


class Training:
    def __init__(self, cfg, model):
        self.model = model

        self.system_type = cfg.system.type
        self.input_dim = cfg.network.input_dim

        self.batch_size = cfg.training.batch_size
        self.epochs = cfg.training.epochs
        self.initial_reg = cfg.training.initial_reg
        self.data_reg = cfg.training.data_reg
        self.physics_reg = cfg.training.physics_reg

        learning_rate = cfg.training.learning_rate
        self.scheduler_step_size = cfg.training.scheduler_step_size
        self.scheduler_gamma = cfg.training.scheduler_gamma
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        self.measure = nn.MSELoss()

        self.loss_minimum = 1.0e10
        self.loss_total_log = []
        self.loss_t0_log = []
        self.loss_data_log = []
        self.loss_physics_log = []
        self.m_log = []
        self.c_log = []
        self.k_log = []

    def train_step(self, t0, u0, y0, input_batch, output_batch):
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
        residual_t0 = (
            self.model.m * dxdt2_t0
            + self.model.c * dxdt_t0
            + self.model.k * x_pred_t0
            - u0
        )
        initial_loss = (
            self.measure(x_pred_t0, x0)
            + self.measure(dxdt_t0, v0)
            # + self.measure(residual_t0, 0.0 * residual_t0)
        )

        x_pred, dxdt, dxdt2 = self.model.forward_physics(input_batch)
        residual = (
            self.model.m * dxdt2
            + self.model.c * dxdt
            + self.model.k * x_pred
            - excitation
        )
        data_loss = self.measure(x_pred, output_batch)
        physics_loss = self.measure(residual, 0.0 * residual)

        total_loss = (
            self.initial_reg * initial_loss
            + self.data_reg * data_loss
            + self.physics_reg * physics_loss
        )
        total_loss.backward()
        self.opt.step()

        return total_loss, initial_loss, data_loss, physics_loss

    def training(
        self,
        t0,
        u0,
        y0,
        train_dataset,
        logger,
        best_model_path,
    ):
        # There are no data loss in the training process for the forward solution
        for epoch in range(self.epochs):
            loss = 0.0
            loss_t0 = 0.0
            loss_data = 0.0
            loss_physics = 0.0
            for batch_dataset in train_dataset:
                batch_input = batch_dataset[0][:, 0:2]
                batch_output = batch_dataset[0][:, 2:3]
                batch_loss, batch_loss_t0, batch_loss_data, batch_loss_physics = (
                    self.train_step(t0, u0, y0, batch_input, batch_output)
                )
                loss += batch_loss
                loss_t0 += batch_loss_t0
                loss_data += batch_loss_data
                loss_physics += batch_loss_physics

            if loss < self.loss_minimum:
                self.loss_minimum = loss
                torch.save(self.model.state_dict(), best_model_path)

            self.loss_total_log.append(loss.cpu().detach().numpy())
            self.loss_t0_log.append(loss_t0.cpu().detach().numpy())
            self.loss_data_log.append(loss_data.cpu().detach().numpy())
            self.loss_physics_log.append(loss_physics.cpu().detach().numpy())

            self.m_log.append(self.model.m.cpu().detach().numpy())
            self.c_log.append(self.model.c.cpu().detach().numpy())
            self.k_log.append(self.model.k.cpu().detach().numpy())
            self.scheduler.step()
            if epoch % 50 == 0:
                logger.info(
                    f"Epoch = {epoch}/{self.epochs}, Loss_total = {float(loss):.10f}, "
                    f" Loss_t0 = {float(loss_t0):.10f}, Loss_data = {float(loss_data):.10f}, "
                    f" Loss_physics = {float(loss_physics):.10f}"
                )
        return {
            "loss_total": self.loss_total_log,
            "loss_t0": self.loss_t0_log,
            "loss_data": self.loss_data_log,
            "loss_physics": self.loss_physics_log,
            "m": self.m_log,
            "c": self.c_log,
            "k": self.k_log,
        }
