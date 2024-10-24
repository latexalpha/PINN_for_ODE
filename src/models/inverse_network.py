import torch
import torch.nn as nn
import torch.autograd as autograd


class MLP(nn.Module):
    def __init__(
        self,
        device,
        cfg,
    ) -> None:
        """
        Builds an neural network to approximate the value of the differential equation
        input_size must be set to number of parameters in the function f for ODE this is 1
        output_size number of values output by the function for ODE this is 1
        """
        super().__init__()
        self.device = device
        self.input_dim = cfg.network.input_dim
        self.output_dim = cfg.network.output_dim
        self.num_hidden_layers = cfg.network.num_hidden_layers
        self.hidden_dim = cfg.network.hidden_dim

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)] * self.num_hidden_layers
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = nn.GELU()
        self.m = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device))
        self.c = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device))
        self.k = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device))

    def _apply_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the input through the hidden layers
        """
        # clip the parameters m c k to be positive
        self.m.data = torch.clamp(self.m.data, 0.5)
        self.c.data = torch.clamp(self.c.data, 0.5)
        self.k.data = torch.clamp(self.k.data, 0.5)

        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        y = self.output_layer(x)
        return y

    def forward_physics(self, input):

        input.requires_grad = True
        x_pred = self._apply_model(input)

        dxdt = autograd.grad(
            x_pred,
            input,
            torch.ones(x_pred.shape[0], 1).to(self.device),
            create_graph=True,
        )[0]

        dxdt2 = autograd.grad(
            dxdt,
            input,
            torch.ones_like(input).to(self.device),
            create_graph=True,
        )[0]
        dxdt = dxdt[:, 0:1]
        dxdt2 = dxdt2[:, 0:1]
        return x_pred, dxdt, dxdt2
