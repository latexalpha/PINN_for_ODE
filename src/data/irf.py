import torch


def IRF(m, c, k, tspan):
    # the impulse response function
    m = torch.tensor(m, dtype=torch.float32, device="cuda")
    c = torch.tensor(c, dtype=torch.float32, device="cuda")
    k = torch.tensor(k, dtype=torch.float32, device="cuda")
    tspan = torch.tensor(tspan, dtype=torch.float32, device="cuda")
    A = 2 / torch.sqrt(4 * m * k - c * c)
    B = torch.exp(-c / m / 2 * tspan)
    C = torch.sin(torch.sqrt(4 * m * k - c * c) / 2 / m * tspan)
    IRF = A * B * C
    return IRF
