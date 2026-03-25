import torch
from torch import optim

from .config import Config
from .models import KernelRowNet
from .loss import weighted_loss

def train_one_row(config: Config):
    model = KernelRowNet(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        output_dim=config.D,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        # placeholder batch
        t = torch.rand(config.batch_size, 1)
        x = torch.rand(config.batch_size, 1)

        pred = model(t, x)

        # placeholder residual
        residuals = pred

        loss = weighted_loss(residuals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch={epoch} loss={loss.item():.6f}")

    return model

if __name__ == "__main__":
    cfg = Config()
    train_one_row(cfg)