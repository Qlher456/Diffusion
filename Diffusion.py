import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import LFWPeople
from torchvision.utils import save_image
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
image_size = 128
channels = 3
batch_size = 128
timesteps = 400  # 增加扩散步骤
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 500
save_and_sample_every = 200
learning_rate = 1e-5

# Diffusion constants
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.01  # 减小终止值
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1),  # Normalize to [-1, 1]
])

dataset = LFWPeople(root="./data", split="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
# U-Net model implementation
class Unet(nn.Module):
    def __init__(self, channels, dim, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults

        # Initial convolution layer
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)  # channels = 3 for RGB input

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling blocks
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),  # Downsampling
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.ReLU()
            ))

        # Upsampling blocks
        for in_dim, out_dim in zip(reversed(dims[1:]), reversed(dims[:-1])):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),  # Upsampling
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.ReLU()
            ))

        # Final convolution layer
        self.final_conv = nn.Conv2d(dims[0], channels, 1)  # Output channels = input channels

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)

        # Downsample
        downs = []
        for down in self.downs:
            x = down(x)
            downs.append(x)

        # Upsample
        for up in self.ups:
            x = up(x + downs.pop())

        # Final convolution
        return self.final_conv(x)

# Loss function
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    # Get noisy image
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # Predict noise using the model
    predicted_noise = denoise_model(x_noisy)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")

    return loss

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Sampling function
@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)  # Start with random noise

    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling loop", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)  # Perform a single reverse diffusion step

    return img

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Instantiate model and optimizer
model = Unet(channels=channels, dim=image_size, dim_mults=(1, 2, 4, 8))
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

losses = []  # 用于记录损失

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch_size = batch[0].shape[0]
        batch = batch[0].to(device)
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, batch, t, loss_type="l2")
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        if step % save_and_sample_every == 0:
            samples = sample(model, image_size, batch_size=4, channels=channels)
            samples = (samples + 1) * 0.5  # Normalize back to [0, 1]
            save_image(samples, results_folder / f"sample-{epoch}-{step}.png", nrow=4)

# Plot loss curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig(results_folder / "loss_curve.png")
plt.show()

# Save final model
torch.save(model.state_dict(), results_folder / "final_model.pth")

