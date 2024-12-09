import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 超参数
nz = 100  # 潜在向量大小
num_epochs = 1000
lr = 0.0001
beta_start = 1e-4
beta_end = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建Image文件夹
os.makedirs('Image', exist_ok=True)

# 生成器 (U-Net 风格)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # 解码器部分
        self.dec1 = self._deconv_block(512, 256)
        self.dec2 = self._deconv_block(256, 128)
        self.dec3 = self._deconv_block(128, 64)
        self.dec4 = self._deconv_block(64, 3)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 解码
        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)

        return d4

# 噪声调度器
def get_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)

# 扩散过程：正向过程
def forward_diffusion(x_0, beta_schedule):
    noise = torch.randn_like(x_0)
    x_t = x_0
    for t in range(len(beta_schedule)):
        x_t = torch.sqrt(1 - beta_schedule[t]) * x_t + torch.sqrt(beta_schedule[t]) * noise
    return x_t

# 反向过程：去噪
def reverse_diffusion(model, x_T, beta_schedule, num_steps=1000):
    x_t = x_T
    for t in reversed(range(num_steps)):
        noise_pred = model(x_t)
        x_t = (x_t - beta_schedule[t] * noise_pred) / torch.sqrt(1 - beta_schedule[t])
    return x_t

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 LFW 数据集
dataset = datasets.LFWPeople(root='data/', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# 初始化模型
model = UNet().to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 损失记录
losses = []

# Beta schedule
beta_schedule = get_beta_schedule(num_steps=1000, beta_start=beta_start, beta_end=beta_end)

# 训练扩散模型
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)

        # 正向扩散过程
        x_T = forward_diffusion(real_images, beta_schedule)

        # 反向去噪过程
        optimizer.zero_grad()
        noise_pred = model(x_T)
        loss = nn.MSELoss()(noise_pred, real_images)
        loss.backward()
        optimizer.step()

        # 记录损失
        losses.append(loss.item())

        if i % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}')

    # 保存图像示例
    with torch.no_grad():
        sample_image = reverse_diffusion(model, x_T, beta_schedule)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch + 1}")
        plt.imshow(sample_image[0].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        plt.savefig(f'Image/Epoch-{epoch + 1}.png')
        plt.close()

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Image/loss_plot.png')

# 保存模型
torch.save(model.state_dict(), 'diffusion_model.pth')
