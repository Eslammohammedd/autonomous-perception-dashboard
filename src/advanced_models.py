"""
advanced_models.py — Phase 3 Advanced Deep Learning Models for AV Perception.

Provides:
  - Generator / Discriminator (DCGAN for synthetic image generation)
  - LSTMAttentionClassifier (Sequence modeling with Attention)
  - train_gan_epoch() / train_gan() (DCGAN training loop)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

# ─────────────────────────────────────────────────────────────
# 1. DCGAN — Deep Convolutional Generative Adversarial Network
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    DCGAN Generator for 32x32 RGB images (CIFAR-10 size).
    Input: (Batch, latent_dim, 1, 1)
    Output: (Batch, 3, 32, 32)
    """
    def __init__(self, latent_dim=100, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # Size: (features*4) x 4 x 4
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # Size: (features*2) x 8 x 8
            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # Size: features x 16 x 16
            nn.ConvTranspose2d(features, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Size: 3 x 32 x 32
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for 32x32 RGB images.
    Input: (Batch, 3, 32, 32)
    Output: (Batch, 1)
    """
    def __init__(self, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: features x 16 x 16
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (features*2) x 8 x 8
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (features*4) x 4 x 4
            nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Size: 1 x 1 x 1
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(netG, netD, dataloader, epochs, device, latent_dim=100):
    """
    DCGAN training loop.
    Returns tracking history of losses and sample generated images.
    """
    netG.to(device)
    netD.to(device)
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    # Standard GAN optimizers
    optD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    G_losses = []
    D_losses = []
    
    print(f"[DCGAN] Starting Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        for i, data in enumerate(dataloader):
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            
            # --- Train Discriminator ---
            netD.zero_grad()
            label = torch.full((b_size, 1), 1.0, dtype=torch.float, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optD.step()
            
            # --- Train Generator ---
            netG.zero_grad()
            label.fill_(1.0) # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optG.step()
            
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        print(f"[{epoch}/{epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
        
    # Generate final samples
    with torch.no_grad():
        fake_samples = netG(fixed_noise).detach().cpu()
        
    return {"G_losses": G_losses, "D_losses": D_losses}, fake_samples


def plot_gan_losses(history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(history['G_losses'], label="G", color="#4C72B0")
    plt.plot(history['D_losses'], label="D", color="#DD8452")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────
# 2. LSTM + Attention for Sequential Data
# ─────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Simple Self-Attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        energy = self.projection(encoder_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, seq_len)
        # Context vector: (batch_size, hidden_dim)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LSTMAttentionClassifier(nn.Module):
    """
    Treats an image as a sequence of rows.
    For CIFAR-10: 32 rows, where each row has 32*3=96 features.
    Demonstrates Sequence Modeling + Attention for Perception tasks.
    """
    def __init__(self, input_size=96, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        # Bidirectional means output dimension is hidden_size * 2
        self.attention = SelfAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x is (batch, C, H, W)
        batch_size, C, H, W = x.size()
        # Reshape to (batch, H, W*C) so that sequence length is H
        x = x.view(batch_size, C, H, W).permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, H, W * C)
        
        out, (hn, cn) = self.lstm(x)
        
        # Apply attention over the sequence of rows
        context, attn_weights = self.attention(out)
        
        # Final classification
        logits = self.fc(context)
        return logits, attn_weights

