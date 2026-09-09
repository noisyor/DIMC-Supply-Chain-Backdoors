# generate_cifar10_stats_both.py
import numpy as np
import torch
from torchvision import datasets, transforms
from cleanfid.features import build_feature_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.makedirs("stats", exist_ok=True)

# Load CIFAR-10 with proper preprocessing
transform = transforms.Compose([
    transforms.Resize(299),  # Resize to 299x299 for Inception
    transforms.ToTensor(),
])

# Build feature extractor (only once)
feat_extractor = build_feature_extractor("clean", device=torch.device("cuda"))

def compute_statistics(train=False):
    """Compute statistics for train or test set"""
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)
    
    # Extract features
    features_list = []
    split_name = "train" if train else "test"
    
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc=f"Extracting {split_name} features"):
            batch = batch.cuda()
            features = feat_extractor(batch)
            features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma

# Compute statistics for both splits
print("Computing test set statistics...")
test_mu, test_sigma = compute_statistics(train=False)
np.savez('stats/cifar10.test.npz', mu=test_mu, sigma=test_sigma)
print(f"Saved test statistics: mu shape {test_mu.shape}, sigma shape {test_sigma.shape}")

print("\nComputing train set statistics...")
train_mu, train_sigma = compute_statistics(train=True)
np.savez('stats/cifar10.train.npz', mu=train_mu, sigma=train_sigma)
print(f"Saved train statistics: mu shape {train_mu.shape}, sigma shape {train_sigma.shape}")
