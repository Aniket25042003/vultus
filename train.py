import os
import math
import time
import random
import argparse
import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from tqdm import tqdm

# Metrics computation
from sklearn.metrics import roc_curve, auc, roc_auc_score


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed: int = 42, verbose: bool = True):
    if verbose:
        print(f"[SEED] Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"[SEED] Random seed initialized successfully")


# ---------------------------
# Custom Dataset for VGGFace2 structure
# ---------------------------
class VGGFace2Dataset(Dataset):
    """
    Custom dataset for VGGFace2 structure:
    - Root folder: vggface2_112x112/
    - Subfolders: id_0, id_1, ..., id_8630 (9131 total)
    - Images: 0.jpg, 1.jpg, etc. in each subfolder
    """
    def __init__(
        self,
        root: str,
        identity_indices: List[int],
        transform=None,
    ):
        self.root = Path(root)
        self.identity_indices = identity_indices
        self.transform = transform
        
        # Build image paths and labels
        self.samples = []
        self.targets = []
        
        # Only show progress on main process to avoid clutter in DDP
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        
        if show_progress:
            print(f"[DATASET] Loading dataset from {root}...")
        
        root_path = Path(root)
        for idx, identity_idx in enumerate(tqdm(identity_indices, desc="Scanning identities", leave=False, disable=not show_progress)):
            identity_folder = root_path / f"id_{identity_idx}"
            if not identity_folder.exists() or not identity_folder.is_dir():
                continue
            
            # Use os.listdir instead of Path.iterdir() for better performance
            try:
                folder_path_str = str(identity_folder)
                file_names = os.listdir(folder_path_str)
                # Filter and sort in one pass
                image_files = sorted([f for f in file_names if f.lower().endswith('.jpg')], key=str.lower)
                
                for img_name in image_files:
                    img_path = os.path.join(folder_path_str, img_name)
                    self.samples.append(img_path)
                    self.targets.append(idx)  # Use idx as label (0 to len(identity_indices)-1)
            except (OSError, PermissionError):
                continue
        
        if show_progress:
            print(f"[DATASET] Loaded {len(self.samples)} images from {len(identity_indices)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        
        # Load image (optimized - convert in one step like ImageFolder)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # If image is corrupted, return a black image
            image = Image.new('RGB', (112, 112), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ---------------------------
# Model: IR-SE-200 (InsightFace IResNet with SE attention)
# ---------------------------
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class IRSEBlock(nn.Module):
    """InsightFace-style IR block with SE."""
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        self.use_shortcut = (in_ch == out_ch and stride == 1)

        self.bn0 = nn.BatchNorm2d(in_ch, eps=bn_eps, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)

        self.se = SEModule(out_ch) if use_se else nn.Identity()

        if self.use_shortcut:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + identity
        return out


class IResNet(nn.Module):
    def __init__(
        self,
        layers,
        embedding_dim=512,
        dropout=0.4,
        use_se=True,
        bn_momentum=0.9,
        bn_eps=1e-5,
    ):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(64)

        self.stage1 = self._make_layer(64,  layers[0], stride=2, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage2 = self._make_layer(128, layers[1], stride=2, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage3 = self._make_layer(256, layers[2], stride=2, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage4 = self._make_layer(512, layers[3], stride=2, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps)

        self.bn2 = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_momentum)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim, bias=True)
        self.emb_bn = nn.BatchNorm1d(embedding_dim, eps=bn_eps, momentum=bn_momentum)

        self._init_weights()

    def _make_layer(self, out_ch, blocks, stride, use_se, bn_momentum, bn_eps):
        layers = []
        layers.append(IRSEBlock(self.in_ch, out_ch, stride=stride, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps))
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(IRSEBlock(self.in_ch, out_ch, stride=1, use_se=use_se, bn_momentum=bn_momentum, bn_eps=bn_eps))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.emb_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def iresnet100(embedding_dim=512, dropout=0.4, use_se=True, bn_momentum=0.9, bn_eps=1e-5):
    """InsightFace iresnet100 definition: layers = [2, 9, 20, 2] (smaller than iresnet200)"""
    return IResNet([2, 9, 20, 2], embedding_dim=embedding_dim, dropout=dropout, use_se=use_se,
                   bn_momentum=bn_momentum, bn_eps=bn_eps)


def iresnet200(embedding_dim=512, dropout=0.4, use_se=True, bn_momentum=0.9, bn_eps=1e-5):
    """Common InsightFace iresnet200 definition: layers = [6, 26, 60, 6]"""
    return IResNet([6, 26, 60, 6], embedding_dim=embedding_dim, dropout=dropout, use_se=use_se,
                   bn_momentum=bn_momentum, bn_eps=bn_eps)


# ---------------------------
# Dynamic ArcFace Head
# ---------------------------
class DynamicArcMarginProduct(nn.Module):
    """Dynamic ArcFace: Additive Angular Margin with settable margin and scale."""
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = float(s)
        self.m = float(m)
        self._update_trig()

    def _update_trig(self):
        m = self.m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_margin(self, m: float):
        """Update margin and recalculate trigonometric values."""
        self.m = float(m)
        self._update_trig()

    def set_scale(self, s: float):
        """Update scale value."""
        self.s = float(s)

    @torch.no_grad()
    def cosine_logits(self, embeddings):
        """Compute cosine logits without margin (for meaningful training accuracy)."""
        cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))
        return cosine * self.s

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ---------------------------
# Data Transforms (Reduced ColorJitter, No Label Smoothing)
# ---------------------------
def build_transforms(img_size: int = 112, verbose: bool = True):
    """Build transforms with minimal ColorJitter for face recognition."""
    if verbose:
        print(f"[TRANSFORMS] Building transforms for image size: {img_size}x{img_size}")
    
    # Training: minimal augmentation (no ColorJitter or very small)
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        # Very small color jitter or none at all
        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # Low probability RandomErasing
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0.0),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    if verbose:
        print(f"[TRANSFORMS] Train: Resize, RandomFlip, Normalize, RandomErasing (p=0.05)")
        print(f"[TRANSFORMS] Val: Resize, Normalize")
    return train_tfms, val_tfms


# ---------------------------
# P-K Identity Sampling
# ---------------------------
class PKBatchSampler(Sampler[List[int]]):
    """P-K batch sampler: sample P identities with K images each per batch."""
    def __init__(self, targets, P, K, steps_per_epoch, seed=42):
        self.targets = list(targets)
        self.P = P
        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

        self.label_to_indices = {}
        for idx, y in enumerate(self.targets):
            self.label_to_indices.setdefault(y, []).append(idx)

        self.labels = sorted(list(self.label_to_indices.keys()))

    def set_epoch(self, epoch: int):
        """Set epoch for different random seed each epoch."""
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            chosen_labels = rng.sample(self.labels, k=self.P)
            batch = []
            for y in chosen_labels:
                idxs = self.label_to_indices[y]
                if len(idxs) >= self.K:
                    batch.extend(rng.sample(idxs, k=self.K))
                else:
                    batch.extend([rng.choice(idxs) for _ in range(self.K)])
            yield batch


# ---------------------------
# Verification Metrics (ROC AUC, TAR@FAR, EER)
# ---------------------------
@torch.no_grad()
def compute_verification_metrics(
    backbone: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    fixed_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute verification metrics: ROC AUC, TAR@FAR (1e-3, 1e-4), EER.
    This is the proper evaluation metric for face recognition.
    
    Args:
        backbone: Face recognition backbone model
        val_loader: Validation data loader
        device: Device for inference
        max_samples: Maximum samples to use (ignored if fixed_indices provided)
        fixed_indices: Pre-computed fixed indices for reproducible sampling.
                       Use get_or_create_fixed_eval_indices() to generate these once.
    """
    # Unwrap DDP model if needed
    if isinstance(backbone, nn.parallel.DistributedDataParallel):
        backbone = backbone.module
    backbone.eval()
    
    print(f"[VERIFICATION] Computing embeddings for verification evaluation...")
    all_embs = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Extracting embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        embs = backbone(images)
        all_embs.append(embs.cpu())
        all_labels.append(labels.cpu())

    embs = torch.cat(all_embs, dim=0).numpy()  # [N, D]
    labels = torch.cat(all_labels, dim=0).numpy()  # [N]
    
    # Use fixed indices if provided (reproducible), otherwise sample randomly
    if fixed_indices is not None:
        # Filter indices that are valid for current dataset size
        valid_indices = fixed_indices[fixed_indices < len(embs)]
        if len(valid_indices) < len(fixed_indices):
            print(f"[VERIFICATION] Warning: {len(fixed_indices) - len(valid_indices)} indices out of range, using {len(valid_indices)} valid indices")
        print(f"[VERIFICATION] Using {len(valid_indices)} FIXED evaluation indices (reproducible)")
        embs = embs[valid_indices]
        labels = labels[valid_indices]
    elif max_samples is not None and len(embs) > max_samples:
        print(f"[VERIFICATION] WARNING: Sampling {max_samples} from {len(embs)} embeddings randomly (NOT reproducible!)")
        print(f"[VERIFICATION] Consider using get_or_create_fixed_eval_indices() for epoch-to-epoch consistency")
        indices = np.random.choice(len(embs), max_samples, replace=False)
        embs = embs[indices]
        labels = labels[indices]
    
    print(f"[VERIFICATION] Collected {len(embs)} embeddings")
    
    # Compute pairwise similarities (cosine similarity since embeddings are L2-normalized)
    print(f"[VERIFICATION] Computing pairwise similarities...")
    n = len(embs)
    
    # For large datasets, compute in batches to avoid memory issues
    batch_size = 1000
    similarities = []
    for i in tqdm(range(0, n, batch_size), desc="Computing similarities", leave=False):
        end_i = min(i + batch_size, n)
        batch_embs = embs[i:end_i]
        sims = np.dot(batch_embs, embs.T)  # [batch_size, N]
        similarities.append(sims)
    
    similarities = np.vstack(similarities)  # [N, N]
    
    # Create ground truth: 1 if same identity, 0 if different (VECTORIZED - much faster)
    print(f"[VERIFICATION] Creating ground truth labels...")
    # Vectorized comparison: compare labels[i] == labels[j] for all pairs
    labels_expanded = labels[:, np.newaxis]  # [N, 1]
    y_true = (labels_expanded == labels).astype(np.int32)  # [N, N] - vectorized!
    
    # Extract upper triangle (excluding diagonal) to avoid duplicates
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    y_true_flat = y_true[mask]
    similarities_flat = similarities[mask]
    
    print(f"[VERIFICATION] Total pairs: {len(y_true_flat)}")
    print(f"[VERIFICATION] Positive pairs: {y_true_flat.sum()}, Negative pairs: {(y_true_flat == 0).sum()}")
    
    # Compute ROC AUC
    print(f"[VERIFICATION] Computing ROC AUC...")
    roc_auc = roc_auc_score(y_true_flat, similarities_flat)
    
    # Compute EER (Equal Error Rate)
    print(f"[VERIFICATION] Computing EER...")
    fpr, tpr, thresholds = roc_curve(y_true_flat, similarities_flat)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Compute TAR@FAR using ROC curve
    print(f"[VERIFICATION] Computing TAR@FAR...")
    idx_1e3 = np.argmin(np.abs(fpr - 1e-3))
    idx_1e4 = np.argmin(np.abs(fpr - 1e-4))
    tar_at_far_1e3 = tpr[idx_1e3]
    tar_at_far_1e4 = tpr[idx_1e4]
    
    return {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tar_at_far_1e3": tar_at_far_1e3,
        "tar_at_far_1e4": tar_at_far_1e4,
    }


# ---------------------------
# Fixed Evaluation Indices (Reproducible Sampling)
# ---------------------------
def get_or_create_fixed_eval_indices(
    total_samples: int,
    max_samples: int,
    save_dir: str,
    seed: int = 42,
    filename: str = "eval_indices.npy",
) -> np.ndarray:
    """
    Get or create fixed evaluation indices for reproducible epoch-to-epoch comparison.
    
    - First call: generate random indices with fixed seed, save to disk
    - Subsequent calls: load from disk
    
    Args:
        total_samples: Total number of samples in the validation set
        max_samples: Maximum number of samples to use for evaluation
        save_dir: Directory to save/load indices
        seed: Random seed for reproducibility
        filename: Name of the indices file
    
    Returns:
        Fixed numpy array of indices
    """
    os.makedirs(save_dir, exist_ok=True)
    indices_path = os.path.join(save_dir, filename)
    
    if os.path.exists(indices_path):
        # Load existing indices
        indices = np.load(indices_path)
        print(f"[EVAL_INDICES] Loaded {len(indices)} fixed evaluation indices from {indices_path}")
        
        # Validate indices are still valid for current dataset size
        if len(indices) > total_samples or (indices >= total_samples).any():
            print(f"[EVAL_INDICES] Warning: Indices out of range for current dataset ({total_samples} samples). Regenerating...")
        else:
            return indices
    
    # Generate new fixed indices
    print(f"[EVAL_INDICES] Generating {min(max_samples, total_samples)} fixed evaluation indices (seed={seed})...")
    rng = np.random.RandomState(seed)
    
    if total_samples <= max_samples:
        indices = np.arange(total_samples)
    else:
        indices = rng.choice(total_samples, max_samples, replace=False)
        indices.sort()  # Sort for consistent ordering
    
    # Save to disk
    np.save(indices_path, indices)
    print(f"[EVAL_INDICES] Saved {len(indices)} fixed evaluation indices to {indices_path}")
    
    return indices


# ---------------------------
# Helper functions
# ---------------------------
def linear_ramp(epoch, start, end, ramp_epochs):
    """Linear ramp function for margin scheduling."""
    if ramp_epochs <= 0:
        return end
    t = min(max(epoch, 0), ramp_epochs) / float(ramp_epochs)
    return start + t * (end - start)


# ---------------------------
# Train Config
# ---------------------------
@dataclass
class TrainConfig:
    data_root: str
    img_size: int = 112
    embedding_dim: int = 512
    batch_size: int = 512
    epochs: int = 30
    lr: float = 0.05  # Lower base LR
    weight_decay: float = 5e-4
    num_workers: int = 4  # Increased for faster data loading
    arc_s: float = 64.0  # Fixed scale
    arc_m: float = 0.50  # Final margin
    amp: bool = True
    seed: int = 42
    save_dir: str = "./checkpoints"
    resume: str = ""
    warmup_epochs: int = 3  # Longer warmup
    early_stopping_patience: int = 10
    margin_ramp_epochs: int = 12  # Slower margin ramp
    margin_start: float = 0.2  # Start margin
    val_max_samples: Optional[int] = 5000  # Max samples for verification metrics (None = use all, faster with subset)
    gradient_accumulation_steps: int = 4  # Accumulate gradients over N steps (simulates larger batch size)


def init_distributed():
    """Initialize distributed training if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Use nccl backend for multi-GPU training
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=1800)  # 30 min timeout
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Set optimal CUDA settings for DDP
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return True, local_rank
    return False, 0


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_device(local_rank: int = 0):
    """Get device for training."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str, payload: dict, verbose: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if verbose:
        print(f"[CHECKPOINT] Saving checkpoint to: {path}")
    torch.save(payload, path)
    if verbose:
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"[CHECKPOINT] Checkpoint saved ({file_size:.2f} MB)")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    start_time = time.time()
    
    if is_main_process():
        print("=" * 80)
        print("VGGFace2 Embedding Training Script (Full Dataset)")
        print("=" * 80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to vggface2_112x112 folder")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Base learning rate (lower for better stability)")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (increase for faster loading)")
    parser.add_argument("--arc_s", type=float, default=64.0,
                        help="ArcFace scale (fixed)")
    parser.add_argument("--arc_m", type=float, default=0.50,
                        help="ArcFace final margin")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--margin_ramp_epochs", type=int, default=12,
                        help="Epochs to ramp margin from start to final")
    parser.add_argument("--margin_start", type=float, default=0.2,
                        help="Starting margin value")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps (simulates larger batch size)")
    parser.add_argument("--val_max_samples", type=int, default=5000,
                        help="Max samples for verification metrics evaluation (None = use all, faster with subset)")
    parser.add_argument("--backbone", type=str, default="iresnet100", choices=["iresnet100", "iresnet200"],
                        help="Backbone architecture: iresnet100 (faster) or iresnet200 (larger)")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        save_dir=args.save_dir,
        img_size=args.img_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        arc_s=args.arc_s,
        arc_m=args.arc_m,
        amp=(not args.no_amp),
        seed=args.seed,
        resume=args.resume,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        margin_ramp_epochs=args.margin_ramp_epochs,
        margin_start=args.margin_start,
        val_max_samples=args.val_max_samples if args.val_max_samples > 0 else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if is_main_process():
        print("\n[CONFIG] Training Configuration:")
        print(f"  Data root: {cfg.data_root}")
        print(f"  Save directory: {cfg.save_dir}")
        print(f"  Image size: {cfg.img_size}x{cfg.img_size}")
        print(f"  Embedding dimension: {cfg.embedding_dim}")
        print(f"  Batch size: {cfg.batch_size} (per GPU)")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Learning rate: {cfg.lr}")
        print(f"  Weight decay: {cfg.weight_decay}")
        print(f"  Warmup epochs: {cfg.warmup_epochs}")
        print(f"  ArcFace scale (s): {cfg.arc_s} (fixed)")
        print(f"  ArcFace margin (m): {cfg.margin_start} -> {cfg.arc_m} (ramp over {cfg.margin_ramp_epochs} epochs)")
        print(f"  Early stopping patience: {cfg.early_stopping_patience} epochs")
        print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
        if cfg.resume:
            print(f"  Resume from: {cfg.resume}")
        print()

    # Initialize distributed training
    is_ddp, local_rank = init_distributed()
    device = get_device(local_rank)
    
    if is_main_process():
        print(f"[INIT] Distributed training: {is_ddp}")
        print(f"[INIT] Device: {device}")
        if is_ddp:
            world_size = dist.get_world_size()
            effective_batch_size = cfg.batch_size * world_size * cfg.gradient_accumulation_steps
            print(f"[INIT] World size: {world_size} GPUs")
            print(f"[INIT] Effective batch size: {effective_batch_size} (batch_size={cfg.batch_size} Ã— {world_size} GPUs Ã— {cfg.gradient_accumulation_steps} accumulation)")
            print(f"[INIT] Rank: {dist.get_rank()}, Local rank: {local_rank}")
        print()
    
    seed_everything(cfg.seed, verbose=is_main_process())

    # Prepare dataset split: 8631 train, 500 val
    if is_main_process():
        print("[DATA] Preparing dataset split...")
        print(f"[DATA] Scanning identities in {cfg.data_root}...")
    
    # Only scan identities on main process to avoid duplicate work in DDP
    data_root_path = Path(cfg.data_root)
    if is_main_process():
        # Use listdir instead of glob for faster scanning
        all_identity_indices = []
        try:
            for folder_name in sorted(os.listdir(data_root_path)):
                if folder_name.startswith("id_") and (data_root_path / folder_name).is_dir():
                    try:
                        identity_idx = int(folder_name.split("_")[1])
                        all_identity_indices.append(identity_idx)
                    except (ValueError, IndexError):
                        continue
        except OSError as e:
            if is_main_process():
                print(f"[ERROR] Failed to scan identities: {e}")
            all_identity_indices = []
        
        all_identity_indices.sort()
        total_identities = len(all_identity_indices)
        print(f"[DATA] Found {total_identities} identities")
    else:
        # Non-main processes: wait for main process to finish scanning
        all_identity_indices = []
        total_identities = 0
    
    # In DDP, synchronize the identity list by broadcasting from main process
    if is_ddp:
        # Main process broadcasts the identity list to all processes
        if is_main_process():
            # Convert to tensor for broadcasting
            identity_tensor = torch.tensor(all_identity_indices, dtype=torch.long, device=device)
            identity_count = torch.tensor([len(all_identity_indices)], dtype=torch.long, device=device)
        else:
            # Non-main processes: create placeholder tensors
            identity_count = torch.tensor([0], dtype=torch.long, device=device)
        
        # Broadcast count first
        dist.broadcast(identity_count, src=0)
        total_identities = identity_count.item()
        
        if not is_main_process():
            # Allocate tensor for receiving identity list
            identity_tensor = torch.zeros(total_identities, dtype=torch.long, device=device)
        
        # Broadcast identity list
        dist.broadcast(identity_tensor, src=0)
        
        if not is_main_process():
            # Convert back to list
            all_identity_indices = identity_tensor.cpu().tolist()
        
        # Synchronize before proceeding
        dist.barrier()
        
        # Split: first 8631 for training, last 500 for validation
        train_identity_indices = all_identity_indices[:8631]
        val_identity_indices = all_identity_indices[8631:8631+500]
        
        if len(val_identity_indices) < 500:
            # If not enough identities, use last 500
            val_identity_indices = all_identity_indices[-500:]
            train_identity_indices = all_identity_indices[:-500]
        
        num_classes = len(train_identity_indices)
        
        if is_main_process():
            print(f"[DATA] Train identities: {len(train_identity_indices)}")
            print(f"[DATA] Val identities: {len(val_identity_indices)}")
    
    # Build transforms
    train_tfms, val_tfms = build_transforms(cfg.img_size, verbose=is_main_process())
    
    # Create datasets from folders
    if is_main_process():
        print("[DATA] Creating datasets from folders...")
    
    train_ds = VGGFace2Dataset(
        cfg.data_root,
        train_identity_indices,
        transform=train_tfms,
    )
    val_ds = VGGFace2Dataset(
        cfg.data_root,
        val_identity_indices,
        transform=val_tfms,
    )
    
    if is_main_process():
        print(f"\n[DATA] Dataset Statistics:")
        print(f"  Train identities: {num_classes}")
        print(f"  Train images: {len(train_ds)}")
        print(f"  Val identities: {len(val_identity_indices)}")
        print(f"  Val images: {len(val_ds)}")
        print()
    
    # Create data loaders with P-K sampling
    pk_sampler = None
    train_sampler = None
    if not is_ddp:
        # P-K sampling: ensure P is large enough
        P = cfg.batch_size // 4
        K = 4
        if P * K != cfg.batch_size:
            raise ValueError(f"batch_size {cfg.batch_size} must be divisible by 4 for P-K sampling")
        
        steps_per_epoch = len(train_ds) // cfg.batch_size
        pk_sampler = PKBatchSampler(train_ds.targets, P=P, K=K, steps_per_epoch=steps_per_epoch, seed=cfg.seed)
        
        if is_main_process():
            print(f"[DATA] Using P-K sampling: P={P} identities, K={K} images per identity")
        
        train_loader = DataLoader(
            train_ds,
            batch_sampler=pk_sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )
    else:
        # DDP: Use DistributedSampler and optimize DataLoader settings
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,  # Shuffle for better data distribution
            drop_last=True,  # Drop last incomplete batch for consistent batch sizes
        )
        # Reduce num_workers per process in DDP to avoid resource contention
        # With 2 GPUs, if num_workers=8, each GPU gets 8 workers = 16 total (too many)
        ddp_num_workers = max(1, cfg.num_workers // 2)  # Reduce workers per GPU
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            shuffle=False,  # Sampler handles shuffling
            num_workers=ddp_num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            persistent_workers=(ddp_num_workers > 0),
            prefetch_factor=1 if ddp_num_workers > 0 else None,  # Reduce prefetch for DDP
        )
    
    # For validation, also optimize num_workers in DDP
    if is_ddp:
        val_num_workers = max(1, cfg.num_workers // 2)  # Reduce workers per GPU
    else:
        val_num_workers = cfg.num_workers
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(val_num_workers > 0),
        prefetch_factor=1 if val_num_workers > 0 else None,  # Reduce prefetch for DDP
    )
    
    if is_main_process():
        print("[DATA] Creating data loaders...")
        if is_ddp:
            print(f"[DATA] DDP: Using {ddp_num_workers} workers per GPU (reduced from {cfg.num_workers} to avoid resource contention)")
        print(f"[DATA] Train loader: {len(train_loader)} batches")
        print(f"[DATA] Val loader: {len(val_loader)} batches")
        print()
    
    # Create fixed evaluation indices ONCE (reproducible epoch-to-epoch comparison)
    # These indices are saved to disk and reused for all epochs
    fixed_eval_indices = None
    if is_main_process() and cfg.val_max_samples is not None:
        fixed_eval_indices = get_or_create_fixed_eval_indices(
            total_samples=len(val_ds),
            max_samples=cfg.val_max_samples,
            save_dir=cfg.save_dir,
            seed=cfg.seed,
            filename="fixed_eval_indices.npy",
        )
        print()
    
    # Create model
    backbone_fn = iresnet100 if args.backbone == "iresnet100" else iresnet200
    backbone_name = "IR-SE-100" if args.backbone == "iresnet100" else "IR-SE-200"
    if is_main_process():
        print(f"[MODEL] Creating {backbone_name} backbone...")
    backbone = backbone_fn(embedding_dim=cfg.embedding_dim, dropout=0.4, use_se=True).to(device)
    
    if is_main_process():
        backbone_params = count_parameters(backbone)
        print(f"[MODEL] Backbone parameters: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    
    if is_main_process():
        print(f"[MODEL] Creating Dynamic ArcFace head...")
    head = DynamicArcMarginProduct(cfg.embedding_dim, num_classes, s=cfg.arc_s, m=cfg.margin_start).to(device)
    
    if is_main_process():
        head_params = count_parameters(head)
        total_params = backbone_params + head_params
        print(f"[MODEL] Head parameters: {head_params:,}")
        print(f"[MODEL] Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print()
    
    # Wrap with DDP - optimize for performance
    if is_ddp:
        # Use optimized DDP settings for better performance
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Set to False for better performance (all params used)
            gradient_as_bucket_view=True,  # More efficient gradient communication
            broadcast_buffers=False,  # Don't broadcast BN buffers every iteration (faster)
        )
        head = torch.nn.parallel.DistributedDataParallel(
            head,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    
    # Optimizer and scheduler
    if is_main_process():
        print("[OPTIMIZER] Setting up optimizer and scheduler...")
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    
    # Calculate steps per epoch accounting for gradient accumulation
    # Effective optimizer steps = batches / accumulation_steps
    steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    
    # LR schedule: warmup then cosine decay to 1e-5
    min_lr_ratio = 1e-5 / cfg.lr  # Decay to 1e-5
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    if is_main_process():
        print(f"[OPTIMIZER] LR schedule: warmup {warmup_steps} steps, then cosine to {cfg.lr * min_lr_ratio:.6f}")
        print()
    
    # Loss: CrossEntropyLoss with label_smoothing=0 (no smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    
    if is_main_process() and cfg.amp and device.type == "cuda":
        print(f"[AMP] Mixed precision training enabled (FP16)")
        print()
    
    # Training state
    start_epoch = 0
    best_roc_auc = -1.0  # Track ROC AUC (or negative loss as fallback)
    last_epoch_completed = start_epoch - 1
    global_step = 0
    epochs_without_improvement = 0
    
    # Resume logic (similar to before but track ROC AUC)
    if cfg.resume and os.path.isfile(cfg.resume):
        if is_main_process():
            print(f"[RESUME] Loading checkpoint from: {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location="cpu")
        
        is_backbone_only = "head" not in ckpt or "optimizer" not in ckpt
        
        if is_backbone_only:
            backbone_state = ckpt["backbone"]
            if any(k.startswith("module.") for k in backbone_state.keys()):
                backbone_state = {k.replace("module.", ""): v for k, v in backbone_state.items()}
            
            if is_ddp:
                backbone.module.load_state_dict(backbone_state)
            else:
                backbone.load_state_dict(backbone_state)
            
            best_roc_auc = ckpt.get("best_roc_auc", ckpt.get("roc_auc", -1.0))
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
            scheduler.last_epoch = global_step - 1
            epochs_without_improvement = 0
            
            if is_main_process():
                print(f"[RESUME] Backbone loaded, resuming from epoch {start_epoch}")
        else:
            backbone_state = ckpt["backbone"]
            head_state = ckpt["head"]
            if any(k.startswith("module.") for k in backbone_state.keys()):
                backbone_state = {k.replace("module.", ""): v for k, v in backbone_state.items()}
            if any(k.startswith("module.") for k in head_state.keys()):
                head_state = {k.replace("module.", ""): v for k, v in head_state.items()}
            
            if is_ddp:
                backbone.module.load_state_dict(backbone_state)
                head.module.load_state_dict(head_state)
            else:
                backbone.load_state_dict(backbone_state)
                head.load_state_dict(head_state)
            
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_roc_auc = ckpt.get("best_roc_auc", ckpt.get("roc_auc", -1.0))
            global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
            scheduler.last_epoch = global_step - 1
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            
            if is_main_process():
                print(f"[RESUME] Full checkpoint loaded, resuming from epoch {start_epoch}")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    backbone_best_path = os.path.join(cfg.save_dir, "backbone_best.pt")
    
    # Only barrier before training starts (not needed for checkpoint saving)
    if is_ddp:
        dist.barrier()
        if is_main_process():
            print("[DDP] All processes synchronized, starting training...")
    
    if is_main_process():
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print()
    
    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start_time = time.time()
        
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{cfg.epochs}")
            print(f"{'='*80}")
        
        # Set epoch for samplers
        if is_ddp:
            train_sampler.set_epoch(epoch)
        elif pk_sampler is not None:
            pk_sampler.set_epoch(epoch)
        
        # Update margin schedule (slower ramp)
        m_now = linear_ramp(
            epoch - start_epoch,
            start=cfg.margin_start,
            end=cfg.arc_m,
            ramp_epochs=cfg.margin_ramp_epochs
        )
        # Keep scale fixed at 64
        s_now = cfg.arc_s
        
        if is_ddp:
            head.module.set_margin(m_now)
            head.module.set_scale(s_now)
        else:
            head.set_margin(m_now)
            head.set_scale(s_now)
        
        if is_main_process() and (epoch < start_epoch + cfg.margin_ramp_epochs or epoch % 5 == 0):
            print(f"[SCHEDULE] Margin: {m_now:.4f}, Scale: {s_now:.2f} (fixed)")
        
        backbone.train()
        head.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        total_for_acc = 0  # Track samples for which we computed accuracy
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", disable=not is_main_process())
        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Gradient accumulation: zero gradients at start of accumulation cycle
            if step % cfg.gradient_accumulation_steps == 1:
                optimizer.zero_grad(set_to_none=True)
            
            # For DDP: use no_sync during accumulation to avoid expensive gradient syncs
            # Only sync gradients at the last step of accumulation
            is_accumulation_step = (step % cfg.gradient_accumulation_steps != 0)
            
            # Compute embeddings (needed for both training and accuracy)
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embs = backbone(images)
                    logits = head(embs, labels)
                    loss = criterion(logits, labels)
                
                loss = loss / cfg.gradient_accumulation_steps
                
                # Use no_sync context for DDP during accumulation steps
                if is_ddp and is_accumulation_step:
                    with backbone.no_sync(), head.no_sync():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()
                
                # Step optimizer only at end of accumulation cycle
                if step % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    # Clear cache periodically to prevent memory fragmentation
                    if is_ddp and step % 200 == 0:
                        torch.cuda.empty_cache()
            else:
                embs = backbone(images)
                logits = head(embs, labels)
                loss = criterion(logits, labels)
                
                loss = loss / cfg.gradient_accumulation_steps
                
                # Use no_sync context for DDP during accumulation steps
                if is_ddp and is_accumulation_step:
                    with backbone.no_sync(), head.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                
                if step % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                    optimizer.step()
                    # Clear cache periodically to prevent memory fragmentation
                    if is_ddp and step % 200 == 0:
                        torch.cuda.empty_cache()
            
            # Step scheduler only after accumulation (moved outside no_sync context)
            if step % cfg.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
            
            # Accumulate metrics (unscale loss for logging)
            running_loss += loss.item() * cfg.gradient_accumulation_steps * images.size(0)
            total += labels.size(0)
            
            # Compute accuracy less frequently to reduce overhead (every 50 steps)
            # Note: embs is computed in both AMP and non-AMP branches above, so it's available here
            if step % 50 == 0 or step == len(train_loader):
                with torch.no_grad():
                    # Detach embs to ensure it's not affected by gradient computation
                    # Convert to float32 if it's in float16 from autocast
                    embs_for_acc = embs.detach()
                    if embs_for_acc.dtype == torch.float16:
                        embs_for_acc = embs_for_acc.float()
                    
                    # Use cosine_logits for accuracy (without margin, more meaningful)
                    if is_ddp:
                        cosine_logits = head.module.cosine_logits(embs_for_acc)
                    else:
                        cosine_logits = head.cosine_logits(embs_for_acc)
                    pred = cosine_logits.argmax(dim=1)
                    batch_correct = (pred == labels).sum().item()
                    correct += batch_correct
                    total_for_acc += labels.size(0)
            
            # Update progress bar less frequently (every 50 steps) to reduce overhead
            if step % 50 == 0 or step == len(train_loader):
                avg_loss = running_loss / total if total > 0 else 0.0
                train_acc = correct / total_for_acc if total_for_acc > 0 else 0.0
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}", lr=f"{lr_now:.6f}")
        
        # Validation: Compute verification metrics at end of each epoch
        # Uses FIXED evaluation indices for reproducible epoch-to-epoch comparison
        if is_main_process():
            print(f"\n[VALIDATION] Computing verification metrics at end of epoch {epoch+1}...")
            print(f"[VALIDATION] Using FIXED evaluation indices for reproducible comparison")
            val_start_time = time.time()
            # Use fixed indices for reproducible sampling (same indices every epoch)
            metrics = compute_verification_metrics(
                backbone, val_loader, device,
                max_samples=cfg.val_max_samples,
                fixed_indices=fixed_eval_indices,  # <-- Key change: use fixed indices
            )
            val_time = time.time() - val_start_time
            
            roc_auc = metrics["roc_auc"]
            eer = metrics["eer"]
            tar_1e3 = metrics["tar_at_far_1e3"]
            tar_1e4 = metrics["tar_at_far_1e4"]
            
            print(f"[VALIDATION] Completed in {val_time:.2f} seconds")
            print(f"[VALIDATION] ROC AUC: {roc_auc:.6f}")
            print(f"[VALIDATION] EER: {eer:.6f}")
            print(f"[VALIDATION] TAR@FAR=1e-3: {tar_1e3:.6f}")
            print(f"[VALIDATION] TAR@FAR=1e-4: {tar_1e4:.6f}")
        else:
            metrics = None
            roc_auc = -1.0
            eer = None
            tar_1e3 = None
            tar_1e4 = None
        
        # Broadcast ROC AUC to all processes for early stopping
        if is_ddp:
            if is_main_process():
                roc_auc_tensor = torch.tensor(roc_auc, device=device)
            else:
                roc_auc_tensor = torch.tensor(-1.0, device=device)
            dist.broadcast(roc_auc_tensor, src=0)
            roc_auc = roc_auc_tensor.item()
        
        epoch_time = time.time() - epoch_start_time
        last_epoch_completed = epoch
        
        if is_main_process():
            final_acc = correct / total_for_acc if total_for_acc > 0 else 0.0
            print(f"\n[EPOCH SUMMARY] Epoch {epoch+1} completed in {epoch_time:.2f} seconds ({epoch_time/60:.1f} min)")
            print(f"  Train Loss: {running_loss/total:.4f}")
            print(f"  Train Accuracy: {final_acc:.4f} (computed every 50 steps)")
            if metrics:
                print(f"  ROC AUC: {roc_auc:.6f}")
                print(f"  EER: {eer:.6f}")
                print(f"  TAR@FAR=1e-3: {tar_1e3:.6f}")
                print(f"  TAR@FAR=1e-4: {tar_1e4:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every 5 epochs (or on last epoch)
        if is_main_process() and ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1):
            backbone_state = backbone.module.state_dict() if is_ddp else backbone.state_dict()
            head_state = head.module.state_dict() if is_ddp else head.state_dict()
            
            ckpt_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(ckpt_path, {
                "epoch": epoch,
                "backbone": backbone_state,
                "head": head_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_roc_auc": best_roc_auc,
                "global_step": global_step,
                "epochs_without_improvement": epochs_without_improvement,
                "cfg": cfg.__dict__,
            }, verbose=True)
        
        # Save best model based on ROC AUC (if computed) or training loss (fallback)
        current_loss = running_loss / total if total > 0 else float('inf')
        
        # Use ROC AUC if available, otherwise use negative loss (lower loss = better)
        if roc_auc > 0:
            current_metric = roc_auc
            best_metric = best_roc_auc if best_roc_auc > 0 else -1.0
            improved = current_metric > best_metric
            metric_name = "ROC AUC"
        else:
            # Use negative loss as metric (lower loss = better)
            current_metric = -current_loss
            best_metric = best_roc_auc if best_roc_auc < 0 else float('-inf')
            improved = current_metric > best_metric
            metric_name = "Training Loss"
        
        if improved:
            old_best = best_roc_auc
            best_roc_auc = current_metric
            epochs_without_improvement = 0
            
            if is_main_process():
                if roc_auc > 0:
                    print(f"\n[BEST MODEL] New best ROC AUC: {best_roc_auc:.6f} (previous: {old_best:.6f})")
                else:
                    print(f"\n[BEST MODEL] New best training loss: {-best_roc_auc:.4f} (previous: {-old_best:.4f})")
                backbone_state = backbone.module.state_dict() if is_ddp else backbone.state_dict()
                save_checkpoint(backbone_best_path, {
                    "backbone": backbone_state,
                    "embedding_dim": cfg.embedding_dim,
                    "img_size": cfg.img_size,
                    "roc_auc": roc_auc if roc_auc > 0 else None,
                    "train_loss": current_loss if roc_auc <= 0 else None,
                    "eer": eer if metrics else None,
                    "tar_at_far_1e3": tar_1e3 if metrics else None,
                    "tar_at_far_1e4": tar_1e4 if metrics else None,
                    "epoch": epoch,
                }, verbose=True)
        else:
            epochs_without_improvement += 1
            if is_main_process():
                print(f"\n[EARLY STOPPING] No improvement in {metric_name}. Epochs without improvement: {epochs_without_improvement}/{cfg.early_stopping_patience}")
        
        # Early stopping based on ROC AUC
        if epochs_without_improvement >= cfg.early_stopping_patience:
            if is_main_process():
                print(f"\n{'='*80}")
                print(f"[EARLY STOPPING] Stopping training early!")
                print(f"  No improvement for {epochs_without_improvement} epochs")
                if best_roc_auc > 0:
                    print(f"  Best ROC AUC: {best_roc_auc:.6f}")
                else:
                    print(f"  Best training loss: {-best_roc_auc:.4f}")
                print(f"{'='*80}")
                
                backbone_state = backbone.module.state_dict() if is_ddp else backbone.state_dict()
                head_state = head.module.state_dict() if is_ddp else head.state_dict()
                
                last_ckpt_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
                save_checkpoint(last_ckpt_path, {
                    "epoch": epoch,
                    "backbone": backbone_state,
                    "head": head_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_roc_auc": best_roc_auc,
                    "global_step": global_step,
                    "epochs_without_improvement": epochs_without_improvement,
                    "cfg": cfg.__dict__,
                }, verbose=True)
            break
    
    # Save final checkpoint
    if is_main_process() and epochs_without_improvement < cfg.early_stopping_patience:
        backbone_state = backbone.module.state_dict() if is_ddp else backbone.state_dict()
        head_state = head.module.state_dict() if is_ddp else head.state_dict()
        
        last_ckpt_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
        save_checkpoint(last_ckpt_path, {
            "epoch": last_epoch_completed,
            "backbone": backbone_state,
            "head": head_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_roc_auc": best_roc_auc,
            "global_step": global_step,
            "epochs_without_improvement": epochs_without_improvement,
            "cfg": cfg.__dict__,
        }, verbose=True)
    
    total_time = time.time() - start_time
    
    if is_main_process():
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Total training time: {total_time/3600:.2f} hours")
        if best_roc_auc > 0:
            print(f"Best ROC AUC: {best_roc_auc:.6f}")
        else:
            print(f"Best training loss: {-best_roc_auc:.4f}")
        print(f"Best backbone saved at: {backbone_best_path}")
        actual_epochs_trained = last_epoch_completed + 1 - start_epoch
        print(f"Total epochs trained: {actual_epochs_trained}")
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"Training stopped early due to no improvement for {epochs_without_improvement} epochs")
        print("=" * 80)
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()