#!/usr/bin/env python3
"""
Standard Face Verification Benchmark Evaluation Script

Evaluate trained face recognition models on standard benchmarks:
- LFW (Labeled Faces in the Wild)
- CFP-FP (Celebrities in Frontal-Profile)
- AgeDB-30
- And more...

Usage:
    python evaluate_benchmarks.py \
        --checkpoint ./checkpoints/backbone_best.pt \
        --lfw_path /path/to/lfw \
        --cfp_fp_path /path/to/cfp_fp

The script follows the standard 10-fold cross-validation protocol used in
face recognition research, making results directly comparable to published work.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold


# ---------------------------
# Model: IR-SE (Same as train.py)
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
    """InsightFace iresnet100 definition: layers = [2, 9, 20, 2]"""
    return IResNet([2, 9, 20, 2], embedding_dim=embedding_dim, dropout=dropout, use_se=use_se,
                   bn_momentum=bn_momentum, bn_eps=bn_eps)


def iresnet200(embedding_dim=512, dropout=0.4, use_se=True, bn_momentum=0.9, bn_eps=1e-5):
    """InsightFace iresnet200 definition: layers = [6, 26, 60, 6]"""
    return IResNet([6, 26, 60, 6], embedding_dim=embedding_dim, dropout=dropout, use_se=use_se,
                   bn_momentum=bn_momentum, bn_eps=bn_eps)


# ---------------------------
# Pair Dataset for Benchmarks
# ---------------------------
class PairDataset(Dataset):
    """
    Dataset for standard face verification benchmarks (LFW, CFP-FP, AgeDB, etc.)
    
    Expected directory structure:
        benchmark_root/
            pairs.txt        # Pair annotations file
            images/          # Aligned face images (112x112)
                name1_0001.jpg
                name2_0001.jpg
                ...
    
    Pairs file format (LFW style):
        10                          # Number of folds
        Name1    1    4             # Positive pair: Name1_0001.jpg and Name1_0004.jpg
        Name1    Name2    1    1    # Negative pair: Name1_0001.jpg and Name2_0001.jpg
        ...
    """
    def __init__(
        self,
        benchmark_root: str,
        pairs_file: str = "pairs.txt",
        transform=None,
        image_ext: str = ".jpg",
    ):
        self.root = Path(benchmark_root)
        self.transform = transform
        self.image_ext = image_ext
        self.pairs = []
        self.issame = []
        
        pairs_path = self.root / pairs_file
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        
        self._parse_pairs(pairs_path)
    
    def _parse_pairs(self, pairs_path: Path):
        """Parse LFW-style pairs file."""
        with open(pairs_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines (number of folds, etc.)
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 1 and parts[0].isdigit():
                start_idx = i + 1
                continue
            if len(parts) >= 3:
                start_idx = i
                break
        
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            
            if len(parts) == 3:
                # Positive pair: Name, idx1, idx2
                name, idx1, idx2 = parts
                img1 = self._get_image_path(name, int(idx1))
                img2 = self._get_image_path(name, int(idx2))
                self.pairs.append((img1, img2))
                self.issame.append(True)
            
            elif len(parts) == 4:
                # Negative pair: Name1, idx1, Name2, idx2
                name1, idx1, name2, idx2 = parts
                img1 = self._get_image_path(name1, int(idx1))
                img2 = self._get_image_path(name2, int(idx2))
                self.pairs.append((img1, img2))
                self.issame.append(False)
        
        print(f"[BENCHMARK] Loaded {len(self.pairs)} pairs ({sum(self.issame)} positive, {len(self.pairs) - sum(self.issame)} negative)")
    
    def _get_image_path(self, name: str, idx: int) -> str:
        """Get image path for a person and index."""
        # Try different naming conventions
        candidates = [
            self.root / f"{name}" / f"{name}_{idx:04d}{self.image_ext}",  # LFW style
            self.root / f"{name}_{idx:04d}{self.image_ext}",  # Flat style
            self.root / "images" / f"{name}_{idx:04d}{self.image_ext}",  # images/ subfolder
            self.root / f"{name}" / f"{idx:04d}{self.image_ext}",  # Just index in folder
        ]
        
        for path in candidates:
            if path.exists():
                return str(path)
        
        # Return first candidate (will fail at load time with informative error)
        return str(candidates[0])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        issame = self.issame[idx]
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Failed to load {img1_path}: {e}")
            img1 = Image.new('RGB', (112, 112), (0, 0, 0))
        
        try:
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Failed to load {img2_path}: {e}")
            img2 = Image.new('RGB', (112, 112), (0, 0, 0))
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, issame


# ---------------------------
# Benchmark Evaluation Functions
# ---------------------------
@torch.no_grad()
def evaluate_benchmark(
    backbone: nn.Module,
    benchmark_root: str,
    device: torch.device,
    transform,
    pairs_file: str = "pairs.txt",
    batch_size: int = 64,
    nfolds: int = 10,
    flip_test: bool = True,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluate on a standard face verification benchmark (LFW, CFP-FP, AgeDB, etc.)
    using the standard 10-fold cross-validation protocol.
    
    Args:
        backbone: Face recognition model
        benchmark_root: Path to benchmark dataset
        device: Device to run inference on
        transform: Image transformation
        pairs_file: Name of pairs file
        batch_size: Batch size for inference
        nfolds: Number of cross-validation folds (default: 10)
        flip_test: Whether to use horizontal flip for test-time augmentation
        num_workers: Number of data loading workers
    
    Returns:
        Dictionary with accuracy, AUC, TAR@FAR metrics
    """
    # Check if benchmark exists
    if not os.path.exists(benchmark_root):
        print(f"[BENCHMARK] Error: Benchmark path does not exist: {benchmark_root}")
        return {"accuracy": 0.0, "accuracy_std": 0.0, "auc": 0.0, "error": "Path not found"}
    
    backbone.eval()
    
    # Load pairs dataset
    try:
        dataset = PairDataset(benchmark_root, pairs_file=pairs_file, transform=transform)
    except FileNotFoundError as e:
        print(f"[BENCHMARK] Error: {e}")
        return {"accuracy": 0.0, "accuracy_std": 0.0, "auc": 0.0, "error": str(e)}
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Extract embeddings for all pairs
    embeddings1 = []
    embeddings2 = []
    issame_list = []
    
    print(f"[BENCHMARK] Extracting embeddings...")
    for img1, img2, issame in tqdm(loader, desc="Processing pairs"):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        
        # Get embeddings
        emb1 = backbone(img1)
        emb2 = backbone(img2)
        
        if flip_test:
            # Horizontal flip test-time augmentation
            emb1_flip = backbone(torch.flip(img1, dims=[3]))
            emb2_flip = backbone(torch.flip(img2, dims=[3]))
            emb1 = F.normalize(emb1 + emb1_flip, p=2, dim=1)
            emb2 = F.normalize(emb2 + emb2_flip, p=2, dim=1)
        
        embeddings1.append(emb1.cpu().numpy())
        embeddings2.append(emb2.cpu().numpy())
        issame_list.extend(issame.numpy().tolist())
    
    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    issame_arr = np.array(issame_list)
    
    # Compute cosine similarity (embeddings are L2-normalized)
    similarities = np.sum(embeddings1 * embeddings2, axis=1)
    
    # 10-fold cross-validation
    print(f"[BENCHMARK] Running {nfolds}-fold cross-validation...")
    kfold = KFold(n_splits=nfolds, shuffle=False)
    
    accuracies = []
    thresholds = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(similarities)):
        # Find best threshold on training fold
        train_sims = similarities[train_idx]
        train_issame = issame_arr[train_idx]
        
        # Grid search for best threshold
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(-1, 1, 0.005):
            pred = train_sims > thresh
            acc = np.mean(pred == train_issame)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        # Apply threshold to test fold
        test_sims = similarities[test_idx]
        test_issame = issame_arr[test_idx]
        test_pred = test_sims > best_thresh
        test_acc = np.mean(test_pred == test_issame)
        
        accuracies.append(test_acc)
        thresholds.append(best_thresh)
    
    # Compute overall ROC AUC
    overall_auc = roc_auc_score(issame_arr, similarities)
    
    # Compute TAR@FAR
    fpr, tpr, _ = roc_curve(issame_arr, similarities)
    tar_at_far_1e1 = tpr[np.argmin(np.abs(fpr - 1e-1))]
    tar_at_far_1e2 = tpr[np.argmin(np.abs(fpr - 1e-2))]
    tar_at_far_1e3 = tpr[np.argmin(np.abs(fpr - 1e-3))]
    tar_at_far_1e4 = tpr[np.argmin(np.abs(fpr - 1e-4))]
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    return {
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "auc": overall_auc,
        "tar_at_far_1e1": tar_at_far_1e1,
        "tar_at_far_1e2": tar_at_far_1e2,
        "tar_at_far_1e3": tar_at_far_1e3,
        "tar_at_far_1e4": tar_at_far_1e4,
        "threshold": np.mean(thresholds),
        "threshold_std": np.std(thresholds),
    }


def load_backbone(checkpoint_path: str, device: torch.device, backbone_type: str = "iresnet100") -> nn.Module:
    """
    Load a trained backbone model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        backbone_type: Type of backbone ("iresnet100" or "iresnet200")
    
    Returns:
        Loaded backbone model
    """
    print(f"[MODEL] Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get embedding dimension from checkpoint if available
    embedding_dim = checkpoint.get("embedding_dim", 512)
    
    # Create backbone
    if backbone_type == "iresnet200":
        backbone = iresnet200(embedding_dim=embedding_dim, dropout=0.0, use_se=True)
    else:
        backbone = iresnet100(embedding_dim=embedding_dim, dropout=0.0, use_se=True)
    
    # Load weights
    if "backbone" in checkpoint:
        state_dict = checkpoint["backbone"]
    else:
        state_dict = checkpoint
    
    # Remove "module." prefix if present (from DDP training)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)
    backbone.eval()
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"[MODEL] Checkpoint epoch: {checkpoint['epoch']}")
    if "roc_auc" in checkpoint and checkpoint["roc_auc"]:
        print(f"[MODEL] Checkpoint ROC AUC: {checkpoint['roc_auc']:.6f}")
    
    param_count = sum(p.numel() for p in backbone.parameters())
    print(f"[MODEL] Backbone loaded: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    return backbone


def build_transform(img_size: int = 112):
    """Build image transformation for evaluation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition model on standard verification benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on LFW only
    python evaluate_benchmarks.py --checkpoint ./checkpoints/backbone_best.pt --lfw_path /path/to/lfw

    # Evaluate on LFW and CFP-FP
    python evaluate_benchmarks.py --checkpoint ./checkpoints/backbone_best.pt \\
        --lfw_path /path/to/lfw --cfp_fp_path /path/to/cfp_fp

    # Evaluate with iresnet200 backbone
    python evaluate_benchmarks.py --checkpoint ./checkpoints/backbone_best.pt \\
        --backbone iresnet200 --lfw_path /path/to/lfw
        """
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained backbone checkpoint")
    parser.add_argument("--backbone", type=str, default="iresnet100",
                        choices=["iresnet100", "iresnet200"],
                        help="Backbone architecture")
    parser.add_argument("--img_size", type=int, default=112,
                        help="Image size for evaluation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--no_flip_test", action="store_true",
                        help="Disable horizontal flip test-time augmentation")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file for results (optional)")
    
    # Benchmark paths
    parser.add_argument("--lfw_path", type=str, default="",
                        help="Path to LFW benchmark dataset (aligned 112x112 with pairs.txt)")
    parser.add_argument("--cfp_fp_path", type=str, default="",
                        help="Path to CFP-FP benchmark dataset")
    parser.add_argument("--cfp_ff_path", type=str, default="",
                        help="Path to CFP-FF benchmark dataset")
    parser.add_argument("--agedb_path", type=str, default="",
                        help="Path to AgeDB-30 benchmark dataset")
    parser.add_argument("--calfw_path", type=str, default="",
                        help="Path to CALFW benchmark dataset")
    parser.add_argument("--cplfw_path", type=str, default="",
                        help="Path to CPLFW benchmark dataset")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Collect benchmark paths
    benchmarks = {}
    if args.lfw_path:
        benchmarks["LFW"] = args.lfw_path
    if args.cfp_fp_path:
        benchmarks["CFP-FP"] = args.cfp_fp_path
    if args.cfp_ff_path:
        benchmarks["CFP-FF"] = args.cfp_ff_path
    if args.agedb_path:
        benchmarks["AgeDB-30"] = args.agedb_path
    if args.calfw_path:
        benchmarks["CALFW"] = args.calfw_path
    if args.cplfw_path:
        benchmarks["CPLFW"] = args.cplfw_path
    
    if not benchmarks:
        print("Error: No benchmark paths provided. Use --lfw_path and/or --cfp_fp_path")
        print("Run with --help for usage information.")
        return
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("=" * 70)
    print("Face Recognition Benchmark Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Flip test: {'Enabled' if not args.no_flip_test else 'Disabled'}")
    print(f"Benchmarks: {list(benchmarks.keys())}")
    print("=" * 70)
    print()
    
    # Load model
    backbone = load_backbone(args.checkpoint, device, args.backbone)
    
    # Build transform
    transform = build_transform(args.img_size)
    
    # Evaluate on each benchmark
    all_results = {}
    
    for name, path in benchmarks.items():
        print(f"\n{'='*70}")
        print(f"Evaluating on {name}")
        print(f"{'='*70}")
        print(f"Path: {path}")
        
        results = evaluate_benchmark(
            backbone=backbone,
            benchmark_root=path,
            device=device,
            transform=transform,
            batch_size=args.batch_size,
            flip_test=not args.no_flip_test,
            num_workers=args.num_workers,
        )
        
        all_results[name] = results
        
        if "error" not in results:
            print(f"\n[RESULTS] {name}:")
            print(f"  Accuracy: {results['accuracy']*100:.2f}% ± {results['accuracy_std']*100:.2f}%")
            print(f"  AUC: {results['auc']:.4f}")
            print(f"  Threshold: {results['threshold']:.4f} ± {results['threshold_std']:.4f}")
            print(f"  TAR@FAR=1e-1: {results['tar_at_far_1e1']*100:.2f}%")
            print(f"  TAR@FAR=1e-2: {results['tar_at_far_1e2']*100:.2f}%")
            print(f"  TAR@FAR=1e-3: {results['tar_at_far_1e3']*100:.2f}%")
            print(f"  TAR@FAR=1e-4: {results['tar_at_far_1e4']*100:.2f}%")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Benchmark':<15} {'Accuracy':<20} {'AUC':<10} {'TAR@FAR=1e-3':<15}")
    print("-" * 70)
    for name, results in all_results.items():
        if "error" not in results:
            acc_str = f"{results['accuracy']*100:.2f}% ± {results['accuracy_std']*100:.2f}%"
            print(f"{name:<15} {acc_str:<20} {results['auc']:.4f}     {results['tar_at_far_1e3']*100:.2f}%")
        else:
            print(f"{name:<15} Error: {results['error']}")
    print("=" * 70)
    
    # Save results to JSON if requested
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "backbone": args.backbone,
            "flip_test": not args.no_flip_test,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
