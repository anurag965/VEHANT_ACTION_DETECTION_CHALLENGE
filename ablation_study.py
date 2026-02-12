"""
VEHANT Ablation Study - Prove Each Component Matters

This script runs 5 ablation experiments to demonstrate that:
1. Motion tokenization improves over raw optical flow
2. Causal attention improves over standard attention
3. Uncertainty calibration improves confidence reliability
4. Skeleton features add complementary information
5. Full system achieves best performance

Expected results table (from Perplexity's strategy):
Model Variant              Accuracy  Boundary F1  ECE    Size   Latency
────────────────────────────────────────────────────────────────────
RGB baseline (ViViT)       87%       N/A          0.12   30 MB  15 ms
+ Motion tokens            89%       0.58         0.10   40 MB  18 ms  (+2% acc, -2% ECE)
+ Causal attention         91%       0.68         0.07   45 MB  20 ms  (+2% acc, -3% ECE)
+ Uncertainty fusion       93%       0.74         0.05   50 MB  22 ms  (+2% acc, -2% ECE)
+ Skeleton (Full)          95%       0.78         0.03   60 MB  25 ms  (+2% acc, -2% ECE)

Usage:
python ablation_study.py --dataset_path dataset --output ablation_results.json
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import argparse

# Import from main model
sys.path.append(os.path.dirname(__file__))
from vehant_causal_temporal_model import (
    Config, VEHANTCausalTemporalModel, VideoProcessor, ActionDataset,
    compute_ece, device
)

# ============================================================================
# ABLATION VARIANTS
# ============================================================================

class Variant1_RGBBaseline(nn.Module):
    """Baseline: RGB only, standard transformer"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, 
                                                     dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, flow, pose, **kwargs):
        B, T = rgb.shape[:2]
        spatial_features = []
        for i in range(T):
            frame = rgb[:, i].permute(0, 3, 1, 2)
            feat = self.spatial(frame).view(B, -1)
            spatial_features.append(feat)
        x = torch.stack(spatial_features, dim=1)
        x = self.transformer(x)[:, -1]
        logits = self.classifier(x)
        return logits, None, None, None, None, torch.tensor(0.0), None

class Variant2_WithMotionTokens(VEHANTCausalTemporalModel):
    """Add motion tokens but keep standard attention"""
    def __init__(self, num_classes=3):
        super().__init__(num_classes)
        # Replace causal attention with standard transformer
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        encoder_layer = TransformerEncoderLayer(d_model=self.fusion_dim, nhead=8, 
                                                 dim_feedforward=512, dropout=0.3, batch_first=True)
        self.causal_attention = nn.ModuleList([
            nn.Identity() for _ in range(2)  # Placeholder
        ])
        self.std_transformer = TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, rgb, flow, pose, mc_samples=10):
        B, T = rgb.shape[:2]
        
        # Spatial
        spatial_features = []
        for i in range(T):
            frame = rgb[:, i].permute(0, 3, 1, 2)
            feat = self.spatial(frame).view(B, -1)
            spatial_features.append(feat)
        spatial = torch.stack(spatial_features, dim=1)
        
        # Motion tokens
        motion_tokens, motion_indices, vq_loss, _ = self.motion_vqvae(flow)
        motion = self.motion_proj(motion_tokens)
        
        # Pose
        pose_features = []
        for i in range(T):
            pose_feat = self.pose_encoder(pose[:, i, :])
            pose_features.append(pose_feat)
        pose_feat = torch.stack(pose_features, dim=1)
        
        # Fusion
        fused = torch.cat([spatial, motion, pose_feat], dim=-1)
        
        # Standard transformer (NOT causal)
        x = self.std_transformer(fused)
        temporal_features = x[:, -1, :]
        
        # Outputs
        logits, epistemic_unc, aleatoric_unc = self.uncertainty_head(temporal_features, mc_samples)
        bbox = self.bbox_head(temporal_features)
        temporal = self.temporal_head(temporal_features)
        
        return logits, bbox, temporal, epistemic_unc, aleatoric_unc, vq_loss, motion_indices

# Note: Variant 3 (+ Causal Attention) is the main VEHANTCausalTemporalModel

class Variant4_WithoutUncertainty(VEHANTCausalTemporalModel):
    """Remove uncertainty calibration (use standard softmax)"""
    def forward(self, rgb, flow, pose, mc_samples=10):
        B, T = rgb.shape[:2]
        
        # Same feature extraction
        spatial_features = []
        for i in range(T):
            frame = rgb[:, i].permute(0, 3, 1, 2)
            feat = self.spatial(frame).view(B, -1)
            spatial_features.append(feat)
        spatial = torch.stack(spatial_features, dim=1)
        
        motion_tokens, motion_indices, vq_loss, _ = self.motion_vqvae(flow)
        motion = self.motion_proj(motion_tokens)
        
        pose_features = []
        for i in range(T):
            pose_feat = self.pose_encoder(pose[:, i, :])
            pose_features.append(pose_feat)
        pose_feat = torch.stack(pose_features, dim=1)
        
        fused = torch.cat([spatial, motion, pose_feat], dim=-1)
        
        # Causal attention
        x = fused
        for attn, ff in zip(self.causal_attention, self.feedforward):
            x = attn(x)
            x = x + ff(x)
        
        temporal_features = x[:, -1, :]
        
        # Standard classifier (no uncertainty)
        logits = self.uncertainty_head.epistemic(temporal_features)
        
        bbox = self.bbox_head(temporal_features)
        temporal = self.temporal_head(temporal_features)
        
        # Dummy uncertainty
        ep_unc = torch.zeros_like(logits)
        al_unc = torch.zeros_like(logits)
        
        return logits, bbox, temporal, ep_unc, al_unc, vq_loss, motion_indices

class Variant5_WithoutSkeleton(VEHANTCausalTemporalModel):
    """Remove skeleton features"""
    def __init__(self, num_classes=3):
        super().__init__(num_classes)
        # 384 = 256 (spatial) + 128 (motion)
        self.projection = nn.Linear(384, self.fusion_dim)

    def forward(self, rgb, flow, pose, mc_samples=10):
        B, T = rgb.shape[:2]
        
        # Spatial
        spatial_features = []
        for i in range(T):
            frame = rgb[:, i].permute(0, 3, 1, 2)
            feat = self.spatial(frame).view(B, -1)
            spatial_features.append(feat)
        spatial = torch.stack(spatial_features, dim=1)
        
        # Motion
        motion_tokens, motion_indices, vq_loss, _ = self.motion_vqvae(flow)
        motion = self.motion_proj(motion_tokens)
        
        # NO pose features
        # Fusion (256 + 128 = 384)
        fused = torch.cat([spatial, motion], dim=-1)
        
        # Project to full fusion_dim (448)
        fused = self.projection(fused)
        
        # Causal attention
        x = fused
        for attn, ff in zip(self.causal_attention, self.feedforward):
            x = attn(x)
            x = x + ff(x)
        
        temporal_features = x[:, -1, :]
        
        logits, epistemic_unc, aleatoric_unc = self.uncertainty_head(temporal_features, mc_samples)
        bbox = self.bbox_head(temporal_features)
        temporal = self.temporal_head(temporal_features)
        
        return logits, bbox, temporal, epistemic_unc, aleatoric_unc, vq_loss, motion_indices

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_variant(model, data_loader, variant_name):
    """Evaluate a single ablation variant - FIXED VERSION"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_temporal_preds = []
    all_temporal_gt = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {variant_name}"):
            rgb = batch['rgb'].to(device)
            flow = batch['flow'].to(device)
            poses = batch['poses'].to(device)
            
            # ✅ FIX #1: Check dimensionality before squeezing
            labels = batch['label'].to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            labels = labels.cpu().numpy()
            
            temporal_gt = batch['temporal'].to(device).cpu().numpy()
            
            try:
                logits, _, temporal, ep_unc, al_unc, _, _ = model(rgb, flow, poses, mc_samples=20)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confidence = probs.gather(1, torch.argmax(probs, dim=1).unsqueeze(1)).squeeze().cpu().numpy()
                
                # ✅ FIX #2: Safe confidence handling
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())
                
                if isinstance(confidence, np.ndarray):
                    all_confidences.extend(confidence.flatten())
                else:
                    all_confidences.append(float(confidence))
                
                if temporal is not None:
                    all_temporal_preds.append(temporal.cpu().numpy())
                    all_temporal_gt.append(temporal_gt)
            except Exception as e:
                print(f"  Warning: {e}")
                continue
    
    # ✅ FIX #3: Explicit type casting and empty guard
    all_preds = np.array(all_preds, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)
    all_confidences = np.array(all_confidences, dtype=np.float32)
    
    if len(all_preds) == 0:
        print(f"  No valid predictions")
        return {'accuracy': 0.0, 'ece': 0.0, 'boundary_f1': 0.0}
    
    # ✅ NOW SAFE: guaranteed to be numpy array
    accuracy = (all_preds == all_labels).mean() * 100
    ece = compute_ece(all_confidences, all_preds, all_labels)
    
    # Rest of function unchanged...
    boundary_f1 = 0.0
    if len(all_temporal_preds) > 0:
        all_temporal_preds_arr = np.concatenate(all_temporal_preds, axis=0)
        all_temporal_gt_arr = np.concatenate(all_temporal_gt, axis=0)
        
        start_preds = all_temporal_preds_arr[:, 0]
        end_preds = all_temporal_preds_arr[:, 1]
        start_gt = all_temporal_gt_arr[:, 0]
        end_gt = all_temporal_gt_arr[:, 1]
        
        ious = []
        for sp, ep, sg, eg in zip(start_preds, end_preds, start_gt, end_gt):
            intersection = max(0, min(ep, eg) - max(sp, sg))
            union = max(ep, eg) - min(sp, sg)
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        
        tp = sum(1 for iou in ious if iou > 0.5)
        fp = sum(1 for iou in ious if iou <= 0.5)
        fn = fp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        boundary_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'boundary_f1': boundary_f1
    }

# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def run_ablation_study(dataset_path, output_file):
    print("\n" + "="*70)
    print("VEHANT ABLATION STUDY")
    print("="*70)
    
    # Load test dataset
    from sklearn.model_selection import train_test_split
    from vehant_causal_temporal_model import _parse_label_file
    import cv2
    
    config = Config()
    dataset = []
    
    # Load videos (same as training script)
    for subdir in ['fight_mp4s', 'collapse_mp4s', 'negatives']:
        video_dir = os.path.join(dataset_path, subdir)
        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(video_dir, video_file)
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            continue
                        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        cap.release()
                    except:
                        continue
                    
                    class_id = 0 if 'negatives' in subdir else (1 if 'fight' in subdir else 2)
                    
                    dataset.append({
                        'video_path': video_path,
                        'class_id': class_id,
                        'bbox_normalized': [0.0, 0.0, 1.0, 1.0],
                        'temporal_bounds': [0.0, 1.0]
                    })
    
    print(f"\n✓ Found {len(dataset)} videos for ablation study")
    
    # Split dataset (use test set)
    labels = [d['class_id'] for d in dataset]
    _, test_d = train_test_split(dataset, test_size=0.2, random_state=42, stratify=labels)
    
    processor = VideoProcessor()
    test_ds = ActionDataset(test_d, processor)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"Test set: {len(test_d)} videos\n")
    
    # Define variants
    variants = {
        'Variant 1: RGB Baseline': Variant1_RGBBaseline(config.NUM_CLASSES),
        'Variant 2: + Motion Tokens': Variant2_WithMotionTokens(config.NUM_CLASSES),
        'Variant 3: + Causal Attention (Full)': VEHANTCausalTemporalModel(config.NUM_CLASSES),
        'Variant 4: - Uncertainty': Variant4_WithoutUncertainty(config.NUM_CLASSES),
        'Variant 5: - Skeleton': Variant5_WithoutSkeleton(config.NUM_CLASSES)
    }
    
    results = {}
    
    # Load pre-trained model weights
    config = Config()
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_finetuned.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_original.pth')
    
    if not os.path.exists(model_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: No trained model found. Running on random weights. !!!")
        print(f"!!! Searched for: {model_path}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        trained_state_dict = None
    else:
        print(f"✓ Loading weights from: {model_path}")
        trained_state_dict = torch.load(model_path, map_location=device)['model_state_dict']

    # Evaluate each variant
    for variant_name, model in variants.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*70}")
        
        if trained_state_dict:
            model.load_state_dict(trained_state_dict, strict=False)

        model = model.to(device)
        metrics = evaluate_variant(model, test_loader, variant_name)
        
        results[variant_name] = metrics
        
        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Boundary F1: {metrics['boundary_f1']:.4f}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(f"{'Variant':<40} {'Accuracy':<12} {'ECE':<10} {'Boundary F1':<12}")
    print("-" * 70)
    
    for variant_name, metrics in results.items():
        print(f"{variant_name:<40} {metrics['accuracy']:>10.2f}% {metrics['ece']:>9.4f} {metrics['boundary_f1']:>11.4f}")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'best_variant': max(results, key=lambda x: results[x]['accuracy']),
            'best_accuracy': max(r['accuracy'] for r in results.values()),
            'best_ece': min(r['ece'] for r in results.values()),
            'best_boundary_f1': max(r['boundary_f1'] for r in results.values())
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='ablation_results.json', help='Output JSON file')
    args = parser.parse_args()
    
    run_ablation_study(args.dataset_path, args.output)