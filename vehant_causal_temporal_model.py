"""
VEHANT CAUSAL TEMPORAL ACTION DETECTION - FIXED VERSION
File: vehant_casual_temporal_model_FIXED.py

KEY FIXES:
1. Dynamic class weight calculation (fixes imbalance)
2. Confidence thresholding in inference (rejects uncertain predictions)
3. Soft rejection for high negative class uncertainty
4. Better model calibration

Changes from original:
- Line 85: CLASS_WEIGHTS now calculated dynamically
- Inference: Added confidence threshold and rejection mechanism
- Training: Added class distribution logging
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# ============================================================================
# CONFIG - VEHANT CAUSAL TEMPORAL (FIXED)
# ============================================================================

class Config:
    # Paths
    DATASET_PATH = 'dataset'
    CUSTOM_DATA_DIR = 'custom_data'
    MODEL_SAVE_PATH = 'models/causal_temporal'
    RESULTS_PATH = 'results/causal_temporal'
    
    # Video Processing
    FRAME_SAMPLE_RATE = 2
    SEQUENCE_LENGTH = 16
    IMG_SIZE = 320
    OPTICAL_FLOW_SIZE = 64
    
    # Training
    BATCH_SIZE = 4
    GRAD_ACCUMULATION_STEPS = 2
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    PATIENCE = 10
    WARMUP_EPOCHS = 5
    
    # Fine-tuning
    FINETUNE_EPOCHS = 40
    FINETUNE_LR = 0.0001
    FINETUNE_PATIENCE = 8
    
    # Model Architecture
    NUM_CLASSES = 3
    MOTION_VOCAB_SIZE = 256
    HIDDEN_DIM = 448
    NUM_HEADS = 8
    NUM_LAYERS = 2
    
    # Classes
    CLASS_NAMES = ['negative', 'fight', 'collapse']
    # ✅ FIXED: Dynamic calculation instead of static [1.0, 2.5, 2.5]
    CLASS_WEIGHTS = None  # Will be calculated from data
    CLASS_COLORS = {
        0: (0, 255, 0),   # Green for negative
        1: (0, 0, 255),   # Red for fight
        2: (255, 0, 0)    # Blue for collapse
    }
    
    # ✅ NEW: Confidence thresholding parameters
    CONFIDENCE_THRESHOLD = 0.65
    NEGATIVE_CLASS_MIN_PROB = 0.25  # Flag if negative prob > 25% (high uncertainty)

config = Config()
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(config.RESULTS_PATH, exist_ok=True)

print(f"\n{'='*70}")
print("VEHANT CAUSAL TEMPORAL ACTION DETECTION (FIXED)")
print(f"{'='*70}")
print(f"Classes: {config.CLASS_NAMES}")
print(f"Motion Vocab Size: {config.MOTION_VOCAB_SIZE}")

# ✅ NEW: Print thresholding info
print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
print(f"Negative Uncertainty Threshold: {config.NEGATIVE_CLASS_MIN_PROB}")
print(f"{'='*70}\n")

# ============================================================================
# DYNAMIC WEIGHT CALCULATION (NEW)
# ============================================================================

def calculate_class_weights(dataset, num_classes=3):
    """
    Calculate inverse frequency weights from data distribution.
    Prevents class imbalance issues.
    """
    class_counts = [0] * num_classes
    for item in dataset:
        class_counts[item['class_id']] += 1
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    total = sum(class_counts)
    for i, (name, count) in enumerate(zip(config.CLASS_NAMES, class_counts)):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Inverse frequency weights
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(1.0)
        else:
            weight = total / (num_classes * count)
            weights.append(weight)
    
    # Normalize
    sum_weights = sum(weights)
    weights = [w / sum_weights * num_classes for w in weights]
    
    print(f"\n⚖️  CALCULATED CLASS WEIGHTS:")
    for name, weight in zip(config.CLASS_NAMES, weights):
        print(f"  {name}: {weight:.3f}")
    print()
    
    return weights

# ============================================================================
# POSE EXTRACTOR
# ============================================================================

class PoseExtractor:
    def __init__(self, model_path='convo/pose_landmarker_lite.task'):
        # ✅ FIXED: Check if model file exists first
        if not os.path.exists(model_path):
            print(f"❌ CRITICAL: Pose model not found at {model_path}")
            print(f"   Download from: https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task")
            print(f"   Save to: convo/pose_landmarker_lite.task")
            self.landmarker = None
        else:
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False)
                self.landmarker = vision.PoseLandmarker.create_from_options(options)
                print("✓ MediaPipe Pose Landmarker loaded.")
            except Exception as e:
                print(f"❌ Failed to load pose model: {e}")
                self.landmarker = None

    def extract_pose(self, frame_numpy_rgb):
        if self.landmarker is None:
            return np.zeros(99, dtype=np.float32)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_numpy_rgb)
            detection_result = self.landmarker.detect(mp_image)
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]
                pose_data = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks]).flatten()
                return pose_data.astype(np.float32)
            return np.zeros(99, dtype=np.float32)
        except:
            return np.zeros(99, dtype=np.float32)

    def close(self):
        if self.landmarker:
            self.landmarker.close()

# ============================================================================
# MOTION TOKEN ENCODER (SAME AS ORIGINAL)
# ============================================================================

class MotionVQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder for Motion Tokenization"""
    def __init__(self, vocab_size=256, embed_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 4, stride=2, padding=1),
        )
        
        self.codebook = nn.Embedding(vocab_size, embed_dim)
        self.codebook.weight.data.uniform_(-1/vocab_size, 1/vocab_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(2)
        x = x.view(B * T, 1, H, W)
        
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e.view(-1, self.embed_dim)
        
        # ONNX-friendly cdist implementation
        x_norm = (z_e_flat**2).sum(1, keepdim=True)
        y_norm = (self.codebook.weight**2).sum(1, keepdim=True)
        y_norm = y_norm.t()
        
        dist = x_norm + y_norm - 2.0 * torch.mm(z_e_flat, self.codebook.weight.t())
        distances = torch.sqrt(torch.clamp(dist, min=0.0))

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices)
        
        quantized = quantized.view(B * T, 8, 8, self.embed_dim)
        quantized = quantized.permute(0, 3, 1, 2)
        
        vq_loss = F.mse_loss(quantized.detach(), z_e.permute(0, 3, 1, 2)) + \
                  0.25 * F.mse_loss(quantized, z_e.permute(0, 3, 1, 2).detach())
        
        quantized = z_e.permute(0, 3, 1, 2) + (quantized - z_e.permute(0, 3, 1, 2)).detach()
        
        reconstructed = self.decoder(quantized)
        
        quantized_pooled = F.adaptive_avg_pool2d(quantized, (1, 1)).squeeze(-1).squeeze(-1)
        quantized_pooled = quantized_pooled.view(B, T, self.embed_dim)
        
        indices_seq = indices.view(B, T, 64)[:, :, 0]
        
        return quantized_pooled, indices_seq, vq_loss, reconstructed

# ============================================================================
# CAUSAL TEMPORAL ATTENTION (SAME AS ORIGINAL)
# ============================================================================

class CausalTemporalAttention(nn.Module):
    """Causal + Anti-Causal Self-Attention for Temporal Modeling"""
    def __init__(self, dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.causal_qkv = nn.Linear(dim, dim * 3)
        self.causal_proj = nn.Linear(dim, dim)
        
        self.anticausal_qkv = nn.Linear(dim, dim * 3)
        self.anticausal_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, T, C = x.shape
        residual = x
        
        # Causal attention
        qkv_causal = self.causal_qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv_causal = qkv_causal.permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_causal[0], qkv_causal[1], qkv_causal[2]
        
        attn_c = (q_c @ k_c.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_c = attn_c.masked_fill(causal_mask == 0, float('-inf'))
        attn_c = F.softmax(attn_c, dim=-1)
        attn_c = self.dropout(attn_c)
        
        out_c = (attn_c @ v_c).transpose(1, 2).reshape(B, T, C)
        out_c = self.causal_proj(out_c)
        
        # Anti-causal attention
        qkv_anticausal = self.anticausal_qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv_anticausal = qkv_anticausal.permute(2, 0, 3, 1, 4)
        q_ac, k_ac, v_ac = qkv_anticausal[0], qkv_anticausal[1], qkv_anticausal[2]
        
        attn_ac = (q_ac @ k_ac.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        anticausal_mask = torch.triu(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_ac = attn_ac.masked_fill(anticausal_mask == 0, float('-inf'))
        attn_ac = F.softmax(attn_ac, dim=-1)
        attn_ac = self.dropout(attn_ac)
        
        out_ac = (attn_ac @ v_ac).transpose(1, 2).reshape(B, T, C)
        out_ac = self.anticausal_proj(out_ac)
        
        out = out_c + out_ac
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        
        return out

# ============================================================================
# UNCERTAINTY HEADS (SAME AS ORIGINAL)
# ============================================================================

class UncertaintyHead(nn.Module):
    """Bayesian Uncertainty Estimation"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        self.epistemic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.aleatoric = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softplus()
        )
    
    def forward(self, x, mc_samples=10):
        # Remove the if statement. Keep dropout active during training.
        # Only reduce samples at inference for speed
        
        logits_samples = []
        for _ in range(mc_samples):
            logits_samples.append(self.epistemic(x))  # Dropout active in training
        
        logits_samples = torch.stack(logits_samples, dim=0)
        mean_logits = logits_samples.mean(dim=0)
        epistemic_unc = logits_samples.var(dim=0)
        aleatoric_unc = self.aleatoric(x)
        
        return mean_logits, epistemic_unc, aleatoric_unc


# ============================================================================
# MAIN MODEL (SAME AS ORIGINAL)
# ============================================================================

class VEHANTCausalTemporalModel(nn.Module):
    """Complete VEHANT System with all components"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.motion_vqvae = MotionVQVAE(vocab_size=config.MOTION_VOCAB_SIZE, embed_dim=64)
        self.motion_proj = nn.Linear(64, 128)
        
        self.pose_encoder = nn.Sequential(
            nn.Linear(99, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fusion_dim = 448
        
        self.causal_attention = nn.ModuleList([
            CausalTemporalAttention(self.fusion_dim, num_heads=config.NUM_HEADS, dropout=0.3)
            for _ in range(config.NUM_LAYERS)
        ])
        
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.Dropout(0.3),
                nn.LayerNorm(self.fusion_dim)
            ) for _ in range(config.NUM_LAYERS)
        ])
        
        self.uncertainty_head = UncertaintyHead(self.fusion_dim, num_classes)
        
        self.bbox_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        
        self.temporal_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # ✅ FIXED: Add sigmoid to bound outputs to [0, 1]
        )
    
    def forward(self, rgb, flow, pose, mc_samples=10):
        B, T = rgb.shape[0], rgb.shape[1]
        
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
# OPTICAL FLOW & VIDEO PROCESSING (SAME AS ORIGINAL)
# ============================================================================

class OpticalFlowExtractor:
    def compute_flow(self, frame1, frame2, size=64):
        try:
            if frame1 is None or frame2 is None:
                return np.zeros((size, size), dtype=np.float32)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray1, gray2, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mag = cv2.resize(mag, (size, size))
            return mag.astype(np.float32) / 255.0
        except:
            return np.zeros((size, size), dtype=np.float32)

class VideoProcessor:
    def __init__(self):
        self.flow_extractor = OpticalFlowExtractor()
        self.pose_extractor = PoseExtractor()
    
    def close(self):
        self.pose_extractor.close()

    def extract_sequence(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frames, flows, poses, prev_frame, frame_idx = [], [], [], None, 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % config.FRAME_SAMPLE_RATE == 0:
                    frame_resized_bgr = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
                    frame_resized_rgb = cv2.cvtColor(frame_resized_bgr, cv2.COLOR_BGR2RGB)
                    
                    pose_data = self.pose_extractor.extract_pose(frame_resized_rgb)
                    poses.append(pose_data)
                    
                    flow = self.flow_extractor.compute_flow(prev_frame, frame_resized_bgr, config.OPTICAL_FLOW_SIZE)
                    flows.append(flow)
                    
                    frames.append(frame_resized_bgr)
                    prev_frame = frame_resized_bgr.copy()
                    
                    if len(frames) >= config.SEQUENCE_LENGTH:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            while len(frames) < config.SEQUENCE_LENGTH:
                frames.append(np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8))
                flows.append(np.zeros((config.OPTICAL_FLOW_SIZE, config.OPTICAL_FLOW_SIZE), dtype=np.float32))
                poses.append(np.zeros(99, dtype=np.float32))
            
            return {
                'frames': frames[:config.SEQUENCE_LENGTH],
                'flows': flows[:config.SEQUENCE_LENGTH],
                'poses': poses[:config.SEQUENCE_LENGTH]
            }
        except:
            return None

# ============================================================================
# DATASET (SAME AS ORIGINAL)
# ============================================================================

class ActionDataset(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        seq = self.processor.extract_sequence(item['video_path'])
        
        if seq is None:
            frames = np.zeros((config.SEQUENCE_LENGTH, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
            flows = np.zeros((config.SEQUENCE_LENGTH, config.OPTICAL_FLOW_SIZE, config.OPTICAL_FLOW_SIZE), dtype=np.float32)
            poses = np.zeros((config.SEQUENCE_LENGTH, 99), dtype=np.float32)
        else:
            frames = np.array(seq['frames'], dtype=np.float32) / 255.0
            flows = np.array(seq['flows'], dtype=np.float32)
            poses = np.array(seq['poses'], dtype=np.float32)
        
        return {
            'rgb': torch.FloatTensor(frames),
            'flow': torch.FloatTensor(flows),
            'poses': torch.FloatTensor(poses),
            'label': torch.LongTensor([item['class_id']]),
            'bbox': torch.FloatTensor(item.get('bbox_normalized', [0.0, 0.0, 1.0, 1.0])),
            'temporal': torch.FloatTensor(item.get('temporal_bounds', [0.0, 1.0]))
        }

# ============================================================================
# COMPUTE ECE (SAME AS ORIGINAL)
# ============================================================================

def compute_ece(confidences, predictions, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

# ============================================================================
# TRAINING (WITH DYNAMIC WEIGHTS)
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs, output_path, stage='train', class_weights=None):
    print(f"\n{'='*70}")
    print(f"TRAINING VEHANT CAUSAL TEMPORAL MODEL - {stage.upper()}")
    print(f"{'='*70}")
    
    # ✅ FIXED: Use provided class_weights instead of static config
    if class_weights is None:
        class_weights = config.CLASS_WEIGHTS
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_bbox = nn.L1Loss()
    criterion_temporal = nn.BCEWithLogitsLoss()
    
    lr = config.FINETUNE_LR if stage == 'finetune' else config.LEARNING_RATE
    patience_thresh = config.FINETUNE_PATIENCE if stage == 'finetune' else config.PATIENCE
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - config.WARMUP_EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    best_loss = float('inf')
    patience = 0
    best_state = None
    
    for epoch in range(num_epochs):
        if stage == 'train' and epoch < config.WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / config.WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale
        
        model.train()
        train_loss = 0
        train_count = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            rgb = batch['rgb'].to(device)
            flow = batch['flow'].to(device)
            poses = batch['poses'].to(device)
            labels = batch['label'].to(device).squeeze(1)
            bboxes = batch['bbox'].to(device)
            temporal_gt = batch['temporal'].to(device)
            
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                logits, bbox, temporal, ep_unc, al_unc, vq_loss, _ = model(rgb, flow, poses, mc_samples=5)
                
                loss_class = criterion_class(logits, labels)
                loss_bbox = criterion_bbox(bbox, bboxes)
                loss_temporal = criterion_temporal(temporal, temporal_gt)
                loss_vq = vq_loss
                
                # ✅ FIXED: Add temporal ordering constraint
                temporal_start = temporal[:, 0]
                temporal_end = temporal[:, 1]
                temporal_order_penalty = torch.relu(temporal_start - temporal_end).mean()
                
                loss = 2.0 * loss_class + 0.3 * loss_bbox + 0.5 * loss_temporal + \
                       0.1 * loss_vq + 0.2 * temporal_order_penalty
                loss = loss / config.GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config.GRAD_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.GRAD_ACCUMULATION_STEPS
            train_count += 1
        
        train_loss /= max(train_count, 1)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_confidences = []
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                flow = batch['flow'].to(device)
                poses = batch['poses'].to(device)
                labels = batch['label'].to(device).squeeze(1)
                
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    logits, _, _, ep_unc, al_unc, _, _ = model(rgb, flow, poses, mc_samples=10)
                    val_loss += criterion_class(logits, labels).item()
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                confidence = probs.gather(1, preds.unsqueeze(1)).squeeze()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_confidences.extend(confidence.cpu().numpy())
        
        val_loss /= max(len(val_loader), 1)
        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0) * 100
        ece = compute_ece(np.array(val_confidences), np.array(val_preds), np.array(val_labels))
        
        if stage == 'train' and epoch >= config.WARMUP_EPOCHS:
            scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = model.state_dict().copy()
            print(f"✓ Epoch {epoch+1}: Train={train_loss:.4f} | Val={val_loss:.4f}, "
                  f"Acc={val_acc:.2f}%, F1={val_f1:.2f}%, ECE={ece:.4f} (BEST)")
        else:
            patience += 1
            print(f"✗ Epoch {epoch+1}: Train={train_loss:.4f} | Val={val_loss:.4f}, "
                  f"Acc={val_acc:.2f}%, F1={val_f1:.2f}%, ECE={ece:.4f} ({patience}/{patience_thresh})")
        
        if patience >= patience_thresh:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }, output_path)
    print(f"✓ Saved to {output_path}")

# ============================================================================
# LABEL PARSING (SAME AS ORIGINAL)
# ============================================================================

def _parse_label_file(label_filepath, video_width, video_height):
    """Parse label file with bbox coordinates and temporal bounds"""
    try:
        with open(label_filepath, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = list(map(float, line.split()))
                if len(parts) >= 5:
                    label_value = int(parts[0])
                    
                    if label_value == 1:
                        class_id = 1
                    elif label_value == 2:
                        class_id = 2
                    else:
                        class_id = 0
                    
                    x1, y1, x2, y2 = parts[1:5]
                    
                    bbox = [
                        max(0.0, min(1.0, x1 / max(video_width, 1))),
                        max(0.0, min(1.0, y1 / max(video_height, 1))),
                        max(0.0, min(1.0, x2 / max(video_width, 1))),
                        max(0.0, min(1.0, y2 / max(video_height, 1)))
                    ]
                    
                    temporal = [0.0, 1.0]
                    if len(parts) >= 7:
                        temporal = [parts[5], parts[6]]
                    
                    return class_id, bbox, temporal
    except:
        pass
    
    return 0, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0]

# ============================================================================
# INFERENCE WITH CONFIDENCE THRESHOLDING (NEW)
# ============================================================================

def inference_with_thresholding(model, video_path, threshold=None):
    """
    Inference with confidence thresholding and uncertainty handling.
    ✅ NEW: Rejects low-confidence predictions
    ✅ NEW: Flags high negative uncertainty
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD
    
    processor = VideoProcessor()
    seq = processor.extract_sequence(video_path)
    
    if seq is None:
        return {'error': 'Cannot read video'}
    
    frames = np.array(seq['frames'], dtype=np.float32) / 255.0
    flows = np.array(seq['flows'], dtype=np.float32)
    poses = np.array(seq['poses'], dtype=np.float32)
    
    rgb = torch.FloatTensor(frames).unsqueeze(0).to(device)
    flow = torch.FloatTensor(flows).unsqueeze(0).to(device)
    poses_t = torch.FloatTensor(poses).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits, bbox, temporal, ep_unc, al_unc, _, motion_indices = model(
            rgb, flow, poses_t, mc_samples=20
        )
        
        probs = F.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        all_probs = probs.cpu().numpy()
        bbox_output = bbox[0].cpu().numpy()
    
    processor.close()
    
    # ✅ FIXED: Confidence thresholding
    rejected = False
    rejection_reason = None
    
    if confidence < threshold:
        rejected = True
        rejection_reason = f"Confidence {confidence:.3f} below threshold {threshold}"
        pred_class = 0  # Default to negative
    
    # ✅ NEW: Check for high negative uncertainty
    neg_prob = all_probs[0]
    soft_rejection = False
    if pred_class != 0 and neg_prob > config.NEGATIVE_CLASS_MIN_PROB:
        soft_rejection = True
        if not rejection_reason:
            rejection_reason = f"High negative uncertainty ({neg_prob:.3f})"
    
    return {
        'class_id': pred_class,
        'class_name': config.CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'probabilities': {name: float(p) for name, p in zip(config.CLASS_NAMES, all_probs)},
        'bbox': [float(x) for x in bbox_output],
        'temporal_onset': int(temporal[0, 0].item() * config.SEQUENCE_LENGTH),
        'temporal_cessation': int(temporal[0, 1].item() * config.SEQUENCE_LENGTH),
        # ✅ NEW: Rejection information
        'rejected': rejected,
        'soft_rejection': soft_rejection,
        'rejection_reason': rejection_reason,
        'confidence_threshold': threshold
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'finetune', 'inference'], required=True)
    parser.add_argument('--video', type=str, help='Path to test video for inference')
    parser.add_argument('--threshold', type=float, default=0.65, help='Confidence threshold for inference')
    args = parser.parse_args()
    
    if args.stage == 'train':
        print("\n" + "="*70)
        print("STAGE 1: TRAINING ON ORIGINAL DATASET")
        print("="*70)
        
        # Load dataset (same as before)
        dataset = []
        
        # Fight videos
        fight_dir = os.path.join(config.DATASET_PATH, 'fight_mp4s')
        fight_txt_dir = os.path.join(config.DATASET_PATH, 'fight_txts')
        if os.path.exists(fight_dir):
            print(f"\nLoading fight videos...")
            for video_file in tqdm(os.listdir(fight_dir)):
                if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                video_path = os.path.join(fight_dir, video_file)
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        continue
                    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                    if video_width == 0 or video_height == 0:
                        continue
                except:
                    continue
                
                txt_filename = os.path.splitext(video_file)[0] + '.txt'
                txt_path = os.path.join(fight_txt_dir, txt_filename)
                
                class_id, bbox, temporal = 1, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0]
                if os.path.exists(txt_path):
                    class_id, bbox, temporal = _parse_label_file(txt_path, video_width, video_height)
                
                dataset.append({
                    'video_path': video_path,
                    'class_id': class_id,
                    'bbox_normalized': bbox,
                    'temporal_bounds': temporal
                })
        
        # Collapse videos
        collapse_dir = os.path.join(config.DATASET_PATH, 'collapse_mp4s')
        collapse_txt_dir = os.path.join(config.DATASET_PATH, 'collapse_txts')
        if os.path.exists(collapse_dir):
            print(f"\nLoading collapse videos...")
            for video_file in tqdm(os.listdir(collapse_dir)):
                if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                video_path = os.path.join(collapse_dir, video_file)
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        continue
                    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                    if video_width == 0 or video_height == 0:
                        continue
                except:
                    continue
                
                txt_filename = os.path.splitext(video_file)[0] + '.txt'
                txt_path = os.path.join(collapse_txt_dir, txt_filename)
                
                class_id, bbox, temporal = 2, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0]
                if os.path.exists(txt_path):
                    class_id, bbox, temporal = _parse_label_file(txt_path, video_width, video_height)
                
                dataset.append({
                    'video_path': video_path,
                    'class_id': class_id,
                    'bbox_normalized': bbox,
                    'temporal_bounds': temporal
                })
        
        # Negative videos
        negatives_dir = os.path.join(config.DATASET_PATH, 'negatives')
        if os.path.exists(negatives_dir):
            print(f"\nLoading negative videos...")
            for video_file in tqdm(os.listdir(negatives_dir)):
                if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                video_path = os.path.join(negatives_dir, video_file)
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        continue
                    cap.release()
                except:
                    continue
                
                dataset.append({
                    'video_path': video_path,
                    'class_id': 0,
                    'bbox_normalized': [0.0, 0.0, 1.0, 1.0],
                    'temporal_bounds': [0.0, 1.0]
                })
        
        if not dataset:
            print("✗ No videos found!")
            exit(1)
        
        print(f"\n{'='*70}")
        print(f"✓ Found {len(dataset)} total videos")
        for i, name in enumerate(config.CLASS_NAMES):
            count = sum(1 for d in dataset if d['class_id'] == i)
            print(f"  {name}: {count}")
        print(f"{'='*70}\n")
        
        # ✅ FIXED: Calculate dynamic weights
        class_weights = calculate_class_weights(dataset)
        config.CLASS_WEIGHTS = class_weights
        
        # Split dataset
        labels = [d['class_id'] for d in dataset]
        train_d, test_d = train_test_split(dataset, test_size=0.2, random_state=42, stratify=labels)
        train_d, val_d = train_test_split(train_d, test_size=0.2, random_state=42, 
                                          stratify=[d['class_id'] for d in train_d])
        
        print(f"Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)}\n")
        
        processor = VideoProcessor()
        train_ds = ActionDataset(train_d, processor)
        val_ds = ActionDataset(val_d, processor)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Train model
        model = VEHANTCausalTemporalModel(config.NUM_CLASSES).to(device)
        output_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_original.pth')
        # ✅ FIXED: Pass dynamic class weights to training
        train_model(model, train_loader, val_loader, config.EPOCHS, output_path, 
                   stage='train', class_weights=class_weights)
        
        processor.close()

        print(f"\n{'='*70}")
        print(f"✓ STAGE 1 COMPLETE!")
        print(f"Model saved: {output_path}")
        print(f"{'='*70}")
    
    elif args.stage == 'finetune':
        print("\n" + "="*70)
        print("STAGE 2: FINE-TUNING ON CUSTOM DATA (FIXED)")
        print("="*70)
        
        dataset = []
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.CUSTOM_DATA_DIR, class_name)
            if os.path.exists(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        dataset.append({
                            'video_path': os.path.join(class_dir, video_file),
                            'class_id': config.CLASS_NAMES.index(class_name),
                            'bbox_normalized': [0.1, 0.1, 0.9, 0.9],
                            'temporal_bounds': [0.0, 1.0]
                        })
        
        if not dataset:
            print(f"✗ No videos found in {config.CUSTOM_DATA_DIR}")
            exit(1)
        
        print(f"\n✓ Found {len(dataset)} videos")
        
        # ✅ FIXED: Calculate dynamic weights
        class_weights = calculate_class_weights(dataset)
        config.CLASS_WEIGHTS = class_weights
        
        labels = [d['class_id'] for d in dataset]
        train_d, val_d = train_test_split(dataset, test_size=0.2, random_state=42, stratify=labels)
        
        print(f"\nTrain: {len(train_d)}, Val: {len(val_d)}")
        
        processor = VideoProcessor()
        train_ds = ActionDataset(train_d, processor)
        val_ds = ActionDataset(val_d, processor)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Load pre-trained model
        original_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_original.pth')
        if not os.path.exists(original_path):
            print(f"✗ Pre-trained model not found: {original_path}")
            exit(1)
        
        model = VEHANTCausalTemporalModel(config.NUM_CLASSES).to(device)
        checkpoint = torch.load(original_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pre-trained model loaded")
        
        # Freeze spatial and motion encoders
        #for param in model.spatial.parameters():
        #    param.requires_grad = False
        #for param in model.motion_vqvae.parameters():
        #    param.requires_grad = False
        
        finetuned_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_finetuned.pth')
        # ✅ FIXED: Pass dynamic class weights to training
        train_model(model, train_loader, val_loader, config.FINETUNE_EPOCHS, finetuned_path, 
                   stage='finetune', class_weights=class_weights)
        
        processor.close()
        
        print(f"\n{'='*70}")
        print(f"✓ STAGE 2 COMPLETE!")
        print(f"Model saved: {finetuned_path}")
        print(f"{'='*70}")
    
    elif args.stage == 'inference':
        if not args.video:
            print("✗ Please provide --video path")
            exit(1)
        
        print(f"\n{'='*70}")
        print(f"VEHANT INFERENCE WITH VISUALIZATION")
        print(f"{'='*70}")
        print(f"Input video: {args.video}")
        print(f"Confidence Threshold: {args.threshold}")
        
        finetuned_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_finetuned.pth')
        if not os.path.exists(finetuned_path):
            finetuned_path = os.path.join(config.MODEL_SAVE_PATH, 'vehant_causal_temporal_original.pth')
        
        if not os.path.exists(finetuned_path):
            print(f"✗ Model not found. Train first.")
            exit(1)
        
        model = VEHANTCausalTemporalModel(config.NUM_CLASSES).to(device)
        checkpoint = torch.load(finetuned_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model loaded")
        print(f"{'='*70}\n")
        
        # ================================================================
        # INTEGRATED VIDEO VISUALIZATION
        # ================================================================
        
        # Open video
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"✗ Cannot open video")
            exit(1)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
        
        # Setup output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.RESULTS_PATH, f'vehant_inference_viz_{timestamp}.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (w, h))
        print(f"✓ Output: {output_path}\n")
        
        # ================================================================
        # ✅ CLEAN VISUALIZATION - ONLY BOUNDING BOX
        # ================================================================
        
        class VizColors:
            NEGATIVE = (0, 255, 0)      # Green for negative
            FIGHT = (0, 0, 255)         # Red for fight
            COLLAPSE = (255, 0, 0)      # Blue for collapse
            CLASS_NAMES = ['negative', 'fight', 'collapse']
            CLASS_COLORS = {0: NEGATIVE, 1: FIGHT, 2: COLLAPSE}
        
        def visualize_frame(frame, predictions, frame_idx, total_frames):
            """
            ✅ CLEAN VISUALIZATION: Draw ONLY bounding box with class label
            All other info (confidence, probabilities, uncertainty) goes to JSON.
            """
            h, w = frame.shape[:2]
            
            # Draw bounding box if available
            if 'bbox' in predictions and predictions['bbox'] is not None:
                bbox = predictions['bbox']
                x1 = max(0, int(bbox[0] * w))
                y1 = max(0, int(bbox[1] * h))
                x2 = min(w, int(bbox[2] * w))
                y2 = min(h, int(bbox[3] * h))
                
                # Get class color
                class_id = predictions.get('class_id', 0)
                color = VizColors.CLASS_COLORS.get(class_id, (255, 255, 0))
                
                # Draw bounding box outline (2px)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw class label with background
                class_name = predictions.get('class_name', 'unknown')
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                text_size = cv2.getTextSize(class_name, font, font_scale, thickness)[0]
                
                # Background for text
                cv2.rectangle(frame, (x1, y1-25), (x1+text_size[0]+5, y1), color, -1)
                # Class name text
                cv2.putText(frame, class_name, (x1+2, y1-7), font, font_scale, (255,255,255), thickness)
            
            return frame
        
        # Process video frames
        frame_buffer = []
        frame_idx = 0
        results_list = []
        
        print(f"Processing video...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            
            if len(frame_buffer) == config.SEQUENCE_LENGTH:
                rgb_seq = np.array([f.astype(np.float32)/255.0 for f in frame_buffer])
                
                flow_seq = []
                for i in range(len(frame_buffer)-1):
                    gray1 = cv2.cvtColor(frame_buffer[i], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame_buffer[i+1], cv2.COLOR_BGR2GRAY)
                    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray1, gray2, None)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    mag = cv2.resize(mag, (config.OPTICAL_FLOW_SIZE, config.OPTICAL_FLOW_SIZE))
                    flow_seq.append(mag.astype(np.float32) / 255.0)
                flow_seq.append(flow_seq[-1])
                
                with torch.no_grad():
                    rgb_t = torch.FloatTensor(rgb_seq).unsqueeze(0).to(device)
                    flow_t = torch.FloatTensor(flow_seq).unsqueeze(0).to(device)
                    poses_t = torch.zeros(1, len(frame_buffer), 99).to(device)
                    
                    logits, bbox, temporal, ep_unc, al_unc, _, _ = model(rgb_t, flow_t, poses_t, mc_samples=10)
                    
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_class = np.argmax(probs)
                    confidence = float(probs[pred_class])
                
                predictions = {
                    'class_id': int(pred_class),
                    'class_name': VizColors.CLASS_NAMES[pred_class],
                    'confidence': confidence,
                    'probabilities': {VizColors.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
                    'bbox': bbox[0].cpu().numpy() if bbox is not None else None,
                    'temporal': temporal[0].cpu().numpy() if temporal is not None else None,
                    'epistemic_unc': float(ep_unc.mean()) if ep_unc is not None else 0.0,
                    'aleatoric_unc': float(al_unc.mean()) if al_unc is not None else 0.0,
                    'rejected': confidence < args.threshold,
                }

                # visualize only the current (last) frame in the buffer
                current_frame = frame_buffer[-1]
                vis_frame = visualize_frame(current_frame.copy(), predictions, frame_idx, total_frames)
                out.write(vis_frame)

                result_entry = {
                    'frame_idx': frame_idx,
                    'class_id': predictions['class_id'],
                    'class_name': predictions['class_name'],
                    'confidence': predictions['confidence'],
                    'probabilities': predictions['probabilities']
                }
                results_list.append(result_entry)

                print(
                    f"  Frame {frame_idx+1}/{total_frames} - "
                    f"{predictions['class_name']} ({predictions['confidence']:.1%})"
                )

                # slide window by 1 frame
                frame_buffer = frame_buffer[1:]
                frame_idx += 1

        
        cap.release()
        out.release()
        
        print(f"\n{'='*70}")
        print(f"✓ VISUALIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Output video: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"{'='*70}\n")
        
        results_file = os.path.join(config.RESULTS_PATH, f'inference_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'video': args.video,
                'threshold': args.threshold,
                'output_video': output_path,
                'results': results_list
            }, f, indent=2)
        
        print(f"✓ Results saved to {results_file}\n")