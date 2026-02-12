"""
VEHANT Action Detection - Testing Script
Generates CSV output with predictions for multiple video clips

Usage:
    python test.py --input_dir <path_to_videos> --output_file <path_to_csv> [--threshold 0.65]

Output CSV format:
    video_name, pred_class_1, x1, y1, x2, y2, pred_class_2, x1, y1, x2, y2, ...

Example:
    python test.py --input_dir ./videos --output_file results.csv
    python test.py --input_dir ./videos --output_file results.csv --threshold 0.7
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import VEHANT components
try:
    from vehant_causal_temporal_model import (
        VEHANTCausalTemporalModel, Config, VideoProcessor, OpticalFlowExtractor, device
    )
except ImportError as e:
    print(f"ERROR: Failed to import VEHANT model: {e}")
    print("Make sure vehant_causal_temporal_model.py is in the same directory")
    sys.exit(1)


class VEHANTInference:
    """VEHANT inference engine for batch video processing"""
    
    def __init__(self, model_path, threshold=0.5, device_type=None):
        """
        Initialize VEHANT inference
        
        Args:
            model_path: Path to trained model checkpoint
            threshold: Confidence threshold for predictions (0.0 to 1.0)
            device_type: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device_type if device_type else device
        self.threshold = max(0.0, min(1.0, threshold))  # Clamp to [0, 1]
        self.config = Config()
        
        # Load model
        self.model = VEHANTCausalTemporalModel(self.config.NUM_CLASSES).to(self.device)
        self.model_loaded = False
        
        if not os.path.exists(model_path):
            print(f"⚠ WARNING: Model not found at {model_path}")
            print(f"  Available paths:")
            print(f"    - models/causal_temporal/vehant_causal_temporal_finetuned.pth")
            print(f"    - models/causal_temporal/vehant_causal_temporal_original.pth")
            print(f"  Using randomly initialized model for testing")
        else:
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model_loaded = True
                print(f"✓ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠ Error loading model weights: {e}")
                print(f"  Using randomly initialized model for testing")
        
        self.model.eval()
        self.processor = VideoProcessor()
        self.flow_extractor = OpticalFlowExtractor()
        
        print(f"✓ Device: {self.device}")
        print(f"✓ Confidence threshold: {self.threshold:.2f}")
        print(f"✓ Classes: {self.config.CLASS_NAMES}")
        print(f"✓ Model loaded: {self.model_loaded}\n")
    
    def extract_sequence(self, video_path):
        """
        Extract frame, flow, and pose sequences from video
        
        Returns:
            dict with keys: frames, flows, poses
            or None if video cannot be read
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frames, flows, poses, prev_frame, frame_idx = [], [], [], None, 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % self.config.FRAME_SAMPLE_RATE == 0:
                    # Resize frame to model input size
                    frame_resized_bgr = cv2.resize(frame, (self.config.IMG_SIZE, self.config.IMG_SIZE))
                    frame_resized_rgb = cv2.cvtColor(frame_resized_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Extract pose landmarks
                    pose_data = self.processor.pose_extractor.extract_pose(frame_resized_rgb)
                    poses.append(pose_data)
                    
                    # Compute optical flow
                    flow = self.flow_extractor.compute_flow(
                        prev_frame, 
                        frame_resized_bgr, 
                        self.config.OPTICAL_FLOW_SIZE
                    )
                    flows.append(flow)
                    
                    frames.append(frame_resized_bgr)
                    prev_frame = frame_resized_bgr.copy()
                    
                    if len(frames) >= self.config.SEQUENCE_LENGTH:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            # Pad sequences if necessary
            while len(frames) < self.config.SEQUENCE_LENGTH:
                frames.append(np.zeros((self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8))
                flows.append(np.zeros((self.config.OPTICAL_FLOW_SIZE, self.config.OPTICAL_FLOW_SIZE), dtype=np.float32))
                poses.append(np.zeros(99, dtype=np.float32))
            
            return {
                'frames': frames[:self.config.SEQUENCE_LENGTH],
                'flows': flows[:self.config.SEQUENCE_LENGTH],
                'poses': poses[:self.config.SEQUENCE_LENGTH]
            }
        
        except Exception as e:
            print(f"    Error extracting sequence: {e}")
            return None
    
    def predict_video(self, video_path):
        """
        Predict actions in video
        
        Returns:
            List of detections: [{'class_id': int, 'bbox': ndarray, 'confidence': float}, ...]
            or empty list if no detections above threshold
        """
        seq = self.extract_sequence(video_path)
        if seq is None:
            return []
        
        frames = np.array(seq['frames'], dtype=np.float32) / 255.0
        flows = np.array(seq['flows'], dtype=np.float32)
        poses = np.array(seq['poses'], dtype=np.float32)
        
        rgb_t = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
        flow_t = torch.FloatTensor(flows).unsqueeze(0).to(self.device)
        poses_t = torch.FloatTensor(poses).unsqueeze(0).to(self.device)
        
        detections = []
        
        try:
            with torch.no_grad():
                logits, bbox, temporal, ep_unc, al_unc, _, _ = self.model(
                    rgb_t, flow_t, poses_t, mc_samples=10
                )
                
                # Get predictions
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class])
                
                # Apply confidence threshold
                if confidence >= self.threshold:
                    bbox_normalized = bbox[0].cpu().numpy()
                    detections.append({
                        'class_id': pred_class,
                        'bbox': bbox_normalized,
                        'confidence': confidence
                    })
        
        except Exception as e:
            print(f"    Error during inference: {e}")
        
        return detections
    
    def close(self):
        """Clean up resources"""
        self.processor.close()


def get_video_files(input_dir):
    """
    Get list of video files from directory
    
    Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    if not os.path.isdir(input_dir):
        return []
    
    for file in sorted(os.listdir(input_dir)):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(os.path.join(input_dir, file))
    
    return video_files


def process_videos(input_dir, output_file, threshold=0.65, model_path=None):
    """
    Process all videos in input directory and generate CSV output
    
    Args:
        input_dir: Directory containing video files
        output_file: Path to output CSV file
        threshold: Confidence threshold for predictions
        model_path: Path to model checkpoint (auto-detected if None)
    
    Returns:
        True if successful, False otherwise
    """
    
    # Auto-detect model path if not provided
    # ✅ MODIFIED: Default to finetuned model
    if model_path is None:
        model_path = 'models/causal_temporal/vehant_causal_temporal_finetuned.pth'
        
        # Fallback to original if finetuned doesn't exist
        if not os.path.exists(model_path):
            fallback_path = 'models/causal_temporal/vehant_causal_temporal_original.pth'
            if os.path.exists(fallback_path):
                model_path = fallback_path
                print(f"Finetuned model not found, using fallback: {fallback_path}\n")
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"✗ Error: Input directory not found: {input_dir}")
        return False
    
    # Get video files
    video_files = get_video_files(input_dir)
    if not video_files:
        print(f"✗ Error: No video files found in {input_dir}")
        print(f"  Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm")
        return False
    
    # Print header
    print(f"{'='*70}")
    print(f"VEHANT ACTION DETECTION - BATCH TESTING")
    print(f"{'='*70}")
    print(f"\nInput directory: {input_dir}")
    print(f"Found {len(video_files)} video files")
    print(f"Output file: {output_file}")
    print(f"Confidence threshold: {threshold:.2f}")
    print(f"Model: {model_path}")
    print()
    
    # Initialize inference engine
    print(f"{'='*70}")
    print("Initializing VEHANT Model")
    print(f"{'='*70}\n")
    inference = VEHANTInference(model_path, threshold)
    
    # Process videos
    print(f"{'='*70}")
    print("Processing Videos")
    print(f"{'='*70}\n")
    
    results = []
    processed_count = 0
    error_count = 0
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        
        try:
            detections = inference.predict_video(video_path)
            
            # Create row: video_name, [class_id, x1, y1, x2, y2, ...] for each detection
            row = [video_name]
            
            if detections:
                for det in detections:
                    class_id = det['class_id']
                    bbox = det['bbox']  # [x1, y1, x2, y2] normalized
                    confidence = det['confidence']
                    
                    row.extend([
                        str(class_id),
                        f"{bbox[0]:.4f}",
                        f"{bbox[1]:.4f}",
                        f"{bbox[2]:.4f}",
                        f"{bbox[3]:.4f}"
                    ])
            else:
                # No detection above threshold - add default negative class with full frame
                row.extend(['0', '0.0000', '0.0000', '1.0000', '1.0000'])
            
            results.append(row)
            processed_count += 1
        
        except Exception as e:
            print(f"  ✗ Error processing {video_name}: {e}")
            # Add error entry
            row = [video_name, '0', '0.0000', '0.0000', '1.0000', '1.0000']
            results.append(row)
            error_count += 1
    
    # Write CSV output
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
        
        print(f"\n{'='*70}")
        print(f"✓ RESULTS SAVED")
        print(f"{'='*70}")
        print(f"\nOutput file: {output_file}")
        print(f"Videos processed: {processed_count}")
        print(f"Videos with errors: {error_count}")
        print(f"Total detections: {sum(1 for row in results if len(row) > 5)}")
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"✗ Error writing CSV: {e}")
        return False
    
    # Clean up
    inference.close()
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='VEHANT Action Detection - Batch Video Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --input_dir ./videos --output_file results.csv
  
  python test.py --input_dir ./videos --output_file results.csv --threshold 0.7
  
  python test.py --input_dir ./videos --output_file results.csv \\
                  --model_path models/my_model.pth
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing video clips to test'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.65,
        help='Confidence threshold for predictions (default: 0.65, range: 0.0-1.0)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint (default: models/causal_temporal/vehant_causal_temporal_finetuned.pth)'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        print(f"ERROR: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)
    
    # Run processing
    success = process_videos(
        args.input_dir,
        args.output_file,
        args.threshold,
        args.model_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
