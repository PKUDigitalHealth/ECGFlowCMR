import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import linalg
import lpips
from torchvision.models import inception_v3, Inception_V3_Weights
from i3d_local import InceptionI3d


# -------------------------------------------------------------------------
# Frechet Distance (FID/FVD)
# -------------------------------------------------------------------------
def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Standard Frechet Distance Calculation"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print(f"WARNING: fid calculation produces singular product; adding {eps} to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"WARNING: Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# -------------------------------------------------------------------------
# FVD Helpers
# -------------------------------------------------------------------------
def smart_load_state_dict(model, state_dict):
    model_keys = set(model.state_dict().keys())
    
    # 1. Branch mapping (b0 -> branch0)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if 'Mixed_' in k and '.b' in k:
            parts = k.split('.')
            if len(parts) > 1:
                branch_map = {
                    'b0': 'branch0', 'b1a': 'branch1a', 'b1b': 'branch1b',
                    'b2a': 'branch2a', 'b2b': 'branch2b', 'b3b': 'branch3b'
                }
                if parts[1] in branch_map:
                    parts[1] = branch_map[parts[1]]
                    new_k = '.'.join(parts)
        new_state_dict[new_k] = v

    # 2. Auto-fill Aliases
    final_state_dict = new_state_dict.copy()
    for k in list(new_state_dict.keys()):
        alias_k = 'end_points.' + k
        if alias_k in model_keys and alias_k not in final_state_dict:
            final_state_dict[alias_k] = new_state_dict[k]
            
    # 3. Load
    model.load_state_dict(final_state_dict, strict=False)
    return model

def load_i3d_model(device):
    if InceptionI3d is None:
        return None

    try:
        i3d = InceptionI3d(400, in_channels=3).to(device)
    except Exception as e:
        print(f"Error initializing InceptionI3d: {e}")
        return None

    paths = [
        'i3d_pretrained_400.pt',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_pretrained_400.pt'),
        '/nvme2/chenggf/fangxiaocheng/ECG-CMR/MAR/i3d_pretrained_400.pt',
        os.path.join('/nvme2/chenggf/fangxiaocheng/VideoGPT', 'i3d_pretrained_400.pt'),
        os.path.expanduser('~/.cache/videogpt/i3d_pretrained_400.pt')
    ]
    weights_path = None
    for p in paths:
        if os.path.exists(p):
            weights_path = p
            break
            
    if weights_path is None:
        print("Warning: 'i3d_pretrained_400.pt' not found. FVD will be skipped.")
        return None
        
    print(f"Loading I3D weights from: {weights_path}")
    raw_state_dict = torch.load(weights_path, map_location=device)
    i3d = smart_load_state_dict(i3d, raw_state_dict)
    i3d.eval()
    return i3d

def preprocess_i3d(videos, target_resolution=224):
    """
    Input: (B, C, T, H, W) range [0, 1]
    Output: (B, C, T, 224, 224) range [-1, 1]
    """
    b, c, t, h, w = videos.shape
    videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    videos_flat = F.interpolate(videos_flat, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)
    videos = videos_flat.reshape(b, t, c, target_resolution, target_resolution).permute(0, 2, 1, 3, 4)
    videos = (videos - 0.5) * 2.0
    return videos

def get_i3d_features(i3d, videos, device, bs=16):
    n = videos.shape[0]
    features = []
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = videos[i : i + bs].to(device)
            batch = preprocess_i3d(batch)
            if hasattr(i3d, 'extract_features'):
                out = i3d.extract_features(batch)
            else:
                out = i3d(batch) 
            features.append(out.cpu().numpy())
    return np.concatenate(features, axis=0)

# -------------------------------------------------------------------------
# Metrics Calculation
# -------------------------------------------------------------------------
def reduce_video_metrics(gt: np.ndarray, pred: np.ndarray, metrics_bs: int = 8):
    """
    Calculate LPIPS, FID, FVD.
    Input: (N, C, T, H, W), [0, 1]
    """
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for metrics calculation")

    n, c, t, h, w = gt.shape
    
    # === 1. LPIPS ===
    # Flatten: (N, C, T, H, W) -> (N, T, C, H, W) -> (N*T, C, H, W)
    gt_flat = gt.transpose(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    pred_flat = pred.transpose(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    total_frames = n * t

    lpips_vals = []
    print(f"Calculating LPIPS (Total frames: {total_frames})...")
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device).eval()

    for i in tqdm(range(0, total_frames, metrics_bs), desc='LPIPS'):
        j = min(i + metrics_bs, total_frames)
        
        gt_b = torch.from_numpy(gt_flat[i:j]).to(device)
        pred_b = torch.from_numpy(pred_flat[i:j]).to(device)

        # LPIPS needs [-1, 1] and 3 channels
        p_lp = pred_b * 2.0 - 1.0
        g_lp = gt_b * 2.0 - 1.0
        if p_lp.shape[1] == 1: p_lp = p_lp.repeat(1, 3, 1, 1)
        if g_lp.shape[1] == 1: g_lp = g_lp.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            lp_val = loss_fn_lpips(p_lp, g_lp)
            lpips_vals.extend(lp_val.cpu().view(-1).numpy().tolist())

    avg_lpips = np.mean(lpips_vals)

    # === 2. FID (2D) ===
    print("Calculating FID (Frame-wise)...")
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()

    mean_2d = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_2d = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    feats_fid_gt = []
    feats_fid_pred = []

    for i in tqdm(range(0, total_frames, metrics_bs), desc='Extracting FID (2D)'):
        j = min(i + metrics_bs, total_frames)
        gt_b = torch.from_numpy(gt_flat[i:j]).to(device)
        pred_b = torch.from_numpy(pred_flat[i:j]).to(device)
        
        # Expand channels & Resize
        if gt_b.shape[1] == 1: gt_b = gt_b.repeat(1, 3, 1, 1)
        if pred_b.shape[1] == 1: pred_b = pred_b.repeat(1, 3, 1, 1)
        
        gt_b = F.interpolate(gt_b, size=(299, 299), mode='bilinear', align_corners=False)
        pred_b = F.interpolate(pred_b, size=(299, 299), mode='bilinear', align_corners=False)
        
        gt_b = (gt_b - mean_2d) / std_2d
        pred_b = (pred_b - mean_2d) / std_2d
        
        with torch.no_grad():
            feats_fid_gt.append(inception(gt_b).cpu().numpy())
            feats_fid_pred.append(inception(pred_b).cpu().numpy())

    all_fid_gt = np.concatenate(feats_fid_gt, axis=0)
    all_fid_pred = np.concatenate(feats_fid_pred, axis=0)
    mu_g, cov_g = np.mean(all_fid_gt, axis=0), np.cov(all_fid_gt, rowvar=False)
    mu_p, cov_p = np.mean(all_fid_pred, axis=0), np.cov(all_fid_pred, rowvar=False)
    fid_score = _frechet_distance(mu_g, cov_g, mu_p, cov_p)

    # === 3. FVD (I3D) ===
    fvd_score = None
    i3d = load_i3d_model(device)
    
    if i3d is not None:
         print("\n=== Calculating FVD (I3D Standard) ===")
         gt_expanded = gt
         pred_expanded = pred
         if gt.shape[1] == 1:
             gt_expanded = np.repeat(gt, 3, axis=1)
         if pred.shape[1] == 1:
             pred_expanded = np.repeat(pred, 3, axis=1)
             
         gt_tensor = torch.from_numpy(gt_expanded)
         pred_tensor = torch.from_numpy(pred_expanded)
         
         fvd_bs = max(1, metrics_bs // 2)
         
         print(f"Extracting GT I3D features (N={len(gt_expanded)})...")
         feats_fvd_gt = get_i3d_features(i3d, gt_tensor, device, bs=fvd_bs)
         
         print(f"Extracting Pred I3D features (N={len(pred_expanded)})...")
         feats_fvd_pred = get_i3d_features(i3d, pred_tensor, device, bs=fvd_bs)
         
         print("Computing FVD statistics...")
         mu_gv, cov_gv = np.mean(feats_fvd_gt, axis=0), np.cov(feats_fvd_gt, rowvar=False)
         mu_pv, cov_pv = np.mean(feats_fvd_pred, axis=0), np.cov(feats_fvd_pred, rowvar=False)
         
         fvd_score = _frechet_distance(mu_gv, cov_gv, mu_pv, cov_pv)

    return {
        'lpips': float(avg_lpips),
        'fid': float(fid_score),
        'fvd': float(fvd_score) if fvd_score is not None else None,
    }

def main():
    parser = argparse.ArgumentParser('Evaluate NPY files (GT vs Fake)')
    parser.add_argument('--gt', default='/nvme2/chenggf/fangxiaocheng/ECG-CMR/MAR/results/flow_e2c/ukbb/samples/overall_cmr_gt_data.npy', help='GT .npy file (N,C,T,H,W)')
    parser.add_argument('--pred', default='/nvme2/chenggf/fangxiaocheng/ECG-CMR/MAR/results/flow_e2c/ukbb/samples/overall_cmr_fake_data.npy', help='Pred .npy file (N,C,T,H,W)')
    parser.add_argument('--metrics_bs', default=16, type=int, help='Batch size for metrics')
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"GT not found: {args.gt}")
    if not os.path.exists(args.pred):
        raise FileNotFoundError(f"Pred not found: {args.pred}")

    print(f"Loading GT: {args.gt}")
    gt = np.load(args.gt)
    print(f"Loading Pred: {args.pred}")
    pred = np.load(args.pred)

    print(f"GT range: [{gt.min():.4f}, {gt.max():.4f}]")
    print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    
    if gt.max() > 1.1 or pred.max() > 1.1:
        print("Warning: Data seems to be in [0, 255] or other range. Normalizing to [0, 1]...")
        gt = gt / 255.0
        pred = pred / 255.0
        
    if gt.min() < -0.1 or pred.min() < -0.1:
        print("Warning: Data contains negative values. Metrics assume [0, 1] input.")

    if gt.ndim != 5 or pred.ndim != 5:
        raise ValueError("Input data must be 5D: (N, C, T, H, W)")

    results = reduce_video_metrics(gt, pred, metrics_bs=args.metrics_bs)
    
    print("\n====== Evaluation Results ======")
    print(json.dumps(results, indent=2))
    print("================================")

if __name__ == '__main__':
    main()
