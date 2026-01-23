import argparse
import os
from typing import List, Tuple

import imageio
import numpy as np
import cv2
from tqdm import tqdm


def normalize_to_3d(data: np.ndarray) -> np.ndarray:
    """将数据规范化为 (T, H, W) 形状。
    
    输入形状：
    - (C, T, H, W) -> (T, H, W) 取单通道或均值
    - (T, H, W) -> (T, H, W) 保持不变
    """
    if data.ndim == 4:
        if data.shape[0] > 1:
            data = data.mean(axis=0)
        else:
            data = data[0]
    assert data.ndim == 3, f'Expected 3D after normalization, got {data.ndim}D'
    return data


def make_side_by_side_gif(gt: np.ndarray, fake: np.ndarray, save_path: str, fps: int = 20) -> None:
    """将单个样本的 GT 与 Fake 在宽度方向拼接并保存为 GIF。

    输入形状：
    - gt: (C, T, H, W) 或 (1, T, H, W) 或 (T, H, W)
    - fake: 同 gt
    要求数据范围在 [0,1]。
    """
    gt = normalize_to_3d(gt)
    fake = normalize_to_3d(fake)
    assert gt.shape == fake.shape, f'shape mismatch: {gt.shape} vs {fake.shape}'

    t, h, w = gt.shape
    frames = []
    for i in range(t):
        g = np.clip(gt[i], 0.0, 1.0)
        f = np.clip(fake[i], 0.0, 1.0)
        pair = np.concatenate([g, f], axis=1)  # (H, 2W)
        pair = (pair * 255.0).astype(np.uint8)
        frames.append(pair)

    imageio.mimsave(save_path, frames, format='GIF', fps=fps)


def make_multi_model_gif(data_list: List[np.ndarray], save_path: str, 
                         fps: int = 20, layout: Tuple[int, int] = (2, 4), padding: int = 10,
                         labels: List[str] = None) -> None:
    """将多个模型的视频数据组合成一个 GIF/MP4，按照指定布局排列。
    
    参数：
    - data_list: 包含多个视频数据的列表，每个元素形状为 (C, T, H, W) 或 (T, H, W)
    - save_path: 保存路径
    - fps: 帧率
    - layout: (rows, cols) 布局，例如 (2, 4) 表示 2 行 4 列
    - padding: 视频之间的间距（像素）
    - labels: 每个视频对应的标签文本
    要求数据范围在 [0,1]。
    """
    rows, cols = layout
    num_videos = len(data_list)
    assert num_videos <= rows * cols, f'视频数量 {num_videos} 超过布局容量 {rows * cols}'
    
    if labels is None:
        labels = [''] * num_videos
    assert len(labels) == num_videos, 'Labels count must match data_list count'
    
    # 规范化所有数据到 (T, H, W)
    normalized_data = [normalize_to_3d(data) for data in data_list]
    
    # 检查所有视频的时间长度和空间尺寸
    t_list = [data.shape[0] for data in normalized_data]
    h_list = [data.shape[1] for data in normalized_data]
    w_list = [data.shape[2] for data in normalized_data]
    
    # 使用最小时间长度（确保所有视频同步）
    t = min(t_list)
    # 使用统一的空间尺寸（取最大值，然后裁剪或填充）
    h = max(h_list)
    w = max(w_list)
    
    # 准备所有视频帧
    all_frames = []
    for data in normalized_data:
        frames = []
        for i in range(t):
            frame = np.clip(data[i], 0.0, 1.0)
            # 如果尺寸不匹配，进行裁剪或填充
            if frame.shape[0] != h or frame.shape[1] != w:
                # 使用填充到目标尺寸
                padded = np.zeros((h, w), dtype=frame.dtype)
                h_start = (h - frame.shape[0]) // 2
                w_start = (w - frame.shape[1]) // 2
                h_end = h_start + frame.shape[0]
                w_end = w_start + frame.shape[1]
                padded[h_start:h_end, w_start:w_end] = frame
                frame = padded
            frames.append(frame)
        all_frames.append(frames)
    
    # 辅助函数：添加白色边框（间距）
    def add_padding(frame: np.ndarray, pad: int) -> np.ndarray:
        """在帧周围添加白色边框作为间距"""
        h_frame, w_frame = frame.shape
        padded = np.ones((h_frame + 2 * pad, w_frame + 2 * pad), dtype=np.uint8) * 255
        padded[pad:h_frame + pad, pad:w_frame + pad] = frame
        return padded

    # 辅助函数：添加底部文字
    def add_caption(frame: np.ndarray, text: str) -> np.ndarray:
        if not text:
            return frame
        h_frame, w_frame = frame.shape
        caption_height = 40
        padded = np.ones((h_frame + caption_height, w_frame), dtype=np.uint8) * 255
        padded[:h_frame, :] = frame
        
        # 绘制文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (fw, fh), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w_frame - fw) // 2
        y = h_frame + (caption_height + fh) // 2 - 8
        
        cv2.putText(padded, text, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)
        return padded
    
    # 组合成布局
    combined_frames = []
    # 预计算每个格子的大小（假设所有frame经过padding和caption后大小一致）
    # 先做一次处理来获取尺寸
    sample_frame = (all_frames[0][0] * 255.0).astype(np.uint8)
    sample_frame = add_padding(sample_frame, padding)
    sample_frame = add_caption(sample_frame, "Test")
    cell_h, cell_w = sample_frame.shape
    
    for i in range(t):
        row_images = []
        current_idx = 0
        
        for row in range(rows):
            col_images = []
            for col in range(cols):
                if current_idx < num_videos:
                    frame = all_frames[current_idx][i]
                    frame = (frame * 255.0).astype(np.uint8)
                    # 添加白色边框作为间距
                    frame = add_padding(frame, padding)
                    # 添加文字
                    frame = add_caption(frame, labels[current_idx])
                    col_images.append(frame)
                    current_idx += 1
                else:
                    # 空白占位（白色背景）
                    blank = np.ones((cell_h, cell_w), dtype=np.uint8) * 255
                    col_images.append(blank)
            
            # 水平拼接一行
            row_img = np.concatenate(col_images, axis=1)  # (H, cols*W)
            row_images.append(row_img)
        
        # 垂直拼接所有行
        combined = np.concatenate(row_images, axis=0)  # (rows*H, cols*W)
        combined_frames.append(combined)
    
    imageio.mimsave(save_path, combined_frames, fps=fps)



def main():
    parser = argparse.ArgumentParser('Make multi-model GIFs with GT and generated samples')
    parser.add_argument('--gt', default='', type=str,
                        help='Ground truth npy file path')
    default_models = [

    ]
    parser.add_argument('--models', nargs='*', type=str, default=None,
                        help='Paths to model-generated npy files (default: 5 predefined model files)')
    parser.add_argument('--out_dir', default='', type=str)
    parser.add_argument('--num', default=32, type=int, help='要导出的样本数')
    parser.add_argument('--fps', default=20, type=int)
    parser.add_argument('--layout', nargs=2, type=int, default=[1, 6],
                        help='Layout: [rows, cols], default [1, 6]')
    parser.add_argument('--padding', default=10, type=int,
                        help='Padding between videos in pixels (default: 10)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 如果没有提供模型路径，使用默认值
    if args.models is None or len(args.models) == 0:
        args.models = default_models
        print(f'Using default model paths: {len(args.models)} models')
    
    # 标签列表
    labels = ['GT', 'VideoGPT', 'ModelScope', 'CardioNets', 'EchoPulse', 'Ours']
    # 如果用户提供了自定义模型，标签可能不匹配，这里简单处理：只在使用默认模型时使用预定义标签
    # 如果模型数量变化，需要调整标签
    if len(args.models) != 5:
        print("Warning: Model count is not 5, labels might be incorrect.")
        # 简单生成标签
        labels = ['GT'] + [f'Model {i+1}' for i in range(len(args.models))]
    
    # 加载 GT 数据
    print(f'Loading GT from {args.gt}...')
    gt = np.load(args.gt)   # (N, C, T, H, W) 或 (N, T, H, W)
    
    # 加载所有模型数据
    model_data_list = []
    for model_path in args.models:
        print(f'Loading model data from {model_path}...')
        model_data = np.load(model_path)
        model_data_list.append(model_data)
    
    # 检查形状一致性
    assert gt.ndim in (4, 5), f'Unsupported GT ndim: {gt.ndim}, expected 4 or 5'
    for i, model_data in enumerate(model_data_list):
        assert model_data.ndim in (4, 5), f'Unsupported model {i} ndim: {model_data.ndim}, expected 4 or 5'
        if gt.ndim == 5:
            assert model_data.ndim == 5, f'Model {i} ndim mismatch with GT'
            assert gt.shape[0] == model_data.shape[0], f'Model {i} sample count mismatch: {gt.shape[0]} vs {model_data.shape[0]}'
        else:
            assert model_data.ndim == 4, f'Model {i} ndim mismatch with GT'
            assert gt.shape[0] == model_data.shape[0], f'Model {i} sample count mismatch: {gt.shape[0]} vs {model_data.shape[0]}'
    
    # 确定样本数量
    if gt.ndim == 5:
        n = gt.shape[0]
    else:
        n = gt.shape[0]
    
    num = min(args.num, n)
    indices = np.random.choice(n, size=num, replace=False)
    
    rows, cols = args.layout
    # 确保布局能容纳所有视频（GT + 模型）
    total_videos = 1 + len(args.models)  # GT + 模型数量
    assert total_videos <= rows * cols, f'视频数量 {total_videos} 超过布局容量 {rows * cols}'
    
    # 生成 GIF/MP4
    for i in tqdm(indices, desc='Saving Videos', ncols=100):
        # 提取单个样本的所有数据
        data_list = []
        
        # GT
        if gt.ndim == 5:
            gt_i = gt[i]    # (C, T, H, W)
        else:
            gt_i = gt[i]    # (T, H, W)
            gt_i = gt_i[np.newaxis, ...]  # 添加通道维度 -> (1, T, H, W)
        data_list.append(gt_i)
        
        # 模型数据
        for model_data in model_data_list:
            if model_data.ndim == 5:
                model_i = model_data[i]    # (C, T, H, W)
            else:
                model_i = model_data[i]    # (T, H, W)
                model_i = model_i[np.newaxis, ...]  # 添加通道维度 -> (1, T, H, W)
            data_list.append(model_i)
        
        save_path = os.path.join(args.out_dir, f'{i + 1}.mp4')
        make_multi_model_gif(data_list, save_path, fps=args.fps, layout=(rows, cols), padding=args.padding, labels=labels)

    print(f'Saved {num} videos to {args.out_dir}')


if __name__ == '__main__':
    main()