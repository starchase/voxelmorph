import os
import sys
import argparse
import glob
import time
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# 需要确保根目录在 PYTHONPATH 中，以便导入 data 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import voxelmorph as vxm
from data import datasets, trans
import utils

def fast_hd95(mask_A, mask_B, voxelspacing=1.0):
    """
    高效计算 HD95 的方法，避免 medpy 的边界开销和一些意外奔溃
    """
    # 提取边界
    border_A = mask_A ^ (distance_transform_edt(mask_A) > 1)
    border_B = mask_B ^ (distance_transform_edt(mask_B) > 1)
    
    if not np.any(border_A) or not np.any(border_B):
        return 0.0
        
    dist_A = distance_transform_edt(~border_A)
    dist_B = distance_transform_edt(~border_B)
    
    distances_A_to_B = dist_B[border_A]
    distances_B_to_A = dist_A[border_B]
    
    return max(np.percentile(distances_A_to_B, 95), np.percentile(distances_B_to_A, 95)) * voxelspacing

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/models/oasis_vxm_20260312_180326/oasis_vxm_best.pt', help='Pytorch model to evaluate')
    parser.add_argument('--val-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/', help='Validation/Test directory')
    parser.add_argument('--pairs-csv', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/pairs_val.csv', help='CSV file defining image pairs')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--max-pairs', type=int, default=1, help='仅计算指定数量的配对以供快速测试(默认1对，设为0则计算全部)')
    parser.add_argument('--out-csv', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/pair_metrics_results.csv', help='保存每对计算结果的CSV路径')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 1. 结构完全一致的模型实例化
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[
            [32, 64, 64, 64], 
            [64, 64, 64, 32]
        ],
        integration_steps=0,
    ).to(device)

    # 加载权重
    print(f"Loading weights from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Nearest Neighbor Transformer 用于处理分割标签 Label
    reg_model = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)

    # 2. 从 CSV 读取配对并准备文件列表
    print(f"Reading pairs from: {args.pairs_csv}")
    pkl_files = []
    with open(args.pairs_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header (fixed, moving)
        for row in reader:
            if not row: continue
            fixed_id = row[0].strip()
            moving_id = row[1].strip()
            
            # OASIS L2R 2021 的 pkl 命名通常是 p_fixed_moving.pkl
            # 但要注意：在你的目录下文件名为 p_0438_0439.pkl
            # 补齐 4 位 0
            fixed_id = fixed_id.zfill(4)
            moving_id = moving_id.zfill(4)
            
            pkl_name = f"p_{fixed_id}_{moving_id}.pkl"
            pkl_path = os.path.join(args.val_dir, pkl_name)
            
            if os.path.exists(pkl_path):
                pkl_files.append((pkl_path, fixed_id, moving_id))
            else:
                print(f"Warning: Could not find dataset file {pkl_path}")

    # 解开元组用于送给 Dataset
    dataset_paths = [p[0] for p in pkl_files]

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_set = datasets.OASISBrainInferDataset(dataset_paths, transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    print(f"Total test paired samples found: {len(val_loader)}")

    # 3. 统计变量
    all_dsc = []
    all_hd95 = []
    all_neg_jac = []
    all_sdlogj = []
    inference_times = []
    pair_results = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader, desc="Evaluating")):
            x = data[0].to(device)
            y = data[1].to(device)
            x_seg = data[2].to(device) # source label
            y_seg = data[3].to(device) # target label

            # GPU Warmup 对第一张图，保证测时准确
            if idx == 0:
                for _ in range(3):
                    _ = model(x, y)
                torch.cuda.synchronize()

            # --------- 测推理时间开始 ---------
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 当 model 处于 eval() 模式且只要求 return_warped_source=True 时，
            # VxmPairwise 返回的 out 格式是：(位移场 displacement, 形变后的原图 warped_source)
            out = model(x, y, return_field_type='displacement', return_warped_source=True)
            displacement = out[0]  # out[0] 是形状为 [B, 3, D, H, W] 的向量场
            warped_source = out[1] # out[1] 是形状为 [B, 1, D, H, W] 的形变图像

            torch.cuda.synchronize()
            infer_time = time.time() - start_time
            if idx > 0: # 排除掉第一张可能受缓存影响的图
                inference_times.append(infer_time)
            # --------- 测推理时间结束 ---------

            # 4. 指标计算

            # 使用 Nearest Neighbor 将 Moving Label 变形到 Target 空间进行比对
            def_out = reg_model(x_seg.float(), displacement).long()
            
            def_out_np = def_out.cpu().numpy()[0, 0]
            y_seg_np = y_seg.cpu().numpy()[0, 0]

            # 获取图像中所有独有的解剖结构标签，排除背景 (0)
            unique_labels = np.unique(y_seg_np)
            unique_labels = unique_labels[unique_labels > 0] 

            sample_dsc = []
            sample_hd95 = []
            
            for label in unique_labels:
                mask_def = (def_out_np == label)
                mask_y = (y_seg_np == label)
                
                # ------ Dice ------
                intersect = np.sum(mask_def & mask_y)
                sum_areas = np.sum(mask_def) + np.sum(mask_y)
                if sum_areas > 0:
                    sample_dsc.append((2.0 * intersect) / sum_areas)
                else:
                    sample_dsc.append(0.0)
                    
                # ------ HD95 ------
                if np.sum(mask_def) > 0 and np.sum(mask_y) > 0:
                    try:
                        h = fast_hd95(mask_def, mask_y)
                        sample_hd95.append(h)
                    except Exception as e:
                        pass # 有时某类的点过少可能报错，直接跳过此类
                        
            if sample_dsc:
                all_dsc.append(np.mean(sample_dsc))
            if sample_hd95:
                all_hd95.append(np.mean(sample_hd95))

            # ------ 负雅可比行列式 (% ||J|| <= 0) ------
            # displacement shape: [1, 3, W, H, D]
            disp_np = displacement.cpu().numpy()[0]
            
            # 使用np.gradient计算相对于z, y, x的偏导
            dz_dz, dz_dy, dz_dx = np.gradient(disp_np[0])
            dy_dz, dy_dy, dy_dx = np.gradient(disp_np[1])
            dx_dz, dx_dy, dx_dx = np.gradient(disp_np[2])
            
            # 公式: T(x) = x + u(x)，由于位移场为 u，形变场为 id + u
            jac_det = ( (1 + dx_dx) * ((1 + dy_dy) * (1 + dz_dz) - dy_dz * dz_dy)
                      - dx_dy * (dy_dx * (1 + dz_dz) - dy_dz * dz_dx)
                      + dx_dz * (dy_dx * dz_dy - (1 + dy_dy) * dz_dx) )
                      
            # 1. 负雅可比占比 (%)
            neg_jac_ratio = np.sum(jac_det <= 0) / jac_det.size
            all_neg_jac.append(neg_jac_ratio)
            
            # 2. SDlogJ (Standard Deviation of the logarithm of the Jacobian determinant)
            # 为了防止对负数或者0取对数报错，我们只对 >0 的雅可比部分计算 SDlogJ，或者加上极小值 eps
            valid_jac = jac_det[jac_det > 0]
            if len(valid_jac) > 0:
                sdlogj = np.std(np.log(valid_jac))
                all_sdlogj.append(sdlogj)
            else:
                sdlogj = np.nan
            
            # 从之前存的 pkl_files 里提取 ID 来打印日志
            _, f_id, m_id = pkl_files[idx]
            
            cur_dsc = np.mean(sample_dsc) if sample_dsc else 0
            cur_hd95 = np.mean(sample_hd95) if sample_hd95 else 0
            pair_results.append({
                'Fixed': f_id,
                'Moving': m_id,
                'DSC': cur_dsc,
                'HD95': cur_hd95,
                'Neg_Jac_Ratio(%)': neg_jac_ratio * 100,
                'SDlogJ': sdlogj,
                'Time(s)': infer_time
            })
            
            print(f"\n[Pair {f_id} -> {m_id}] - DSC: {cur_dsc:.4f} | HD95: {cur_hd95:.4f} mm | |J|<=0: {neg_jac_ratio*100:.4f}% | SDlogJ: {sdlogj:.4f} | Time: {infer_time:.4f}s")
            
            if args.max_pairs > 0 and (idx + 1) >= args.max_pairs:
                print(f"\n已达到设置的测试对数上限 ({args.max_pairs})，停止评估。")
                break

    # ================= 打印最终结果 =================
    avg_inference_time = np.mean(inference_times) if len(inference_times) > 0 else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    print("\n" + "="*50)
    print("                FINAL RESULTS               ")
    print("="*50)
    print(f"Model Evaluated: {args.model}")
    print(f"Number of Subjects: {len(all_dsc)}")
    print(f"Average DSC (Dice):              {np.mean(all_dsc):.4f}  ± {np.std(all_dsc):.4f}")
    if len(all_hd95)>0:
       print(f"Average HD95:                    {np.mean(all_hd95):.4f} mm ± {np.std(all_hd95):.4f} mm")
    print(f"Average % |Jac| <= 0 voxels:     {np.mean(all_neg_jac)*100:.4f} %  ± {np.std(all_neg_jac)*100:.4f} %")
    if len(all_sdlogj)>0:
       print(f"Average SDlogJ:                  {np.mean(all_sdlogj):.4f}  ± {np.std(all_sdlogj):.4f}")
    print(f"Average Inference Time/Volume:   {avg_inference_time:.4f} seconds ({fps:.2f} FPS)")
    print("="*50)

    if pair_results:
        print(f"\nSaving detailed pair results to {args.out_csv}...")
        keys = pair_results[0].keys()
        with open(args.out_csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(pair_results)
        print("Save completed!")

if __name__ == '__main__':
    test()