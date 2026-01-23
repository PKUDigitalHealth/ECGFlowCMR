import os
import glob
import lmdb
import torch
import pickle
import numpy as np
import io
import re
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class UKBBLMDBDataset(Dataset):
    """
    UK Biobank 数据集加载器
    加载包含 ECG (lead_data) 和 CMR (cmr_data) 的 LMDB 文件
    """
    def __init__(self, root_dir: str = '', split: str = ''):
        self.root_dir = root_dir
        self.split = split

        # 若 root_dir/split 存在则使用该子目录；否则当作直接包含 *.lmdb 的目录使用
        dataset_dir_candidate = os.path.join(root_dir, split) if isinstance(split, str) else root_dir
        if os.path.isdir(dataset_dir_candidate) and len(glob.glob(os.path.join(dataset_dir_candidate, '*.lmdb'))) > 0:
            dataset_dir = dataset_dir_candidate
        else:
            dataset_dir = root_dir

        self.all_subject_data_dirs = glob.glob(os.path.join(dataset_dir, '*.lmdb'))

    def __getitem__(self, index):
        eids, lead_data, cmr_data = self.load_lmdb_data(self.all_subject_data_dirs[index])
        # 将 CMR 按样本逐个归一化到 [-1, 1] 并转换为 float32
        cmr_data = cmr_data.astype(np.float32)
        # 逐样本（按首维 N）计算 min/max，兼容任意维度的后续轴
        reduce_axes = tuple(range(1, cmr_data.ndim))
        vmin = cmr_data.min(axis=reduce_axes, keepdims=True)
        vmax = cmr_data.max(axis=reduce_axes, keepdims=True)
        denom = np.maximum(vmax - vmin, 1e-6)
        cmr_data = 2.0 * (cmr_data - vmin) / denom - 1.0
        return (
            eids, 
            torch.tensor(lead_data, dtype=torch.float32),
            torch.tensor(cmr_data, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.all_subject_data_dirs)

    @staticmethod
    def load_lmdb_data(lmdb_path: str, max_readers: int = 16):
        """
        从 LMDB 文件加载数据
        
        返回:
            eids: numpy array, 样本ID列表
            lead_data: numpy array, ECG数据 (N, 12, 5000)
            cmr_data: numpy array, CMR数据
        """
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            max_readers=max_readers,
        )
        
        with env.begin() as txn:
            # 遍历键名对齐 'eid*'、'lead*' 与 'cmr*'
            cursor = txn.cursor()
            id_to_eid = {}
            id_to_lead = {}
            id_to_cmr = {}
            
            for key, value in cursor:
                key_str = key.decode('utf-8', errors='ignore') if isinstance(key, bytes) else str(key)
                lowered = key_str.lower()
                ident_digits = re.findall(r'\d+', key_str)
                ident = ident_digits[-1] if len(ident_digits) > 0 else key_str
                
                if 'cmr' in lowered:
                    try:
                        # 解码 CMR 数组
                        try:
                            bio = io.BytesIO(value)
                            obj = np.load(bio, allow_pickle=True)
                            arr = obj['arr_0'] if hasattr(obj, 'files') else obj
                        except Exception:
                            try:
                                arr = pickle.loads(value)
                            except Exception:
                                arr = value
                        id_to_cmr[ident] = np.array(arr)
                    except Exception:
                        pass
                
                elif 'lead' in lowered:
                    try:
                        # 解码 ECG lead 数组
                        try:
                            bio = io.BytesIO(value)
                            obj = np.load(bio, allow_pickle=True)
                            arr = obj['arr_0'] if hasattr(obj, 'files') else obj
                        except Exception:
                            try:
                                arr = pickle.loads(value)
                            except Exception:
                                arr = value
                        id_to_lead[ident] = np.array(arr)
                    except Exception:
                        pass
                
                elif 'eid' in lowered:
                    # 解码 eid 标量或字符串
                    try:
                        bio = io.BytesIO(value)
                        obj = np.load(bio, allow_pickle=True)
                        v = obj['arr_0'] if hasattr(obj, 'files') else obj
                    except Exception:
                        try:
                            v = pickle.loads(value)
                        except Exception:
                            v = value
                    
                    # 处理不同类型的eid
                    if isinstance(v, bytes):
                        id_to_eid[ident] = v.decode('utf-8')
                    elif np.isscalar(v) or isinstance(v, (int, np.integer, float, str)):
                        id_to_eid[ident] = v
                    else:
                        try:
                            arr = np.array(v)
                            if arr.ndim == 0:
                                id_to_eid[ident] = arr.item()
                        except Exception:
                            pass

            # 找到共同的ID并对齐数据
            common_ids = [k for k in id_to_eid.keys() if k in id_to_lead and k in id_to_cmr]
            
            if len(common_ids) > 0:
                try:
                    common_ids_sorted = sorted(common_ids, key=lambda x: int(x))
                except Exception:
                    common_ids_sorted = sorted(common_ids)
                
                eids = np.array([id_to_eid[k] for k in common_ids_sorted])
                lead_list = [id_to_lead[k] for k in common_ids_sorted]
                cmr_list = [id_to_cmr[k] for k in common_ids_sorted]
                
                try:
                    lead_data = np.stack(lead_list, axis=0)
                except Exception:
                    lead_data = np.array(lead_list, dtype=object)
                
                try:
                    cmr_data = np.stack(cmr_list, axis=0)
                except Exception:
                    cmr_data = np.array(cmr_list, dtype=object)
                
                return eids, lead_data, cmr_data

            # 未找到期望键时给出明确提示
            raise RuntimeError(
                f"Missing keys in {lmdb_path}. "
                f"Found keys: eid={len(id_to_eid)}, lead={len(id_to_lead)}, cmr={len(id_to_cmr)}"
            )

    @staticmethod
    def collate_fn(batch):
        """
        合并批次数据
        
        参数:
            batch: list of (eids, lead_data, cmr_data)
        
        返回:
            eids: list, 所有样本的ID
            lead_data: torch.Tensor, ECG数据 (batch_size, 12, 5000)
            cmr_data: torch.Tensor, CMR数据
        """
        # 合并 eids
        eids_list = []
        for item in batch:
            eids_item = item[0]
            if isinstance(eids_item, np.ndarray):
                eids_list.append(eids_item)
            else:
                eids_list.append(np.array(eids_item))
        
        if len(eids_list) > 0:
            eids = np.concatenate(eids_list, axis=0).tolist()
        else:
            eids = []

        # 合并 lead_data (ECG)
        lead_tensors = [item[1] for item in batch]
        lead_data = torch.cat(lead_tensors, dim=0)
        
        # 合并 cmr_data
        cmr_tensors = [item[2] for item in batch]
        cmr_data = torch.cat(cmr_tensors, dim=0)
        
        return eids, lead_data, cmr_data


if __name__ == "__main__":
    # 测试数据集加载
    dataset = UKBBLMDBDataset()
    print(f"数据集中共有 {len(dataset)} 个 LMDB 文件")
    
    if len(dataset) == 0:
        print("警告：未找到 LMDB 文件，请检查数据集路径")
    else:
        # 统计所有样本数量总和
        try:
            total_samples = 0
            for i in range(len(dataset)):
                eids, _, _ = dataset[i]
                total_samples += len(eids)
            print(f"\n所有样本数量总和: {total_samples}")
        except Exception as e:
            print(f"\n统计样本总数时出错: {e}")

        # 测试单个样本加载
        print("\n测试单个 LMDB 文件加载：")
        eids, lead_data, cmr_data = dataset[0]
        print(f"  - EIDs 数量: {len(eids)}")
        print(f"  - ECG (lead_data) 形状: {tuple(lead_data.shape)}")
        print(f"  - CMR (cmr_data) 形状: {tuple(cmr_data.shape)}")
        print(f"  - ECG 数据类型: {lead_data.dtype}")
        print(f"  - CMR 数据类型: {cmr_data.dtype}")
        
        # 测试 DataLoader
        print("\n测试 DataLoader 批量加载：")
        loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=UKBBLMDBDataset.collate_fn,
            num_workers=0  # 设置为0避免多进程问题
        )
        
        for batch_idx, (eids, lead_data, cmr_data) in enumerate(loader):
            print(f"Batch {batch_idx}:")
            print(f"  - EIDs 数量: {len(eids)}")
            print(f"  - ECG (lead_data) 形状: {tuple(lead_data.shape)}")
            print(f"  - CMR (cmr_data) 形状: {tuple(cmr_data.shape)}")
            
            if batch_idx >= 1:  # 只测试前两个批次
                break
        
        print("\n数据集加载测试完成！")

