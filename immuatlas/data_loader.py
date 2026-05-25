"""
PyTorch Dataset classes for peptide-HLA pairs.

* ``PeptideHLADataset2`` – lightweight dataset for **inference** (no labels).
* ``PeptideHLADataset``  – dataset for **training** (reads TSV with labels).
"""

from typing import List

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class PeptideHLADataset(Dataset):
    """
    肽段-HLA数据集类，支持长度8-25的肽段（用于训练）
    """

    def __init__(self, data_file: str, hla_seq_dict: dict):
        """
        Parameters
        ----------
        data_file : str
            TSV 文件路径，需包含 peptide / HLA / label 列
        hla_seq_dict : dict
            HLA 等位基因 → 伪序列 的映射字典
        """
        self.data = pd.read_csv(data_file, sep='\t')
        self.hla_seq_dict = hla_seq_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        peptide = row['peptide']
        hla = row['HLA']
        hla_seq = self.hla_seq_dict.get(hla, '')
        eluted_ligand_label = int(row['label'])
        return {
            'peptide': peptide,
            'hla': hla,
            'hla_seq': hla_seq,
            'eluted_ligand': eluted_ligand_label,
        }


def PeptideHLADataset2(peptides: List[str], hla_seqs: List[str]):
    """
    创建用于预测的数据集（无标签）。

    Parameters
    ----------
    peptides : list[str]
        肽段序列列表
    hla_seqs : list[str]
        与 peptides 等长的 HLA 伪序列列表

    Returns
    -------
    Dataset
        每个元素为 ``(peptide_str, hla_seq_str)`` 元组
    """

    class _PeptideHLADataset2(Dataset):
        def __init__(self, peptides, hla_seqs):
            if len(peptides) != len(hla_seqs):
                raise ValueError("Peptides and HLA sequences must have the same length")
            self.peptides = peptides
            self.hla_seqs = hla_seqs

        def __len__(self):
            return len(self.peptides)

        def __getitem__(self, idx):
            return self.peptides[idx], self.hla_seqs[idx]

    return _PeptideHLADataset2(peptides, hla_seqs)


# ---------------------------------------------------------------------------
# Training utilities (optional – only needed for model training)
# ---------------------------------------------------------------------------

def create_data_loaders(train_file, test_file, hla_seq_dict, batch_size=64, val_split=0.1):
    """基于训练集（默认90%训练与10%验证）与测试集文件，创建训练、验证和测试数据加载器"""
    full_train_dataset = PeptideHLADataset(train_file, hla_seq_dict)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = PeptideHLADataset(test_file, hla_seq_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_loaders2(train_file, val_file, test_file, hla_seq_dict, batch_size=64):
    """基于训练集、验证集和测试集文件，创建训练、验证和测试数据加载器"""
    train_dataset = PeptideHLADataset(train_file, hla_seq_dict)
    val_dataset = PeptideHLADataset(val_file, hla_seq_dict)
    test_dataset = PeptideHLADataset(test_file, hla_seq_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
