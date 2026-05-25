"""
Model parameter statistics utilities for ImmuAtlas.
"""

import torch
from collections import OrderedDict


def format_params_count(count: int) -> str:
    """格式化参数数量显示"""
    if count >= 1e6:
        return f"{count/1e6:.1f} M"
    elif count >= 1e3:
        return f"{count/1e3:.1f} K"
    else:
        return str(count)


def detailed_model_summary(model: torch.nn.Module) -> dict:
    """
    生成详细的模型参数统计信息

    Returns
    -------
    dict
        包含 total_params / trainable_params / non_trainable_params / size_mb
    """
    summary = []
    total_params = 0
    trainable_params = 0
    total_size = 0

    module_params = OrderedDict()

    for name, module in model.named_children():
        params_count = sum(p.numel() for p in module.parameters())
        trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)

        module_params[name] = {
            'module': module,
            'params_count': params_count,
            'trainable_count': trainable_count,
            'type': module.__class__.__name__
        }

        total_params += params_count
        trainable_params += trainable_count

        for p in module.parameters():
            total_size += p.numel() * p.element_size()

    for name, param in model.named_parameters():
        if '.' not in name:
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            total_size += param.numel() * param.element_size()

            module_params[name] = {
                'module': None,
                'params_count': param.numel(),
                'trainable_count': param.numel() if param.requires_grad else 0,
                'type': type(param).__name__
            }

    COLUMN_WIDTH = 10
    NAME_WIDTH = 20
    TYPE_WIDTH = 20

    print(f"\n  | {'Name':<{NAME_WIDTH}} | {'Type':<{TYPE_WIDTH}} | {'Params':<{COLUMN_WIDTH}}")
    print("-----------------------------------------------------------")

    idx = 0
    for name, info in module_params.items():
        params_str = format_params_count(info['params_count'])
        print(f"{idx:<2}| {name:<{NAME_WIDTH}} | {info['type']:<{TYPE_WIDTH}} | {params_str:<{COLUMN_WIDTH}}")
        idx += 1

    print("-----------------------------------------------------------")
    print(f"{format_params_count(trainable_params):<{COLUMN_WIDTH}} Trainable params")
    non_trainable_params = total_params - trainable_params
    print(f"{format_params_count(non_trainable_params):<{COLUMN_WIDTH}} Non-trainable params")
    print(f"{format_params_count(total_params):<{COLUMN_WIDTH}} Total params")

    size_mb = total_size / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB"
    print(f"{size_str:<{COLUMN_WIDTH}} Total estimated model params size\n")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'size_mb': size_mb
    }
