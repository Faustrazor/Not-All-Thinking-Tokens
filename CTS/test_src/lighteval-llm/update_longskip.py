#!/usr/bin/env python3
"""
更新lighteval.sh中的skip值的工具

使用方法: python update_longskip.py skip2_85 [脚本路径]
默认脚本路径为当前目录下的lighteval.sh
"""

import sys
import re
import os


def update_skip_pattern(new_pattern, script_path="lighteval.sh"):
    """
    更新指定脚本文件中的skip值
    
    参数:
    new_pattern: 新的skip值，例如 'skip2_85' 或 'longskip_80'
    script_path: 要更新的脚本文件路径，默认为 'lighteval.sh'
    """
    # 验证输入格式
    if not re.match(r'^[a-zA-Z]+\d*_\d+$', new_pattern):
        print(f"错误: 输入格式不正确，请使用类似'skip2_85'或'longskip_80'的格式")
        sys.exit(1)
    
    # 解析前缀和数字部分
    parts = new_pattern.split('_')
    prefix = parts[0]
    number_part = parts[1]
    
    # 不同场景使用的值
    model_pattern = f"{prefix}_{number_part}_"  # 用于MODEL路径中
    
    # 检查文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 找不到文件 {script_path}")
        sys.exit(1)
    
    # 读取文件内容
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    changed = False
    
    # 逐行处理替换
    for line in lines:
        original_line = line
        
        # 替换MODEL行中的skip值
        if line.startswith('MODEL='):
            # 查找当前的模式，支持多种前缀
            match = re.search(r'([a-zA-Z]+\d*)_(\d+)_', line)
            if match:
                old_prefix = match.group(1)
                old_number = match.group(2)
                old_pattern = f"{old_prefix}_{old_number}_"
                # 执行替换
                if old_pattern != model_pattern:  # 只有当值不同时才替换
                    line = line.replace(old_pattern, model_pattern)
                    changed = True
        
        # 替换target行中的目录名部分
        elif line.startswith('target='):
            # 查找target行格式，确保只替换目录名部分
            # 格式如: target=/data0/users/yuanh/lighteval-llm/llama8B/longskip90
            parts = line.strip().split('/')
            if len(parts) >= 2:
                # 找到最后一个部分（目录名）
                last_part = parts[-1]
                # 检查目录名是否包含skip模式
                dir_match = re.match(r'([a-zA-Z]+\d*)(\d+|_\d+)$', last_part)
                if dir_match:
                    old_dir_prefix = dir_match.group(1)
                    old_dir_suffix = dir_match.group(2)
                    
                    # 确定新的目录名格式
                    new_dir_name = f"{prefix}_{number_part}"
                    
                    # 替换最后一个部分
                    parts[-1] = new_dir_name
                    new_target_line = '/'.join(parts) + '\n'
                    if new_target_line != line:
                        line = new_target_line
                        changed = True
        
        new_lines.append(line)
        
        # 打印替换信息（调试用）
        if original_line != line:
            print(f"替换: \n  从: {original_line.strip()}\n  到: {line.strip()}")
    
    if not changed:
        print("未找到需要替换的内容或内容已经是最新的")
        return
    
    # 写入修改后的内容
    with open(script_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"已成功将{script_path}中的skip值更新为 {new_pattern}")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python update_longskip.py 前缀_数字 [脚本路径]")
        print("例如: python update_longskip.py skip2_85")
        print("      python update_longskip.py longskip_80 my_script.sh")
        sys.exit(1)
    
    # 解析命令行参数
    new_skip = sys.argv[1]
    
    # 如果提供了脚本路径，则使用它，否则使用默认值
    if len(sys.argv) > 2:
        script_path = sys.argv[2]
        update_skip_pattern(new_skip, script_path)
    else:
        update_skip_pattern(new_skip) 