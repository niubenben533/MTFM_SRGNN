"""
运行基于物品的协同过滤推荐算法的便捷脚本
"""

import os
import subprocess
import sys

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置数据路径和输出路径
    data_path = os.path.join(current_dir, '..', 'SR_GNN_NEW', 'data')
    output_path = os.path.join(current_dir, 'item_cf_results')
    
    # 构建命令
    script_path = os.path.join(current_dir, 'item_based_cf.py')
    cmd = [
        sys.executable, script_path,
        '--data_path', data_path,
        '--output_path', output_path,
        '--top_k_start', '1',
        '--top_k_end', '30'
    ]
    
    print("运行基于物品的协同过滤推荐算法...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print()
    
    # 运行脚本
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("算法运行成功!")
        print("\n--- 输出 ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- 警告/错误 ---")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"算法运行失败! 错误代码: {e.returncode}")
        print("\n--- 错误输出 ---")
        print(e.stderr)
        print("\n--- 标准输出 ---")
        print(e.stdout)

if __name__ == "__main__":
    main()