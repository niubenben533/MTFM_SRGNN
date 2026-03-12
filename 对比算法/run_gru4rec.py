"""
运行GRU4Rec推荐算法的便捷脚本
"""

import subprocess
import sys
import os

def main():
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'SR_GNN_NEW', 'data')
    output_path = os.path.join(script_dir, 'gru4rec_results')
    
    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(script_dir, 'gru4rec_baseline.py'),
        '--data_path', data_path,
        '--output_path', output_path,
        '--embedding_dim', '100',
        '--hidden_dim', '100',
        '--num_layers', '1',
        '--dropout', '0.2',
        '--learning_rate', '0.001',
        '--batch_size', '32',
        '--epochs', '10',
        '--max_length', '50',
        '--sample_size', '10000',
        '--max_k', '30'
    ]
    
    print("开始运行GRU4Rec推荐算法...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 50)
    
    # 运行算法
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nGRU4Rec算法运行完成!")
    except subprocess.CalledProcessError as e:
        print(f"\n运行出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())