"""
运行SR-GNN基线算法的便捷脚本
"""

import subprocess
import os

def run_srgnn():
    """运行SR-GNN基线算法"""
    
    # 设置路径
    script_path = os.path.join(os.path.dirname(__file__), 'srgnn_baseline.py')
    data_path = 'e:/data/merge/SR_GNN_NEW/data'
    output_path = 'e:/data/merge/基线算法/srgnn_results'
    
    # 构建命令
    cmd = [
        'python', script_path,
        '--data_path', data_path,
        '--output_path', output_path,
        '--hidden_size', '100',
        '--step', '1',
        '--learning_rate', '0.001',
        '--batch_size', '32',
        '--epochs', '10'
    ]
    
    print("开始运行SR-GNN基线算法...")
    print(f"命令: {' '.join(cmd)}")
    
    # 运行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("SR-GNN基线算法运行成功!")
            print("输出:")
            print(result.stdout)
        else:
            print("SR-GNN基线算法运行失败!")
            print("错误信息:")
            print(result.stderr)
            
    except Exception as e:
        print(f"运行过程中出现异常: {e}")

if __name__ == "__main__":
    run_srgnn()