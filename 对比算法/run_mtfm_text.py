"""
运行MTFM文本消融实验基线算法的脚本
"""

import subprocess
import os
import sys

def run_mtfm_text():
    """运行MTFM文本基线算法"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'SR_GNN_NEW', 'data')
    output_path = os.path.join(current_dir, 'mtfm_text_results')
    script_path = os.path.join(current_dir, 'mtfm_text_baseline.py')
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("运行MTFM文本消融实验基线算法")
    print("=" * 60)
    
    # 构建命令
    cmd = [
        sys.executable, script_path,
        '--data_path', data_path,
        '--output_path', output_path,
        '--embed_dim', '100',
        '--num_kernel', '256',
        '--feature_dim', '8',
        '--learning_rate', '0.001',
        '--batch_size', '32',
        '--epochs', '10',
        '--max_length', '50'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 运行算法
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\nMTFM文本基线算法运行完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        return False
    except Exception as e:
        print(f"发生异常: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_mtfm_text()
    if success:
        print("\n所有任务完成!")
    else:
        print("\n任务执行失败!")