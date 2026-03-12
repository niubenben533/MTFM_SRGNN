"""
运行Popular基线算法的简化脚本
"""

import subprocess
import os
import sys

def run_popular_baseline():
    """运行Popular基线算法"""
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置数据路径和输出路径
    data_path = os.path.join(os.path.dirname(current_dir), 'SR_GNN_NEW', 'data')
    output_path = os.path.join(current_dir, 'results')
    
    # Popular基线脚本路径
    popular_script = os.path.join(current_dir, 'popular_baseline.py')
    
    print("运行Popular推荐算法基线...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("-" * 50)
    
    # 构建命令
    cmd = [
        sys.executable,
        popular_script,
        '--data_path', data_path,
        '--output_path', output_path,
        '--top_k_max', '30'
    ]
    
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"运行失败，错误码: {e.returncode}")
        print("标准输出:")
        print(e.stdout)
        print("标准错误:")
        print(e.stderr)
    except Exception as e:
        print(f"运行出错: {str(e)}")

if __name__ == "__main__":
    run_popular_baseline()