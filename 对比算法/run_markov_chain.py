"""
运行Markov Chain推荐算法的便捷脚本
"""

import subprocess
import os
import sys

def main():
    # 设置路径
    data_path = r'e:\data\merge\SR_GNN_NEW\data'
    output_path = r'e:\data\merge\基线算法\markov_results'
    
    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), 'markov_chain_baseline.py')
    
    cmd = [
        sys.executable, script_path,
        '--data_path', data_path,
        '--output_path', output_path,
        '--smoothing_alpha', '0.1',
        '--max_k', '30'
    ]
    
    print("运行Markov Chain推荐算法...")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    # 运行脚本
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
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

if __name__ == "__main__":
    main()