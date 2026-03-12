"""
运行基于内容的推荐算法的便捷脚本
"""

import subprocess
import sys
import os

def run_content_based():
    """运行基于内容的推荐算法"""
    
    # 设置路径
    data_path = r'E:\data\merge\SR_GNN_NEW\data'
    output_path = r'e:\data\merge\基线算法\results'
    script_path = os.path.join(os.path.dirname(__file__), 'content_based_baseline.py')
    
    # 构建命令
    cmd = [
        sys.executable,
        script_path,
        '--data_path', data_path,
        '--output_path', output_path,
        '--max_k', '30'
    ]
    
    print("运行基于内容的推荐算法...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("-" * 50)
    
    # 运行脚本
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n算法运行完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n运行出错: {e}")
        return e.returncode
    except Exception as e:
        print(f"\n未知错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_content_based()
    sys.exit(exit_code)