"""
运行Session-KNN推荐算法的便捷脚本
"""

import os
import subprocess
import sys

def run_session_knn():
    """运行Session-KNN推荐算法"""
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置数据路径和输出路径
    data_path = os.path.join(current_dir, '..', 'SR_GNN_NEW', 'data')
    output_path = os.path.join(current_dir, 'session_knn_results')
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("Session-KNN推荐算法评估")
    print("=" * 60)
    
    # 测试不同的配置
    configs = [
        {
            'name': 'Jaccard相似度',
            'similarity_type': 'jaccard',
            'position_weight': False,
            'k_neighbors': 100
        },
        {
            'name': 'Jaccard相似度 + 位置加权',
            'similarity_type': 'jaccard',
            'position_weight': True,
            'k_neighbors': 100
        },
        {
            'name': '余弦相似度 + 位置加权',
            'similarity_type': 'cosine',
            'position_weight': True,
            'k_neighbors': 100
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] 运行配置: {config['name']}")
        print("-" * 40)
        
        # 构建命令
        cmd = [
            sys.executable, 'session_knn_baseline.py',
            '--data_path', data_path,
            '--output_path', output_path,
            '--k_neighbors', str(config['k_neighbors']),
            '--similarity_type', config['similarity_type'],
            '--sample_size', '1000',  # 为了效率限制采样大小
            '--max_k', '30'
        ]
        
        if config['position_weight']:
            cmd.append('--position_weight')
        
        try:
            # 运行算法
            result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 运行成功")
                # 显示部分输出
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-10:]:  # 显示最后10行
                    if line.strip():
                        print(f"  {line}")
            else:
                print("✗ 运行失败")
                print("错误信息:")
                print(result.stderr)
                
        except Exception as e:
            print(f"✗ 运行出错: {e}")
    
    print(f"\n所有结果已保存到: {output_path}")
    print("Session-KNN推荐算法评估完成!")

if __name__ == "__main__":
    run_session_knn()