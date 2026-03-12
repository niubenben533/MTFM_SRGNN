"""
Random推荐算法 - 作为基线对比方法
随机推荐API，不考虑任何历史信息或特征
"""

import json
import random
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class RandomRecommender:
    def __init__(self, data_path):
        """
        初始化Random推荐器
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.api_list = []
        self.train_data = None
        self.test_data = None
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载训练和测试数据"""
        # 加载API列表
        with open(os.path.join(self.data_path, 'used_api_list.json'), 'r', encoding='utf-8') as f:
            self.api_list = json.load(f)
        
        # 加载训练数据
        with open(os.path.join(self.data_path, 'train.json'), 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        
        # 加载测试数据
        with open(os.path.join(self.data_path, 'test.json'), 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            
        print(f"加载完成:")
        print(f"- API总数: {len(self.api_list)}")
        print(f"- 训练序列数: {len(self.train_data['sequences'])}")
        print(f"- 测试序列数: {len(self.test_data['sequences'])}")
        
    def recommend(self, session_sequence, k=20):
        """
        为给定的会话序列推荐top-k个API
        Args:
            session_sequence: 会话序列（API ID列表）
            k: 推荐数量
        Returns:
            推荐的API ID列表
        """
        # 获取所有可能的API ID（从0到API总数-1）
        all_api_ids = list(range(len(self.api_list)))
        
        # 从会话序列中排除已经出现的API（避免重复推荐）
        if session_sequence:
            available_apis = [api_id for api_id in all_api_ids if api_id not in session_sequence]
        else:
            available_apis = all_api_ids
            
        # 如果可用API数量少于k，则推荐所有可用API
        if len(available_apis) < k:
            return available_apis
        
        # 随机选择k个API
        return random.sample(available_apis, k)
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """
        在测试集上评估Random推荐算法
        Args:
            top_k_list: 评估的k值列表，默认为1到30
        Returns:
            评估结果字典
        """
        print("开始评估Random推荐算法...")
        print(f"评估Top-K范围: {min(top_k_list)} 到 {max(top_k_list)}")
        
        # 初始化评估指标
        metrics = {k: {'recall': [], 'precision': [], 'ndcg': [], 'map': []} for k in top_k_list}
        
        test_sequences = self.test_data['sequences']
        test_labels = self.test_data['labels']
        
        total_sessions = len(test_sequences)
        
        for i, (sequence, label) in enumerate(zip(test_sequences, test_labels)):
            if i % 100 == 0:
                print(f"处理进度: {i}/{total_sessions}")
            
            # 获取会话序列（除了最后一个API）
            session_sequence = sequence[:-1] if len(sequence) > 1 else []
            
            # 真实标签（最后一个API）
            target = [label]
            
            # 生成最大k值的推荐结果，然后截取不同长度
            max_k = max(top_k_list)
            full_recommendations = self.recommend(session_sequence, max_k)
            
            # 为每个k值进行评估
            for k in top_k_list:
                # 截取前k个推荐
                recommendations = full_recommendations[:k]
                
                # 计算评估指标
                metrics[k]['recall'].append(recall(target, recommendations))
                metrics[k]['precision'].append(precision(target, recommendations))
                metrics[k]['ndcg'].append(ndcg(target, recommendations))
                metrics[k]['map'].append(ap(target, recommendations))
        
        # 计算平均值
        results = {}
        for k in top_k_list:
            results[k] = {
                'recall': np.mean(metrics[k]['recall']),
                'precision': np.mean(metrics[k]['precision']),
                'ndcg': np.mean(metrics[k]['ndcg']),
                'map': np.mean(metrics[k]['map'])
            }
            
        return results
    
    def save_results(self, results, output_path):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 保存详细结果
        result_file = os.path.join(output_path, f"random_baseline_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存格式化的结果报告
        report_file = os.path.join(output_path, f"random_baseline_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Random推荐算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {timestamp}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"API总数: {len(self.api_list)}\n")
            f.write(f"测试序列数: {len(self.test_data['sequences'])}\n\n")
            
            f.write("评估结果:\n")
            f.write("-" * 30 + "\n")
            for k in sorted(results.keys()):
                f.write(f"\nTop-{k} 推荐结果:\n")
                f.write(f"  Recall@{k}:    {results[k]['recall']:.4f}\n")
                f.write(f"  Precision@{k}: {results[k]['precision']:.4f}\n")
                f.write(f"  NDCG@{k}:      {results[k]['ndcg']:.4f}\n")
                f.write(f"  MAP@{k}:       {results[k]['map']:.4f}\n")
        
        # 保存CSV格式的结果（便于后续分析）
        csv_file = os.path.join(output_path, f"random_baseline_results_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("topk,recall,precision,ndcg,map\n")
            for k in sorted(results.keys()):
                f.write(f"{k},{results[k]['recall']:.6f},{results[k]['precision']:.6f},"
                       f"{results[k]['ndcg']:.6f},{results[k]['map']:.6f}\n")
        
        print(f"结果已保存到:")
        print(f"- {result_file}")
        print(f"- {report_file}")
        print(f"- {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Random推荐算法基线')
    parser.add_argument('--data_path', type=str, 
                       default='E:\\data\\merge\\SR_GNN_NEW\\data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default='E:\\data\\merge\\基线算法\\results',
                       help='结果输出路径')
    parser.add_argument('--top_k', type=int, nargs='+',
                       default=list(range(1, 31)),
                       help='评估的top-k值列表，默认1-30')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    print("Random推荐算法基线评估")
    print("=" * 40)
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    print(f"评估Top-K: {min(args.top_k)} 到 {max(args.top_k)}")
    print(f"随机种子: {args.seed}")
    print()
    
    # 初始化推荐器
    recommender = RandomRecommender(args.data_path)
    
    # 评估
    results = recommender.evaluate(args.top_k)
    
    # 显示结果（只显示部分关键结果，避免输出过长）
    print("\n评估完成！部分关键结果如下:")
    print("-" * 40)
    key_k_values = [1, 5, 10, 15, 20, 25, 30]
    for k in key_k_values:
        if k in results:
            print(f"Top-{k}:")
            print(f"  Recall:    {results[k]['recall']:.4f}")
            print(f"  Precision: {results[k]['precision']:.4f}")
            print(f"  NDCG:      {results[k]['ndcg']:.4f}")
            print(f"  MAP:       {results[k]['map']:.4f}")
            print()
    
    # 保存结果
    recommender.save_results(results, args.output_path)

if __name__ == "__main__":
    main()