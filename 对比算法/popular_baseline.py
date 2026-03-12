"""
Popular推荐算法 - 作为基线对比方法
基于流行度推荐API，推荐在训练集中出现频率最高的API
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from collections import Counter

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class PopularRecommender:
    def __init__(self, data_path):
        """
        初始化Popular推荐器
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.api_list = []
        self.train_data = None
        self.test_data = None
        self.api_popularity = {}  # API流行度统计
        self.popular_apis = []    # 按流行度排序的API列表
        
        # 加载数据
        self.load_data()
        # 计算API流行度
        self.calculate_popularity()
        
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
    
    def calculate_popularity(self):
        """计算API在训练集中的流行度"""
        print("计算API流行度...")
        
        # 统计每个API在训练集中的出现次数
        api_counter = Counter()
        
        for sequence in self.train_data['sequences']:
            for api_id in sequence:
                api_counter[api_id] += 1
        
        # 转换为字典格式
        self.api_popularity = dict(api_counter)
        
        # 按流行度排序API（从高到低）
        self.popular_apis = sorted(self.api_popularity.keys(), 
                                 key=lambda x: self.api_popularity[x], 
                                 reverse=True)
        
        print(f"流行度统计完成:")
        print(f"- 最受欢迎的API (ID: {self.popular_apis[0]}) 出现次数: {self.api_popularity[self.popular_apis[0]]}")
        print(f"- 前10个最受欢迎的API: {self.popular_apis[:10]}")
        
        # 处理未在训练集中出现的API（如果有的话）
        all_api_ids = set(range(len(self.api_list)))
        train_api_ids = set(self.api_popularity.keys())
        missing_apis = all_api_ids - train_api_ids
        
        if missing_apis:
            print(f"- 训练集中未出现的API数量: {len(missing_apis)}")
            # 将未出现的API添加到列表末尾，流行度为0
            for api_id in missing_apis:
                self.api_popularity[api_id] = 0
            self.popular_apis.extend(list(missing_apis))
    
    def recommend(self, session_sequence, k=20):
        """
        为给定的会话序列推荐top-k个API
        Args:
            session_sequence: 会话序列（在Popular算法中不使用，但保持接口一致性）
            k: 推荐的API数量
        Returns:
            推荐的API ID列表
        """
        # Popular算法不考虑会话序列，直接返回最受欢迎的k个API
        return self.popular_apis[:k]
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """
        评估Popular推荐算法
        Args:
            top_k_list: 要评估的top-k值列表
        Returns:
            评估结果字典
        """
        print("开始评估Popular推荐算法...")
        
        results = {}
        
        # 获取测试数据
        test_sequences = self.test_data['sequences']
        test_labels = self.test_data['labels']
        
        for k in top_k_list:
            print(f"评估 Top-{k}...")
            
            recalls = []
            precisions = []
            ndcgs = []
            maps = []
            
            for i, (sequence, label) in enumerate(zip(test_sequences, test_labels)):
                # 获取推荐结果（截取到k个）
                recommendations = self.recommend(sequence, k)
                
                # 计算各项指标 - 注意这里只传递两个参数
                rec = recall([label], recommendations)
                prec = precision([label], recommendations)
                ndcg_score = ndcg([label], recommendations)
                map_score = ap([label], recommendations)
                
                recalls.append(rec)
                precisions.append(prec)
                ndcgs.append(ndcg_score)
                maps.append(map_score)
            
            # 计算平均值
            avg_recall = np.mean(recalls)
            avg_precision = np.mean(precisions)
            avg_ndcg = np.mean(ndcgs)
            avg_map = np.mean(maps)
            
            results[f'top_{k}'] = {
                'recall': avg_recall,
                'precision': avg_precision,
                'ndcg': avg_ndcg,
                'map': avg_map
            }
            
            print(f"Top-{k} - Recall: {avg_recall:.4f}, Precision: {avg_precision:.4f}, "
                  f"NDCG: {avg_ndcg:.4f}, MAP: {avg_map:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """
        保存评估结果
        Args:
            results: 评估结果字典
            output_path: 输出路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 保存JSON格式结果
        json_file = os.path.join(output_path, f"popular_baseline_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        csv_file = os.path.join(output_path, f"popular_baseline_results_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Top-K,Recall,Precision,NDCG,MAP\n")
            for key, metrics in results.items():
                k = key.replace('top_', '')
                f.write(f"{k},{metrics['recall']:.6f},{metrics['precision']:.6f},"
                       f"{metrics['ndcg']:.6f},{metrics['map']:.6f}\n")
        
        # 保存详细报告
        report_file = os.path.join(output_path, f"popular_baseline_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Popular推荐算法评估报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {timestamp}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"API总数: {len(self.api_list)}\n")
            f.write(f"训练序列数: {len(self.train_data['sequences'])}\n")
            f.write(f"测试序列数: {len(self.test_data['sequences'])}\n")
            f.write(f"最受欢迎的API: {self.popular_apis[0]} (出现{self.api_popularity[self.popular_apis[0]]}次)\n")
            f.write("\n详细结果:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Top-K':<6} {'Recall':<10} {'Precision':<10} {'NDCG':<10} {'MAP':<10}\n")
            f.write("-" * 50 + "\n")
            
            for key, metrics in results.items():
                k = key.replace('top_', '')
                f.write(f"{k:<6} {metrics['recall']:<10.6f} {metrics['precision']:<10.6f} "
                       f"{metrics['ndcg']:<10.6f} {metrics['map']:<10.6f}\n")
        
        print(f"\n结果已保存:")
        print(f"- JSON: {json_file}")
        print(f"- CSV: {csv_file}")
        print(f"- 报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Popular推荐算法基线')
    parser.add_argument('--data_path', type=str, 
                       default=r'E:\data\merge\SR_GNN_NEW\data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default=r'E:\data\merge\基线算法\results',
                       help='结果输出路径')
    parser.add_argument('--top_k_max', type=int, default=30,
                       help='最大top-k值 (默认: 30)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    
    print("Popular推荐算法基线评估")
    print("=" * 40)
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    print(f"评估范围: Top-1 到 Top-{args.top_k_max}")
    print("=" * 40)
    
    # 创建推荐器
    recommender = PopularRecommender(args.data_path)
    
    # 评估
    top_k_list = list(range(1, args.top_k_max + 1))
    results = recommender.evaluate(top_k_list)
    
    # 保存结果
    recommender.save_results(results, args.output_path)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()