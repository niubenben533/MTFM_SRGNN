"""
基于物品的协同过滤推荐算法 - 作为基线对比方法
基于API共现关系计算物品相似度，根据用户历史行为推荐相似API
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import csv

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class ItemBasedCFRecommender:
    def __init__(self, data_path):
        """
        初始化基于物品的协同过滤推荐器
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.api_list = []
        self.train_data = None
        self.test_data = None
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.api_popularity = {}
        
        # 加载数据
        self.load_data()
        # 构建用户-物品矩阵
        self.build_user_item_matrix()
        # 计算物品相似度矩阵
        self.calculate_item_similarity()
        
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
    
    def build_user_item_matrix(self):
        """构建用户-物品矩阵"""
        print("构建用户-物品矩阵...")
        
        num_users = len(self.train_data['sequences'])
        num_items = len(self.api_list)
        
        # 创建用户-物品矩阵
        user_item_data = []
        row_indices = []
        col_indices = []
        
        # 统计API流行度
        api_counter = Counter()
        
        for user_id, sequence in enumerate(self.train_data['sequences']):
            # 去重，每个用户对每个API只计算一次
            unique_apis = set(sequence)
            for api_id in unique_apis:
                if 0 <= api_id < num_items:  # 确保API ID有效
                    user_item_data.append(1.0)
                    row_indices.append(user_id)
                    col_indices.append(api_id)
                    api_counter[api_id] += 1
        
        # 创建稀疏矩阵
        self.user_item_matrix = csr_matrix(
            (user_item_data, (row_indices, col_indices)),
            shape=(num_users, num_items)
        )
        
        # 保存API流行度
        self.api_popularity = dict(api_counter)
        
        print(f"用户-物品矩阵构建完成: {self.user_item_matrix.shape}")
        print(f"矩阵密度: {self.user_item_matrix.nnz / (num_users * num_items):.6f}")
    
    def calculate_item_similarity(self):
        """计算物品相似度矩阵"""
        print("计算物品相似度矩阵...")
        
        # 转置矩阵，得到物品-用户矩阵
        item_user_matrix = self.user_item_matrix.T
        
        # 计算余弦相似度
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        
        # 将对角线设为0（物品与自身的相似度不考虑）
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print(f"物品相似度矩阵计算完成: {self.item_similarity_matrix.shape}")
        
        # 输出一些统计信息
        non_zero_similarities = np.count_nonzero(self.item_similarity_matrix)
        total_pairs = len(self.api_list) * (len(self.api_list) - 1)
        print(f"非零相似度对数: {non_zero_similarities}/{total_pairs}")
        print(f"平均相似度: {np.mean(self.item_similarity_matrix):.6f}")
        print(f"最大相似度: {np.max(self.item_similarity_matrix):.6f}")
    
    def get_user_profile(self, session_sequence):
        """
        获取用户画像（历史使用的API）
        Args:
            session_sequence: 用户会话序列
        Returns:
            用户使用过的API集合
        """
        return set(session_sequence)
    
    def recommend(self, session_sequence, k=20):
        """
        为给定的会话序列推荐API
        Args:
            session_sequence: 用户会话序列
            k: 推荐数量
        Returns:
            推荐的API ID列表
        """
        if not session_sequence:
            # 如果没有历史记录，返回最流行的API
            popular_apis = sorted(self.api_popularity.keys(), 
                                key=lambda x: self.api_popularity[x], 
                                reverse=True)
            return popular_apis[:k]
        
        # 获取用户历史使用的API
        user_apis = self.get_user_profile(session_sequence)
        
        # 计算每个候选API的推荐分数
        api_scores = defaultdict(float)
        
        for api_id in user_apis:
            if api_id < len(self.api_list):  # 确保API ID有效
                # 获取与当前API相似的所有API
                similarities = self.item_similarity_matrix[api_id]
                
                for candidate_api, similarity in enumerate(similarities):
                    if candidate_api not in user_apis and similarity > 0:
                        # 基于相似度和流行度的加权分数
                        popularity_weight = self.api_popularity.get(candidate_api, 0)
                        api_scores[candidate_api] += similarity * (1 + np.log(1 + popularity_weight))
        
        # 如果没有找到相似的API，使用流行度推荐
        if not api_scores:
            popular_apis = sorted(self.api_popularity.keys(), 
                                key=lambda x: self.api_popularity[x], 
                                reverse=True)
            candidates = [api for api in popular_apis if api not in user_apis]
            return candidates[:k]
        
        # 按分数排序并返回top-k
        recommended_apis = sorted(api_scores.items(), key=lambda x: x[1], reverse=True)
        return [api_id for api_id, score in recommended_apis[:k]]
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """
        评估推荐算法性能
        Args:
            top_k_list: 要评估的top-k值列表
        Returns:
            评估结果字典
        """
        print("开始评估...")
        
        test_sequences = self.test_data['sequences']
        test_labels = self.test_data['labels']
        
        results = {}
        
        for k in top_k_list:
            print(f"评估 Top-{k}...")
            
            recall_scores = []
            precision_scores = []
            ndcg_scores = []
            map_scores = []
            
            for i, (sequence, label) in enumerate(zip(test_sequences, test_labels)):
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{len(test_sequences)}")
                
                # 获取推荐结果
                recommendations = self.recommend(sequence, k)
                
                # 计算评估指标
                target = [label]  # 真实标签
                pred = recommendations  # 推荐结果
                
                # 计算各项指标
                recall_score = recall(target, pred)
                precision_score = precision(target, pred)
                ndcg_score = ndcg(target, pred)
                map_score = ap(target, pred)
                
                recall_scores.append(recall_score)
                precision_scores.append(precision_score)
                ndcg_scores.append(ndcg_score)
                map_scores.append(map_score)
            
            # 计算平均值
            results[k] = {
                'recall': np.mean(recall_scores),
                'precision': np.mean(precision_scores),
                'ndcg': np.mean(ndcg_scores),
                'map': np.mean(map_scores)
            }
            
            print(f"Top-{k} 结果: Recall={results[k]['recall']:.4f}, "
                  f"Precision={results[k]['precision']:.4f}, "
                  f"NDCG={results[k]['ndcg']:.4f}, "
                  f"MAP={results[k]['map']:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """
        保存评估结果
        Args:
            results: 评估结果字典
            output_path: 输出路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式结果
        json_file = os.path.join(output_path, f"item_based_cf_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        csv_file = os.path.join(output_path, f"item_based_cf_results_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Top-K', 'Recall', 'Precision', 'NDCG', 'MAP'])
            
            for k in sorted(results.keys()):
                writer.writerow([
                    k,
                    f"{results[k]['recall']:.6f}",
                    f"{results[k]['precision']:.6f}",
                    f"{results[k]['ndcg']:.6f}",
                    f"{results[k]['map']:.6f}"
                ])
        
        # 保存详细报告
        report_file = os.path.join(output_path, f"item_based_cf_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("基于物品的协同过滤推荐算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"算法类型: Item-based Collaborative Filtering\n")
            f.write(f"API总数: {len(self.api_list)}\n")
            f.write(f"训练序列数: {len(self.train_data['sequences'])}\n")
            f.write(f"测试序列数: {len(self.test_data['sequences'])}\n")
            f.write(f"用户-物品矩阵密度: {self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.6f}\n\n")
            
            f.write("评估结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Top-K':<6} {'Recall':<10} {'Precision':<10} {'NDCG':<10} {'MAP':<10}\n")
            f.write("-" * 50 + "\n")
            
            for k in sorted(results.keys()):
                f.write(f"{k:<6} {results[k]['recall']:<10.6f} {results[k]['precision']:<10.6f} "
                       f"{results[k]['ndcg']:<10.6f} {results[k]['map']:<10.6f}\n")
        
        print(f"\n结果已保存到:")
        print(f"- JSON: {json_file}")
        print(f"- CSV: {csv_file}")
        print(f"- 报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='基于物品的协同过滤推荐算法')
    parser.add_argument('--data_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW', 'data'),
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'item_cf_results'),
                       help='输出文件夹路径')
    parser.add_argument('--top_k_start', type=int, default=1, help='Top-K评估起始值')
    parser.add_argument('--top_k_end', type=int, default=30, help='Top-K评估结束值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    print("基于物品的协同过滤推荐算法")
    print("=" * 40)
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    print(f"评估范围: Top-{args.top_k_start} 到 Top-{args.top_k_end}")
    print()
    
    # 初始化推荐器
    recommender = ItemBasedCFRecommender(args.data_path)
    
    # 评估算法
    top_k_list = list(range(args.top_k_start, args.top_k_end + 1))
    results = recommender.evaluate(top_k_list)
    
    # 保存结果
    recommender.save_results(results, args.output_path)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()