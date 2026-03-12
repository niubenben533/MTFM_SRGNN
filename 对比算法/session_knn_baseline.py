"""
Session-KNN推荐算法 - 作为基线对比方法
基于会话相似度的k近邻推荐，支持多种相似度计算方法和位置加权
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
import math

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class SessionKNNRecommender:
    def __init__(self, data_path, k_neighbors=100, similarity_type='jaccard', 
                 position_weight=True, position_decay=0.1, sample_size=1000):
        """
        初始化Session-KNN推荐器
        Args:
            data_path: 数据文件夹路径
            k_neighbors: 考虑的邻居会话数量
            similarity_type: 相似度计算方法 ('jaccard', 'cosine', 'dice')
            position_weight: 是否使用位置加权
            position_decay: 位置衰减因子
            sample_size: 训练会话采样大小（为了效率）
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.similarity_type = similarity_type
        self.position_weight = position_weight
        self.position_decay = position_decay
        self.sample_size = sample_size
        
        self.api_list = []
        self.train_data = None
        self.test_data = None
        self.train_sessions = []  # 训练会话列表
        self.api_popularity = {}  # API流行度统计
        
        # 加载数据
        self.load_data()
        # 预处理训练会话
        self.preprocess_sessions()
        
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
        
        print(f"加载完成: {len(self.api_list)} APIs, {len(self.train_data['sequences'])} 训练序列, {len(self.test_data['sequences'])} 测试序列")
    
    def preprocess_sessions(self):
        """预处理训练会话，计算API流行度"""
        # 统计API流行度
        api_counts = Counter()
        
        # 处理训练会话
        for sequence in self.train_data['sequences']:
            if len(sequence) >= 2:  # 至少需要2个API才能形成会话
                self.train_sessions.append(sequence)
                api_counts.update(sequence)
        
        # 如果训练会话太多，进行采样
        if len(self.train_sessions) > self.sample_size:
            import random
            random.seed(42)
            self.train_sessions = random.sample(self.train_sessions, self.sample_size)
            print(f"采样训练会话: {len(self.train_sessions)}")
        
        # 计算API流行度
        total_count = sum(api_counts.values())
        self.api_popularity = {api: count / total_count for api, count in api_counts.items()}
        
        print(f"预处理完成: {len(self.train_sessions)} 训练会话")
    
    def calculate_jaccard_similarity(self, session1, session2):
        """计算Jaccard相似度"""
        set1, set2 = set(session1), set(session2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def calculate_cosine_similarity(self, session1, session2):
        """计算余弦相似度"""
        set1, set2 = set(session1), set(session2)
        intersection = len(set1 & set2)
        magnitude = math.sqrt(len(set1) * len(set2))
        return intersection / magnitude if magnitude > 0 else 0.0
    
    def calculate_dice_similarity(self, session1, session2):
        """计算Dice相似度"""
        set1, set2 = set(session1), set(session2)
        intersection = len(set1 & set2)
        return 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0
    
    def calculate_position_weighted_similarity(self, session1, session2):
        """计算位置加权相似度"""
        # 为会话中的每个位置分配权重
        weights1 = {api: math.exp(-self.position_decay * (len(session1) - 1 - i)) 
                   for i, api in enumerate(session1)}
        weights2 = {api: math.exp(-self.position_decay * (len(session2) - 1 - i)) 
                   for i, api in enumerate(session2)}
        
        # 计算加权交集
        common_apis = set(session1) & set(session2)
        weighted_intersection = sum(min(weights1[api], weights2[api]) for api in common_apis)
        
        # 计算加权并集
        all_apis = set(session1) | set(session2)
        weighted_union = sum(max(weights1.get(api, 0), weights2.get(api, 0)) for api in all_apis)
        
        return weighted_intersection / weighted_union if weighted_union > 0 else 0.0
    
    def calculate_similarity(self, session1, session2):
        """计算两个会话之间的相似度"""
        if self.position_weight:
            return self.calculate_position_weighted_similarity(session1, session2)
        
        if self.similarity_type == 'jaccard':
            return self.calculate_jaccard_similarity(session1, session2)
        elif self.similarity_type == 'cosine':
            return self.calculate_cosine_similarity(session1, session2)
        elif self.similarity_type == 'dice':
            return self.calculate_dice_similarity(session1, session2)
        else:
            raise ValueError(f"不支持的相似度类型: {self.similarity_type}")
    
    def find_similar_sessions(self, current_session):
        """找到与当前会话最相似的k个历史会话"""
        similarities = []
        
        for train_session in self.train_sessions:
            # 跳过相同的会话
            if train_session == current_session:
                continue
            
            similarity = self.calculate_similarity(current_session, train_session)
            if similarity > 0:  # 只考虑有相似度的会话
                similarities.append((similarity, train_session))
        
        # 按相似度排序并返回top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:self.k_neighbors]
    
    def recommend(self, session_sequence, k=20):
        """
        为给定会话序列推荐API
        Args:
            session_sequence: 当前会话序列
            k: 推荐的API数量
        Returns:
            推荐的API列表
        """
        if not session_sequence:
            return self.get_popular_recommendations(k)
        
        # 找到相似会话
        similar_sessions = self.find_similar_sessions(session_sequence)
        
        if not similar_sessions:
            return self.get_popular_recommendations(k)
        
        # 聚合推荐分数
        api_scores = defaultdict(float)
        current_session_set = set(session_sequence)
        
        for similarity, train_session in similar_sessions:
            # 为训练会话中的每个API分配分数
            for i, api in enumerate(train_session):
                if api not in current_session_set:  # 排除已经在当前会话中的API
                    # 位置权重：后面的API权重更高
                    position_weight = math.exp(-self.position_decay * (len(train_session) - 1 - i)) if self.position_weight else 1.0
                    api_scores[api] += similarity * position_weight
        
        # 如果没有推荐结果，使用流行度回退
        if not api_scores:
            return self.get_popular_recommendations(k, exclude=current_session_set)
        
        # 按分数排序并返回top-k
        sorted_apis = sorted(api_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [api for api, score in sorted_apis[:k]]
        
        # 如果推荐数量不足，用流行度补充
        if len(recommendations) < k:
            popular_apis = self.get_popular_recommendations(k - len(recommendations), 
                                                          exclude=current_session_set | set(recommendations))
            recommendations.extend(popular_apis)
        
        return recommendations[:k]
    
    def get_popular_recommendations(self, k, exclude=None):
        """获取流行度推荐作为回退策略"""
        if exclude is None:
            exclude = set()
        
        # 按流行度排序
        sorted_apis = sorted(self.api_popularity.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for api, popularity in sorted_apis:
            if api not in exclude and len(recommendations) < k:
                recommendations.append(api)
        
        return recommendations
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """评估推荐算法性能"""
        print("开始评估Session-KNN推荐算法...")
        
        results = {}
        
        for k in top_k_list:
            print(f"评估 Top-{k}...")
            
            precisions = []
            recalls = []
            ndcgs = []
            aps = []
            num_evaluated = 0  # 添加计数器
            
            for i, sequence in enumerate(self.test_data['sequences']):
                if len(sequence) < 2:
                    continue
                
                # 分割序列：前n-1个作为输入，最后一个作为目标
                input_sequence = sequence[:-1]
                target_api = sequence[-1]
                
                # 获取推荐
                recommendations = self.recommend(input_sequence, k)
                
                # 计算指标
                if recommendations:
                    # 修正：使用正确的参数格式
                    target_list = [target_api]  # 目标API列表
                    pred_list = recommendations  # 推荐API列表
                    
                    # 计算各项指标
                    prec = precision(target_list, pred_list)
                    rec = recall(target_list, pred_list)
                    ndcg_score = ndcg(target_list, pred_list)
                    ap_score = ap(target_list, pred_list)
                    
                    precisions.append(prec)
                    recalls.append(rec)
                    ndcgs.append(ndcg_score)
                    aps.append(ap_score)
                    num_evaluated += 1  # 增加计数
                
                # 进度显示
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(self.test_data['sequences'])} 个测试序列")
            
            # 计算平均指标
            avg_precision = np.mean(precisions) if precisions else 0.0
            avg_recall = np.mean(recalls) if recalls else 0.0
            avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
            avg_ap = np.mean(aps) if aps else 0.0
            
            results[k] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'ndcg': avg_ndcg,
                'map': avg_ap,
                'hit_rate': avg_recall,  # 对于单目标，hit_rate等于recall
                'num_evaluated': num_evaluated  # 添加评估数量字段
            }
            
            print(f"  Top-{k}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, "
                  f"NDCG={avg_ndcg:.4f}, MAP={avg_ap:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果到JSON
        json_file = os.path.join(output_path, f'session_knn_results_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        csv_file = os.path.join(output_path, f'session_knn_results_{timestamp}.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('k,precision,recall,ndcg,map,num_evaluated\n')
            for k, metrics in results.items():
                f.write(f"{k},{metrics['precision']:.6f},{metrics['recall']:.6f},"
                       f"{metrics['ndcg']:.6f},{metrics['map']:.6f},{metrics['num_evaluated']}\n")
        
        # 保存可读性报告
        report_file = os.path.join(output_path, f'session_knn_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Session-KNN推荐算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"算法参数:\n")
            f.write(f"  - K邻居数: {self.k_neighbors}\n")
            f.write(f"  - 相似度类型: {self.similarity_type}\n")
            f.write(f"  - 位置加权: {self.position_weight}\n")
            f.write(f"  - 位置衰减: {self.position_decay}\n")
            f.write(f"  - 采样大小: {self.sample_size}\n")
            f.write(f"  - 训练会话数: {len(self.train_sessions)}\n")
            f.write(f"  - 测试序列数: {len(self.test_data['sequences'])}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'K':<5} {'Precision':<12} {'Recall':<12} {'NDCG':<12} {'MAP':<12} {'Evaluated':<10}\n")
            f.write("-" * 80 + "\n")
            
            for k in sorted(results.keys()):
                metrics = results[k]
                f.write(f"{k:<5} {metrics['precision']:<12.6f} {metrics['recall']:<12.6f} "
                       f"{metrics['ndcg']:<12.6f} {metrics['map']:<12.6f} {metrics['num_evaluated']:<10}\n")
        
        print(f"\n结果已保存到:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Session-KNN推荐算法')
    parser.add_argument('--data_path', type=str, default='../SR_GNN_NEW/data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str, default='./session_knn_results',
                       help='结果输出路径')
    parser.add_argument('--k_neighbors', type=int, default=100,
                       help='考虑的邻居会话数量')
    parser.add_argument('--similarity_type', type=str, default='jaccard',
                       choices=['jaccard', 'cosine', 'dice'],
                       help='相似度计算方法')
    parser.add_argument('--position_weight', action='store_true', default=True,
                       help='是否使用位置加权')
    parser.add_argument('--position_decay', type=float, default=0.1,
                       help='位置衰减因子')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='训练会话采样大小')
    parser.add_argument('--max_k', type=int, default=30,
                       help='评估的最大K值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 初始化推荐器
    recommender = SessionKNNRecommender(
        data_path=args.data_path,
        k_neighbors=args.k_neighbors,
        similarity_type=args.similarity_type,
        position_weight=args.position_weight,
        position_decay=args.position_decay,
        sample_size=args.sample_size
    )
    
    # 评估算法
    results = recommender.evaluate(top_k_list=list(range(1, args.max_k + 1)))
    
    # 保存结果
    recommender.save_results(results, args.output_path)
    
    print("\nSession-KNN推荐算法评估完成!")

if __name__ == "__main__":
    main()