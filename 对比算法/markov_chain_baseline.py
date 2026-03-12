"""
Markov Chain推荐算法 - 作为基线对比方法
基于一阶马尔可夫链建模API序列转移概率，支持平滑和回退策略
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class MarkovChainRecommender:
    def __init__(self, data_path, smoothing_alpha=0.1, use_popularity_fallback=True):
        """
        初始化Markov Chain推荐器
        Args:
            data_path: 数据文件夹路径
            smoothing_alpha: 平滑参数，用于处理未见过的转移
            use_popularity_fallback: 是否使用流行度作为回退策略
        """
        self.data_path = data_path
        self.smoothing_alpha = smoothing_alpha
        self.use_popularity_fallback = use_popularity_fallback
        
        self.api_list = []
        self.train_data = None
        self.test_data = None
        self.transition_matrix = defaultdict(lambda: defaultdict(int))  # 转移计数矩阵
        self.transition_probs = defaultdict(dict)  # 转移概率矩阵
        self.api_popularity = {}  # API流行度统计
        self.total_transitions = defaultdict(int)  # 每个API的总转移次数
        
        # 加载数据
        self.load_data()
        # 构建转移矩阵
        self.build_transition_matrix()
        # 计算转移概率
        self.calculate_transition_probabilities()
        
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
    
    def build_transition_matrix(self):
        """从训练序列构建转移矩阵"""
        print("构建马尔可夫转移矩阵...")
        
        # 统计API流行度（用于回退策略）
        api_counter = Counter()
        
        for sequence in self.train_data['sequences']:
            if len(sequence) < 2:
                continue
                
            # 统计API流行度
            api_counter.update(sequence)
            
            # 构建转移关系
            for i in range(len(sequence) - 1):
                current_api = sequence[i]
                next_api = sequence[i + 1]
                
                self.transition_matrix[current_api][next_api] += 1
                self.total_transitions[current_api] += 1
        
        # 计算API流行度
        total_api_count = sum(api_counter.values())
        self.api_popularity = {api: count / total_api_count 
                              for api, count in api_counter.items()}
        
        print(f"转移矩阵构建完成: {len(self.transition_matrix)} 个起始API, "
              f"{sum(len(transitions) for transitions in self.transition_matrix.values())} 个转移关系")
    
    def calculate_transition_probabilities(self):
        """计算转移概率，应用平滑策略"""
        print("计算转移概率...")
        
        vocab_size = len(self.api_list)
        
        for current_api in self.transition_matrix:
            total_count = self.total_transitions[current_api]
            
            # 为每个可能的下一个API计算概率
            for next_api in range(len(self.api_list)):
                # 获取转移计数
                count = self.transition_matrix[current_api].get(next_api, 0)
                
                # 应用加法平滑 (Add-alpha smoothing)
                smoothed_prob = (count + self.smoothing_alpha) / (total_count + self.smoothing_alpha * vocab_size)
                
                self.transition_probs[current_api][next_api] = smoothed_prob
        
        print("转移概率计算完成")
    
    def recommend(self, session_sequence, k=20):
        """
        基于当前会话序列推荐下一个API
        Args:
            session_sequence: 当前会话序列
            k: 推荐数量
        Returns:
            推荐的API ID列表
        """
        if not session_sequence:
            # 如果序列为空，返回最流行的API
            return self.get_popular_recommendations(k)
        
        # 获取最后一个API作为当前状态
        last_api = session_sequence[-1]
        
        # 如果最后一个API在转移矩阵中
        if last_api in self.transition_probs:
            # 获取所有可能的下一个API及其概率
            next_api_probs = self.transition_probs[last_api]
            
            # 排序并选择top-k
            sorted_apis = sorted(next_api_probs.items(), key=lambda x: x[1], reverse=True)
            
            # 过滤掉已经在当前会话中的API（避免重复推荐）
            session_set = set(session_sequence)
            recommendations = []
            
            for api_id, prob in sorted_apis:
                if api_id not in session_set and len(recommendations) < k:
                    recommendations.append(api_id)
            
            # 如果推荐数量不足，用流行度补充
            if len(recommendations) < k and self.use_popularity_fallback:
                popular_recs = self.get_popular_recommendations(k - len(recommendations), exclude=session_set | set(recommendations))
                recommendations.extend(popular_recs)
            
            return recommendations[:k]
        
        else:
            # 如果当前API未见过，使用流行度推荐
            return self.get_popular_recommendations(k, exclude=set(session_sequence))
    
    def get_popular_recommendations(self, k, exclude=None):
        """基于流行度获取推荐"""
        if exclude is None:
            exclude = set()
        
        # 按流行度排序
        sorted_apis = sorted(self.api_popularity.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for api_id, popularity in sorted_apis:
            if api_id not in exclude and len(recommendations) < k:
                recommendations.append(api_id)
        
        return recommendations[:k]
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """
        评估推荐算法性能
        Args:
            top_k_list: 要评估的k值列表
        Returns:
            评估结果字典
        """
        print("开始评估...")
        
        results = {k: {'hit': [], 'mrr': [], 'ndcg': [], 'precision': [], 'recall': []} 
                  for k in top_k_list}
        
        total_sessions = len(self.test_data['sequences'])
        
        for idx, sequence in enumerate(self.test_data['sequences']):
            if len(sequence) < 2:
                continue
            
            # 使用序列的前n-1个API作为输入，最后一个作为目标
            input_sequence = sequence[:-1]
            target_api = sequence[-1]
            
            # 获取推荐
            max_k = max(top_k_list)
            recommendations = self.recommend(input_sequence, k=max_k)
            
            # 计算各种指标
            for k in top_k_list:
                top_k_recs = recommendations[:k]
                
                # Hit@K
                hit = 1 if target_api in top_k_recs else 0
                results[k]['hit'].append(hit)
                
                # MRR@K
                if target_api in top_k_recs:
                    rank = top_k_recs.index(target_api) + 1
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0
                results[k]['mrr'].append(mrr)
                
                # NDCG@K
                if target_api in top_k_recs:
                    rank = top_k_recs.index(target_api) + 1
                    ndcg_score = 1.0 / np.log2(rank + 1)
                else:
                    ndcg_score = 0.0
                results[k]['ndcg'].append(ndcg_score)
                
                # Precision@K
                precision_score = hit / k
                results[k]['precision'].append(precision_score)
                
                # Recall@K (对于单个目标项，recall等于hit)
                results[k]['recall'].append(hit)
            
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{total_sessions} 个测试序列")
        
        # 计算平均指标
        final_results = {}
        for k in top_k_list:
            final_results[k] = {
                'hit': np.mean(results[k]['hit']),
                'mrr': np.mean(results[k]['mrr']),
                'ndcg': np.mean(results[k]['ndcg']),
                'precision': np.mean(results[k]['precision']),
                'recall': np.mean(results[k]['recall'])
            }
        
        return final_results
    
    def save_results(self, results, output_path):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 保存JSON格式结果
        json_file = os.path.join(output_path, f"markov_chain_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        csv_file = os.path.join(output_path, f"markov_chain_results_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("K,Hit@K,MRR@K,NDCG@K,Precision@K,Recall@K\n")
            for k in sorted(results.keys()):
                metrics = results[k]
                f.write(f"{k},{metrics['hit']:.4f},{metrics['mrr']:.4f},"
                       f"{metrics['ndcg']:.4f},{metrics['precision']:.4f},{metrics['recall']:.4f}\n")
        
        # 保存详细报告
        report_file = os.path.join(output_path, f"markov_chain_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Markov Chain 推荐算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"平滑参数: {self.smoothing_alpha}\n")
            f.write(f"使用流行度回退: {self.use_popularity_fallback}\n")
            f.write(f"API总数: {len(self.api_list)}\n")
            f.write(f"训练序列数: {len(self.train_data['sequences'])}\n")
            f.write(f"测试序列数: {len(self.test_data['sequences'])}\n")
            f.write(f"转移关系数: {len(self.transition_matrix)}\n\n")
            
            f.write("评估结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'K':<3} {'Hit@K':<8} {'MRR@K':<8} {'NDCG@K':<8} {'Precision@K':<12} {'Recall@K':<10}\n")
            f.write("-" * 60 + "\n")
            
            for k in sorted(results.keys()):
                metrics = results[k]
                f.write(f"{k:<3} {metrics['hit']:<8.4f} {metrics['mrr']:<8.4f} "
                       f"{metrics['ndcg']:<8.4f} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f}\n")
        
        print(f"结果已保存到: {output_path}")
        print(f"- JSON文件: {json_file}")
        print(f"- CSV文件: {csv_file}")
        print(f"- 报告文件: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Markov Chain 推荐算法')
    parser.add_argument('--data_path', type=str, 
                       default=r'e:\data\merge\SR_GNN_NEW\data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default=r'e:\data\merge\基线算法\markov_results',
                       help='结果输出路径')
    parser.add_argument('--smoothing_alpha', type=float, default=0.1,
                       help='平滑参数')
    parser.add_argument('--no_popularity_fallback', action='store_true',
                       help='不使用流行度回退策略')
    parser.add_argument('--max_k', type=int, default=30,
                       help='最大K值')
    
    args = parser.parse_args()
    
    print("Markov Chain 推荐算法")
    print("=" * 50)
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    print(f"平滑参数: {args.smoothing_alpha}")
    print(f"使用流行度回退: {not args.no_popularity_fallback}")
    print()
    
    # 初始化推荐器
    recommender = MarkovChainRecommender(
        data_path=args.data_path,
        smoothing_alpha=args.smoothing_alpha,
        use_popularity_fallback=not args.no_popularity_fallback
    )
    
    # 评估
    top_k_list = list(range(1, args.max_k + 1))
    results = recommender.evaluate(top_k_list)
    
    # 保存结果
    recommender.save_results(results, args.output_path)
    
    # 打印关键结果
    print("\n关键评估结果:")
    print("-" * 40)
    for k in [1, 5, 10, 20]:
        if k in results:
            metrics = results[k]
            print(f"K={k:2d}: Hit={metrics['hit']:.4f}, MRR={metrics['mrr']:.4f}, "
                  f"NDCG={metrics['ndcg']:.4f}")

if __name__ == "__main__":
    main()