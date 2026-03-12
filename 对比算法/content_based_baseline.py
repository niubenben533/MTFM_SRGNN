"""
基于内容的推荐算法 - 作为基线对比方法
基于API描述的TF-IDF特征和余弦相似度进行推荐
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class ContentBasedRecommender:
    def __init__(self, data_path):
        """
        初始化基于内容的推荐器
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.api_list = []
        self.api_descriptions = []
        self.train_data = None
        self.test_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        # 加载数据
        self.load_data()
        # 构建TF-IDF特征和相似度矩阵
        self.build_content_features()
        
    def load_data(self):
        """加载训练和测试数据"""
        # 加载API列表
        with open(os.path.join(self.data_path, 'used_api_list.json'), 'r', encoding='utf-8') as f:
            self.api_list = json.load(f)
        
        # 加载API描述
        with open(os.path.join(self.data_path, 'api_description.json'), 'r', encoding='utf-8') as f:
            self.api_descriptions = json.load(f)
        
        # 加载训练数据
        with open(os.path.join(self.data_path, 'train.json'), 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        
        # 加载测试数据
        with open(os.path.join(self.data_path, 'test.json'), 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            
        print(f"加载完成: {len(self.api_list)} APIs, {len(self.api_descriptions)} 描述, {len(self.train_data['sequences'])} 训练序列, {len(self.test_data['sequences'])} 测试序列")
        
        # 检查数据一致性
        if len(self.api_list) != len(self.api_descriptions):
            print(f"警告: API列表数量({len(self.api_list)})与描述数量({len(self.api_descriptions)})不匹配")
            # 调整到较小的数量
            min_len = min(len(self.api_list), len(self.api_descriptions))
            self.api_list = self.api_list[:min_len]
            self.api_descriptions = self.api_descriptions[:min_len]
            print(f"已调整为: {len(self.api_list)} APIs")
        
    def preprocess_text(self, text):
        """
        预处理文本：清理、分词、去停用词
        Args:
            text: 原始文本
        Returns:
            处理后的文本
        """
        if not text or text.strip() == "":
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除HTML标签和特殊字符
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # 移除常见停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
        
    def build_content_features(self):
        """构建TF-IDF特征和API相似度矩阵"""
        print("构建内容特征...")
        
        # 预处理API描述
        processed_descriptions = []
        for i in range(len(self.api_list)):
            if i < len(self.api_descriptions):
                processed_desc = self.preprocess_text(self.api_descriptions[i])
            else:
                processed_desc = ""
            
            if not processed_desc:  # 如果描述为空，使用API名称
                processed_desc = self.preprocess_text(self.api_list[i])
            processed_descriptions.append(processed_desc)
        
        # 构建TF-IDF矩阵
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # 最多5000个特征
            min_df=2,          # 至少在2个文档中出现
            max_df=0.8,        # 最多在80%的文档中出现
            ngram_range=(1, 2), # 使用1-gram和2-gram
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_descriptions)
        
        # 计算API之间的余弦相似度
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"TF-IDF矩阵形状: {self.tfidf_matrix.shape}")
        print(f"相似度矩阵形状: {self.similarity_matrix.shape}")
        
    def get_user_profile(self, session_sequence):
        """
        基于用户历史序列构建用户画像
        Args:
            session_sequence: 用户历史API序列
        Returns:
            用户画像向量
        """
        if not session_sequence:
            return np.zeros(self.tfidf_matrix.shape[1])
        
        # 获取历史API的TF-IDF向量
        user_vectors = []
        for api_id in session_sequence:
            if 0 <= api_id < len(self.api_list):
                user_vectors.append(self.tfidf_matrix[api_id].toarray().flatten())
        
        if not user_vectors:
            return np.zeros(self.tfidf_matrix.shape[1])
        
        # 计算用户画像（平均向量）
        user_profile = np.mean(user_vectors, axis=0)
        return user_profile
        
    def recommend(self, session_sequence, k=20):
        """
        基于内容相似度进行推荐
        Args:
            session_sequence: 用户历史API序列
            k: 推荐数量
        Returns:
            推荐的API ID列表
        """
        if not session_sequence:
            # 如果没有历史记录，返回随机推荐
            return np.random.choice(len(self.api_list), size=min(k, len(self.api_list)), replace=False).tolist()
        
        # 获取用户画像
        user_profile = self.get_user_profile(session_sequence)
        
        # 计算所有API与用户画像的相似度
        api_scores = []
        for api_id in range(len(self.api_list)):
            if api_id in session_sequence:
                # 已经使用过的API，相似度设为0
                api_scores.append(0.0)
            else:
                # 计算与用户画像的相似度
                api_vector = self.tfidf_matrix[api_id].toarray().flatten()
                similarity = cosine_similarity([user_profile], [api_vector])[0][0]
                api_scores.append(similarity)
        
        # 获取top-k推荐
        top_k_indices = np.argsort(api_scores)[::-1][:k]
        return top_k_indices.tolist()
        
    def evaluate(self, top_k_list=list(range(1, 31))):
        """
        评估推荐算法性能
        Args:
            top_k_list: 要评估的k值列表
        Returns:
            评估结果字典
        """
        print("开始评估...")
        
        all_predictions = []
        all_targets = []
        
        # 对每个测试序列进行推荐
        for i, (sequence, target) in enumerate(zip(self.test_data['sequences'], self.test_data['labels'])):
            if i % 100 == 0:
                print(f"处理测试样本: {i+1}/{len(self.test_data['sequences'])}")
            
            # 生成推荐列表（使用最大k值）
            max_k = max(top_k_list)
            predictions = self.recommend(sequence, k=max_k)
            
            all_predictions.append(predictions)
            all_targets.append([target])  # 转换为列表格式
        
        # 计算各种指标
        results = {}
        for k in top_k_list:
            # 截取前k个推荐
            k_predictions = [pred[:k] for pred in all_predictions]
            
            # 计算指标
            recall_scores = [recall(target, pred) for target, pred in zip(all_targets, k_predictions)]
            precision_scores = [precision(target, pred) for target, pred in zip(all_targets, k_predictions)]
            ndcg_scores = [ndcg(target, pred) for target, pred in zip(all_targets, k_predictions)]
            map_scores = [ap(target, pred) for target, pred in zip(all_targets, k_predictions)]
            
            results[k] = {
                'recall': np.mean(recall_scores),
                'precision': np.mean(precision_scores),
                'ndcg': np.mean(ndcg_scores),
                'map': np.mean(map_scores)
            }
        
        return results
        
    def save_results(self, results, output_path):
        """
        保存评估结果
        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON格式结果
        json_file = os.path.join(output_path, f'content_based_results_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'algorithm': 'Content-Based Recommendation',
                'timestamp': timestamp,
                'results': results,
                'summary': {
                    'avg_recall@10': results[10]['recall'],
                    'avg_precision@10': results[10]['precision'],
                    'avg_ndcg@10': results[10]['ndcg'],
                    'avg_map@10': results[10]['map']
                }
            }, f, indent=2, ensure_ascii=False)
        
        # CSV格式结果
        csv_file = os.path.join(output_path, f'content_based_results_{timestamp}.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('k,recall,precision,ndcg,map\n')
            for k in sorted(results.keys()):
                f.write(f"{k},{results[k]['recall']:.6f},{results[k]['precision']:.6f},"
                       f"{results[k]['ndcg']:.6f},{results[k]['map']:.6f}\n")
        
        # 控制台输出关键结果
        print("\n" + "="*60)
        print("基于内容的推荐算法 - 评估结果")
        print("="*60)
        print(f"{'K':<5} {'Recall':<10} {'Precision':<10} {'NDCG':<10} {'MAP':<10}")
        print("-"*60)
        
        key_ks = [1, 5, 10, 15, 20, 30]
        for k in key_ks:
            if k in results:
                print(f"{k:<5} {results[k]['recall']:<10.6f} {results[k]['precision']:<10.6f} "
                      f"{results[k]['ndcg']:<10.6f} {results[k]['map']:<10.6f}")
        
        print("="*60)
        print(f"结果已保存到:")
        print(f"  JSON: {json_file}")
        print(f"  CSV:  {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='基于内容的推荐算法')
    parser.add_argument('--data_path', type=str, 
                       default=r'E:\data\merge\SR_GNN_NEW\data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default=r'e:\data\merge\基线算法\results',
                       help='结果输出路径')
    parser.add_argument('--max_k', type=int, default=30,
                       help='最大k值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 初始化推荐器
    print("初始化基于内容的推荐器...")
    recommender = ContentBasedRecommender(args.data_path)
    
    # 评估算法
    top_k_list = list(range(1, args.max_k + 1))
    results = recommender.evaluate(top_k_list)
    
    # 保存结果
    recommender.save_results(results, args.output_path)

if __name__ == "__main__":
    main()