"""
MTFM文本消融实验基线算法
基于MTFM的文本处理部分，仅使用文本信息进行API推荐，忽略序列信息
包含语义组件(CNN)和特征交互组件，但不使用序列建模
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
import random
from tqdm import tqdm
import pickle

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class MTFMTextDataset(Dataset):
    """MTFM文本数据集"""
    
    def __init__(self, sessions, api_to_id, vocab, max_length=50):
        self.sessions = sessions
        self.api_to_id = api_to_id
        self.vocab = vocab
        self.max_length = max_length
        self.data = []
        self.prepare_data()
    
    def prepare_data(self):
        """准备训练数据"""
        for session in self.sessions:
            if len(session) < 2:
                continue
            
            # 对于每个会话，使用前面的API作为上下文，预测下一个API
            for i in range(1, len(session)):
                context_apis = session[:i]
                target_api = session[i]
                
                if target_api in self.api_to_id:
                    # 创建文本描述（简化处理，使用API名称作为文本）
                    text_tokens = []
                    for api in context_apis:
                        if api in self.api_to_id:
                            # 简化处理：将API名称分词作为文本特征
                            tokens = api.lower().replace('_', ' ').split()
                            text_tokens.extend(tokens)
                    
                    # 如果没有文本tokens，跳过这个样本
                    if not text_tokens:
                        continue
                    
                    # 转换为词汇表ID
                    token_ids = []
                    for token in text_tokens:
                        if token in self.vocab:
                            token_ids.append(self.vocab[token])
                        else:
                            token_ids.append(self.vocab.get('<UNK>', 1))
                    
                    # 如果没有有效的token_ids，跳过这个样本
                    if not token_ids:
                        continue
                    
                    # 截断或填充到固定长度
                    if len(token_ids) > self.max_length:
                        token_ids = token_ids[:self.max_length]
                    else:
                        token_ids.extend([0] * (self.max_length - len(token_ids)))
                    
                    self.data.append({
                        'text_ids': token_ids,
                        'target': self.api_to_id[target_api]
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['text_ids'], dtype=torch.long),
            torch.tensor(item['target'], dtype=torch.long)
        )

class MTFMTextModel(nn.Module):
    """MTFM文本模型（消融实验版本）"""
    
    def __init__(self, vocab_size, num_apis, embed_dim=100, num_kernel=256, 
                 kernel_sizes=[2, 3, 4, 5], feature_dim=8, dropout=0.2):
        super(MTFMTextModel, self).__init__()
        
        self.num_apis = num_apis
        self.feature_dim = feature_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 语义组件：CNN
        self.sc_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_kernel, kernel_size=h),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for h in kernel_sizes
        ])
        
        self.sc_fcl = nn.Linear(num_kernel * len(kernel_sizes), num_apis)
        
        # 特征交互组件
        self.fic_fc = nn.Linear(num_kernel * len(kernel_sizes), feature_dim)
        self.fic_api_feature_embedding = nn.Parameter(torch.rand(feature_dim, num_apis))
        
        self.fic_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Linear(feature_dim, 1),
            nn.Tanh()
        )
        
        self.fic_fcl = nn.Linear(num_apis * 2, num_apis)
        
        # 融合层
        self.fusion_layer = nn.Linear(num_apis * 2, num_apis)
        
        # 输出层
        self.output_layer = nn.Linear(num_apis, num_apis)
        
        # 激活函数和dropout
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text_ids):
        """前向传播"""
        batch_size = text_ids.size(0)
        
        # 嵌入
        embedded = self.embedding(text_ids)  # [batch_size, max_length, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embed_dim, max_length]
        
        # 语义组件：多尺度CNN
        conv_outputs = []
        for conv in self.sc_convs:
            conv_out = conv(embedded)  # [batch_size, num_kernel, 1]
            conv_out = conv_out.squeeze(-1)  # [batch_size, num_kernel]
            conv_outputs.append(conv_out)
        
        e = torch.cat(conv_outputs, dim=1)  # [batch_size, num_kernel * len(kernel_sizes)]
        u_sc = self.sc_fcl(e)  # [batch_size, num_apis]
        
        # 特征交互组件
        u_sc_trans = self.fic_fc(e)  # [batch_size, feature_dim]
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)  # [batch_size, num_apis]
        
        # MLP交互
        u_concate = []
        for i in range(batch_size):
            u_sc_single = u_sc_trans[i]  # [feature_dim]
            u_concate_single = torch.cat([
                u_sc_single.repeat(self.num_apis, 1),  # [num_apis, feature_dim]
                self.fic_api_feature_embedding.t()     # [num_apis, feature_dim]
            ], dim=1)  # [num_apis, feature_dim * 2]
            
            mlp_out = self.fic_mlp(u_concate_single).squeeze()  # [num_apis]
            u_concate.append(mlp_out)
        
        u_mlp = torch.stack(u_concate)  # [batch_size, num_apis]
        
        u_fic = self.fic_fcl(torch.cat([u_mm, u_mlp], dim=1))  # [batch_size, num_apis]
        u_fic = self.tanh(u_fic)
        
        # 融合层
        u_mmf = self.fusion_layer(torch.cat([u_sc, u_fic], dim=1))  # [batch_size, num_apis]
        u_mmf = self.dropout(u_mmf)
        
        # 输出层
        logits = self.output_layer(u_mmf)  # [batch_size, num_apis]
        
        return logits

class MTFMTextRecommender:
    """MTFM文本推荐器"""
    
    def __init__(self, data_path, embed_dim=100, num_kernel=256, kernel_sizes=[2, 3, 4, 5],
                 feature_dim=8, dropout=0.2, learning_rate=0.001, batch_size=32, 
                 epochs=10, max_length=50):
        self.data_path = data_path
        self.embed_dim = embed_dim
        self.num_kernel = num_kernel
        self.kernel_sizes = kernel_sizes
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 数据
        self.api_list = []
        self.api_to_id = {}
        self.id_to_api = {}
        self.train_sessions = []
        self.test_sessions = []
        self.vocab = {}
        self.api_popularity = {}
        
        # 模型
        self.model = None
        
        # 加载数据并训练模型
        self.load_data()
        self.build_vocabulary()
        self.train_model()
        
    def load_data(self):
        """加载数据"""
        # 加载API列表
        with open(os.path.join(self.data_path, 'used_api_list.json'), 'r', encoding='utf-8') as f:
            self.api_list = json.load(f)
        
        # 构建API映射
        self.api_to_id = {api: idx for idx, api in enumerate(self.api_list)}
        self.id_to_api = {idx: api for idx, api in enumerate(self.api_list)}
        
        # 加载训练数据
        with open(os.path.join(self.data_path, 'train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            # 修复：正确处理数据格式
            if isinstance(train_data, dict) and 'sequences' in train_data:
                sequences = train_data['sequences']
            else:
                sequences = train_data
            
            # 将API ID转换为API名称
            self.train_sessions = []
            for session in sequences:
                if len(session) >= 2:
                    # 将ID转换为API名称
                    api_session = []
                    for api_id in session:
                        if api_id < len(self.api_list):
                            api_session.append(self.api_list[api_id])
                    if len(api_session) >= 2:
                        self.train_sessions.append(api_session)
        
        # 加载测试数据
        with open(os.path.join(self.data_path, 'test.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            # 修复：正确处理数据格式
            if isinstance(test_data, dict) and 'sequences' in test_data:
                sequences = test_data['sequences']
            else:
                sequences = test_data
            
            # 将API ID转换为API名称
            self.test_sessions = []
            for session in sequences:
                if len(session) >= 2:
                    # 将ID转换为API名称
                    api_session = []
                    for api_id in session:
                        if api_id < len(self.api_list):
                            api_session.append(self.api_list[api_id])
                    if len(api_session) >= 2:
                        self.test_sessions.append(api_session)
        
        # 计算API流行度
        api_counts = Counter()
        for session in self.train_sessions:
            api_counts.update(session)
        
        total_count = sum(api_counts.values())
        self.api_popularity = {api: count / total_count for api, count in api_counts.items()}
        
        print(f"加载完成: {len(self.api_list)} APIs, {len(self.train_sessions)} 训练序列, {len(self.test_sessions)} 测试序列")
    
    def build_vocabulary(self):
        """构建词汇表"""
        vocab_counter = Counter()
        
        # 从API名称中提取词汇
        for api in self.api_list:
            tokens = api.lower().replace('_', ' ').split()
            vocab_counter.update(tokens)
        
        # 构建词汇表
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in vocab_counter.most_common(10000):  # 限制词汇表大小
            if count >= 2:  # 过滤低频词
                self.vocab[word] = len(self.vocab)
        
        print(f"词汇表大小: {len(self.vocab)} 词汇")
    
    def train_model(self):
        """训练模型"""
        # 创建数据集
        train_dataset = MTFMTextDataset(self.train_sessions, self.api_to_id, self.vocab, self.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型
        self.model = MTFMTextModel(
            vocab_size=len(self.vocab),
            num_apis=len(self.api_list),
            embed_dim=self.embed_dim,
            num_kernel=self.num_kernel,
            kernel_sizes=self.kernel_sizes,
            feature_dim=self.feature_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("开始训练MTFM文本模型...")
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for text_ids, targets in progress_bar:
                text_ids = text_ids.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits = self.model(text_ids)
                loss = criterion(logits, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.epochs}, 平均损失: {avg_loss:.4f}")
    
    def recommend(self, session_sequence, k=20):
        """为给定会话序列推荐API"""
        if not session_sequence:
            return self.get_popular_recommendations(k)
        
        # 准备文本输入
        text_tokens = []
        for api in session_sequence:
            if api in self.api_to_id:
                tokens = api.lower().replace('_', ' ').split()
                text_tokens.extend(tokens)
        
        # 如果没有有效的文本tokens，返回流行度推荐
        if not text_tokens:
            return self.get_popular_recommendations(k)
        
        # 转换为词汇表ID
        token_ids = []
        for token in text_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab.get('<UNK>', 1))
        
        # 如果没有有效的token_ids，返回流行度推荐
        if not token_ids:
            return self.get_popular_recommendations(k)
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        
        # 转换为张量
        text_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_tensor)
            scores = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # 排除已经使用的API
        used_apis = set(session_sequence)
        
        # 获取推荐
        recommendations = []
        api_scores = [(i, score) for i, score in enumerate(scores)]
        api_scores.sort(key=lambda x: x[1], reverse=True)
        
        for api_id, score in api_scores:
            if api_id < len(self.id_to_api):  # 确保api_id有效
                api = self.id_to_api[api_id]
                if api not in used_apis:
                    recommendations.append(api)
                    if len(recommendations) >= k:
                        break
        
        # 如果推荐不足，用流行度补充
        if len(recommendations) < k:
            popular_apis = self.get_popular_recommendations(k - len(recommendations), exclude=used_apis.union(set(recommendations)))
            recommendations.extend(popular_apis)
        
        return recommendations[:k]
    
    def get_popular_recommendations(self, k, exclude=None):
        """获取流行度推荐"""
        if exclude is None:
            exclude = set()
        
        # 按流行度排序
        sorted_apis = sorted(self.api_popularity.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for api, _ in sorted_apis:
            if api not in exclude and api in self.api_to_id:
                recommendations.append(api)
                if len(recommendations) >= k:
                    break
        
        return recommendations
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """评估模型性能"""
        print("开始评估...")
        
        all_precisions = {k: [] for k in top_k_list}
        all_recalls = {k: [] for k in top_k_list}
        all_ndcgs = {k: [] for k in top_k_list}
        all_maps = {k: [] for k in top_k_list}
        
        num_evaluated = 0
        max_k = max(top_k_list)
        
        for session in tqdm(self.test_sessions, desc="评估进度"):
            if len(session) < 2:
                continue
            
            # 使用前面的API作为输入，预测最后一个API
            input_sequence = session[:-1]
            target_api = session[-1]
            
            if target_api not in self.api_to_id:
                continue
            
            # 获取最大k的推荐列表，然后为不同k值截取
            recommendations = self.recommend(input_sequence, max_k)
            
            # 计算指标
            target_list = [target_api]
            
            # 为每个k值计算指标
            for k in top_k_list:
                pred_list = recommendations[:k]
                
                # 计算各项指标
                prec = precision(pred_list, target_list)
                rec = recall(pred_list, target_list)
                ndcg_score = ndcg(pred_list, target_list)
                map_score = ap(pred_list, target_list)
                
                all_precisions[k].append(prec)
                all_recalls[k].append(rec)
                all_ndcgs[k].append(ndcg_score)
                all_maps[k].append(map_score)
            
            num_evaluated += 1
        
        # 计算平均指标
        results = {}
        for k in top_k_list:
            results[k] = {
                'precision': np.mean(all_precisions[k]) if all_precisions[k] else 0.0,
                'recall': np.mean(all_recalls[k]) if all_recalls[k] else 0.0,
                'ndcg': np.mean(all_ndcgs[k]) if all_ndcgs[k] else 0.0,
                'map': np.mean(all_maps[k]) if all_maps[k] else 0.0,
                'num_evaluated': num_evaluated
            }
        
        # 打印结果
        print("\n评估结果:")
        print("Top-K\tPrecision\tRecall\t\tNDCG\t\tMAP")
        print("-" * 60)
        for k in top_k_list:
            print(f"Top-{k:2d}\t{results[k]['precision']:.4f}\t\t{results[k]['recall']:.4f}\t\t{results[k]['ndcg']:.4f}\t\t{results[k]['map']:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = os.path.join(output_path, f"mtfm_text_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式
        csv_file = os.path.join(output_path, f"mtfm_text_results_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Top-K,Precision,Recall,NDCG,MAP,Num_Evaluated\n")
            for k, metrics in results.items():
                f.write(f"{k},{metrics['precision']:.6f},{metrics['recall']:.6f},"
                       f"{metrics['ndcg']:.6f},{metrics['map']:.6f},{metrics['num_evaluated']}\n")
        
        # 保存文本报告
        report_file = os.path.join(output_path, f"mtfm_text_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MTFM文本消融实验基线算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估样本数: {results[1]['num_evaluated']}\n\n")
            
            f.write("算法特点:\n")
            f.write("- 基于MTFM的文本处理部分\n")
            f.write("- 使用CNN进行语义特征提取\n")
            f.write("- 包含特征交互组件\n")
            f.write("- 忽略序列信息，仅使用文本信息\n")
            f.write("- 适用于消融实验对比\n\n")
            
            f.write("详细结果:\n")
            f.write("Top-K\tPrecision\tRecall\t\tNDCG\t\tMAP\n")
            f.write("-" * 60 + "\n")
            for k, metrics in results.items():
                f.write(f"Top-{k:2d}\t{metrics['precision']:.4f}\t\t{metrics['recall']:.4f}\t\t"
                       f"{metrics['ndcg']:.4f}\t\t{metrics['map']:.4f}\n")
        
        print(f"\n结果已保存到:")
        print(f"- JSON: {results_file}")
        print(f"- CSV: {csv_file}")
        print(f"- 报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='MTFM文本消融实验基线算法')
    parser.add_argument('--data_path', type=str, default='../SR_GNN_NEW/data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str, default='./mtfm_text_results',
                       help='结果输出路径')
    parser.add_argument('--embed_dim', type=int, default=100,
                       help='嵌入维度')
    parser.add_argument('--num_kernel', type=int, default=256,
                       help='卷积核数量')
    parser.add_argument('--feature_dim', type=int, default=8,
                       help='特征交互维度')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大文本长度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 创建推荐器并训练
    recommender = MTFMTextRecommender(
        data_path=args.data_path,
        embed_dim=args.embed_dim,
        num_kernel=args.num_kernel,
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length
    )
    
    # 评估
    results = recommender.evaluate()
    
    # 保存结果
    recommender.save_results(results, args.output_path)

if __name__ == "__main__":
    main()