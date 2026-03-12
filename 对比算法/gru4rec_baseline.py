"""
GRU4Rec推荐算法 - 作为基线对比方法
使用GRU建模会话序列，预测下一个API
基于论文: "Session-based Recommendations with Recurrent Neural Networks"
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

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class SessionDataset(Dataset):
    """会话数据集类"""
    def __init__(self, sessions, api_to_id, max_length=50):
        self.sessions = sessions
        self.api_to_id = api_to_id
        self.max_length = max_length
        self.data = self.prepare_data()
    
    def prepare_data(self):
        """准备训练数据"""
        data = []
        for session in self.sessions:
            if len(session) < 2:
                continue
            
            # 将API转换为ID
            session_ids = [self.api_to_id.get(api, 0) for api in session]
            
            # 创建输入-输出对
            for i in range(1, len(session_ids)):
                input_seq = session_ids[:i]
                target = session_ids[i]
                
                # 截断或填充序列
                if len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length:]
                
                data.append((input_seq, target))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """批处理函数"""
    sequences, targets = zip(*batch)
    
    # 找到最大长度
    max_len = max(len(seq) for seq in sequences)
    
    # 填充序列
    padded_sequences = []
    lengths = []
    
    for seq in sequences:
        length = len(seq)
        padded_seq = [0] * (max_len - length) + seq  # 左填充
        padded_sequences.append(padded_seq)
        lengths.append(length)
    
    return (torch.LongTensor(padded_sequences), 
            torch.LongTensor(lengths),
            torch.LongTensor(targets))

class GRU4RecModel(nn.Module):
    """GRU4Rec模型"""
    def __init__(self, num_items, embedding_dim=100, hidden_dim=100, num_layers=1, dropout=0.2):
        super(GRU4RecModel, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # GRU层
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)
        
    def forward(self, sequences, lengths):
        """前向传播"""
        batch_size = sequences.size(0)
        
        # 嵌入
        embedded = self.embedding(sequences)
        
        # 打包序列 - lengths必须在CPU上
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths_cpu, 
                                                  batch_first=True, enforce_sorted=False)
        
        # GRU
        output, hidden = self.gru(packed)
        
        # 解包
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # 获取每个序列的最后一个有效输出
        last_outputs = []
        for i, length in enumerate(lengths_cpu):
            last_outputs.append(output[i, length-1, :])
        
        last_outputs = torch.stack(last_outputs)
        
        # 应用dropout和输出层
        last_outputs = self.dropout(last_outputs)
        logits = self.output_layer(last_outputs)
        
        return logits

class GRU4RecRecommender:
    def __init__(self, data_path, embedding_dim=100, hidden_dim=100, num_layers=1, 
                 dropout=0.2, learning_rate=0.001, batch_size=32, epochs=10, 
                 max_length=50, sample_size=10000):
        """
        初始化GRU4Rec推荐器
        Args:
            data_path: 数据文件夹路径
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout率
            learning_rate: 学习率
            batch_size: 批大小
            epochs: 训练轮数
            max_length: 最大序列长度
            sample_size: 训练数据采样大小
        """
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.sample_size = sample_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.api_list = []
        self.train_data = None
        self.test_data = None
        self.api_to_id = {}
        self.id_to_api = {}
        self.model = None
        self.api_popularity = {}
        
        # 加载数据
        self.load_data()
        # 构建词汇表
        self.build_vocabulary()
        # 训练模型
        self.train_model()
        
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
        
        print(f"加载完成: {len(self.api_list)} APIs, {len(self.train_data['sequences'])} 训练序列, "
              f"{len(self.test_data['sequences'])} 测试序列")
    
    def build_vocabulary(self):
        """构建API词汇表"""
        # 创建API到ID的映射（0保留给padding）
        self.api_to_id = {api: i+1 for i, api in enumerate(self.api_list)}
        self.id_to_api = {i+1: api for i, api in enumerate(self.api_list)}
        
        # 计算API流行度
        api_counts = Counter()
        for sequence in self.train_data['sequences']:
            api_counts.update(sequence)
        
        total_count = sum(api_counts.values())
        self.api_popularity = {api: count / total_count for api, count in api_counts.items()}
        
        print(f"词汇表大小: {len(self.api_to_id)} APIs")
    
    def train_model(self):
        """训练GRU4Rec模型"""
        print("开始训练GRU4Rec模型...")
        
        # 采样训练数据
        train_sessions = self.train_data['sequences']
        if len(train_sessions) > self.sample_size:
            train_sessions = random.sample(train_sessions, self.sample_size)
            print(f"采样训练会话: {len(train_sessions)}")
        
        # 创建数据集和数据加载器
        dataset = SessionDataset(train_sessions, self.api_to_id, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                               collate_fn=collate_fn)
        
        # 创建模型
        self.model = GRU4RecModel(
            num_items=len(self.api_list),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for sequences, lengths, targets in progress_bar:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits = self.model(sequences, lengths)
                loss = criterion(logits, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.epochs}, 平均损失: {avg_loss:.4f}")
        
        print("模型训练完成!")
    
    def recommend(self, session_sequence, k=20):
        """
        为给定会话序列推荐API
        Args:
            session_sequence: 当前会话序列
            k: 推荐的API数量
        Returns:
            推荐的API列表
        """
        if not session_sequence or self.model is None:
            return self.get_popular_recommendations(k)
        
        self.model.eval()
        with torch.no_grad():
            # 转换为ID序列
            session_ids = [self.api_to_id.get(api, 0) for api in session_sequence]
            session_ids = [id for id in session_ids if id != 0]  # 移除未知API
            
            if not session_ids:
                return self.get_popular_recommendations(k)
            
            # 截断序列
            if len(session_ids) > self.max_length:
                session_ids = session_ids[-self.max_length:]
            
            # 准备输入
            sequences = torch.LongTensor([session_ids]).to(self.device)
            lengths = torch.LongTensor([len(session_ids)]).to(self.device)
            
            # 预测
            logits = self.model(sequences, lengths)
            scores = torch.softmax(logits, dim=1)[0]  # 获取第一个（也是唯一一个）样本的分数
            
            # 排除已在会话中的API和padding
            current_session_set = set(session_sequence)
            
            # 获取top-k推荐
            recommendations = []
            _, top_indices = torch.topk(scores, min(k * 3, len(scores)))  # 获取更多候选
            
            for idx in top_indices:
                api_id = idx.item()
                if api_id == 0:  # 跳过padding
                    continue
                
                api = self.id_to_api.get(api_id)
                if api and api not in current_session_set:
                    recommendations.append(api)
                    if len(recommendations) >= k:
                        break
            
            # 如果推荐不足，用流行度补充
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
        for api, _ in sorted_apis:
            if api not in exclude:
                recommendations.append(api)
                if len(recommendations) >= k:
                    break
        
        return recommendations
    
    def evaluate(self, top_k_list=list(range(1, 31))):
        """评估推荐算法性能"""
        print("开始评估GRU4Rec推荐算法...")
        
        results = {}
        
        for k in top_k_list:
            print(f"评估 Top-{k}...")
            
            precisions = []
            recalls = []
            ndcgs = []
            aps = []
            num_evaluated = 0
            
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
                    target_list = [target_api]
                    pred_list = recommendations
                    
                    # 计算各项指标
                    prec = precision(target_list, pred_list)
                    rec = recall(target_list, pred_list)
                    ndcg_score = ndcg(target_list, pred_list)
                    ap_score = ap(target_list, pred_list)
                    
                    precisions.append(prec)
                    recalls.append(rec)
                    ndcgs.append(ndcg_score)
                    aps.append(ap_score)
                    num_evaluated += 1
                
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
                'hit_rate': avg_recall,
                'num_evaluated': num_evaluated
            }
            
            print(f"  Top-{k}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, "
                  f"NDCG={avg_ndcg:.4f}, MAP={avg_ap:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存详细结果到JSON
        json_file = os.path.join(output_path, f'gru4rec_results_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        csv_file = os.path.join(output_path, f'gru4rec_results_{timestamp}.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('k,precision,recall,ndcg,map,num_evaluated\n')
            for k, metrics in results.items():
                f.write(f"{k},{metrics['precision']:.6f},{metrics['recall']:.6f},"
                       f"{metrics['ndcg']:.6f},{metrics['map']:.6f},{metrics['num_evaluated']}\n")
        
        # 保存可读性报告
        report_file = os.path.join(output_path, f'gru4rec_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GRU4Rec推荐算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"算法参数:\n")
            f.write(f"  - 嵌入维度: {self.embedding_dim}\n")
            f.write(f"  - 隐藏层维度: {self.hidden_dim}\n")
            f.write(f"  - GRU层数: {self.num_layers}\n")
            f.write(f"  - Dropout率: {self.dropout}\n")
            f.write(f"  - 学习率: {self.learning_rate}\n")
            f.write(f"  - 批大小: {self.batch_size}\n")
            f.write(f"  - 训练轮数: {self.epochs}\n")
            f.write(f"  - 最大序列长度: {self.max_length}\n")
            f.write(f"  - 采样大小: {self.sample_size}\n")
            f.write(f"  - 词汇表大小: {len(self.api_to_id)}\n")
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
    parser = argparse.ArgumentParser(description='GRU4Rec推荐算法')
    parser.add_argument('--data_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW', 'data'),
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'gru4rec_results'),
                       help='输出文件夹路径')
    parser.add_argument('--embedding_dim', type=int, default=100, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=100, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1, help='GRU层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=50, help='最大序列长度')
    parser.add_argument('--sample_size', type=int, default=10000, help='训练数据采样大小')
    parser.add_argument('--max_k', type=int, default=30, help='最大K值')
    
    args = parser.parse_args()
    
    # 创建推荐器
    recommender = GRU4RecRecommender(
        data_path=args.data_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        sample_size=args.sample_size
    )
    
    # 评估
    results = recommender.evaluate(top_k_list=list(range(1, args.max_k + 1)))
    
    # 保存结果
    recommender.save_results(results, args.output_path)

if __name__ == "__main__":
    main()