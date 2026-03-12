"""
SR-GNN推荐算法 - 纯序列基线（剔除MTFM文本信息）
基于会话图神经网络的推荐算法，仅使用序列信息，不包含文本特征
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import math
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader

# 添加SR_GNN_NEW路径以导入评估指标
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SR_GNN_NEW'))
from tools.metric import ndcg, recall, precision, ap

class GNN(nn.Module):
    """图神经网络模块"""
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SRGNNModel(nn.Module):
    """纯SR-GNN模型（不包含MTFM特征）"""
    def __init__(self, n_node, hidden_size, step=1, nonhybrid=False):
        super(SRGNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.nonhybrid = nonhybrid
        
        # 嵌入层
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        
        # 图神经网络
        self.gnn = GNN(self.hidden_size, step=step)
        
        # 注意力机制
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """计算推荐分数"""
        # 获取最后一个有效位置的隐藏状态
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        
        # 注意力机制
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        # 混合表示
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        
        # 计算分数
        b = self.embedding.weight[1:]  # 排除padding
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        """前向传播"""
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

class SRGNNDataset(Dataset):
    """SR-GNN数据集"""
    def __init__(self, sessions, api_to_id):
        self.sessions = sessions
        self.api_to_id = api_to_id
        self.data = []
        self.prepare_data()
    
    def prepare_data(self):
        """准备训练数据"""
        for session in self.sessions:
            if len(session) < 2:
                continue
            
            # 为每个会话创建训练样本
            for i in range(1, len(session)):
                input_seq = session[:i+1]
                target = session[i]
                
                # 转换为ID
                input_ids = []
                for api in input_seq:
                    if api in self.api_to_id:
                        input_ids.append(self.api_to_id[api])
                
                if len(input_ids) >= 2 and target in self.api_to_id:
                    self.data.append({
                        'inputs': input_ids,
                        'target': self.api_to_id[target]
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """批处理函数"""
    max_len = max(len(item['inputs']) for item in batch)
    
    inputs = []
    targets = []
    masks = []
    
    for item in batch:
        input_seq = item['inputs']
        # 填充序列
        padded = input_seq + [0] * (max_len - len(input_seq))
        inputs.append(padded)
        targets.append(item['target'])
        
        # 创建mask
        mask = [1] * len(input_seq) + [0] * (max_len - len(input_seq))
        masks.append(mask)
    
    # 构建邻接矩阵
    adjacency = build_adjacency_matrix(inputs)
    
    return {
        'inputs': torch.LongTensor(inputs),
        'targets': torch.LongTensor(targets),
        'masks': torch.LongTensor(masks),
        'adjacency': torch.FloatTensor(adjacency)
    }

def build_adjacency_matrix(sessions):
    """构建邻接矩阵"""
    batch_size = len(sessions)
    max_len = len(sessions[0])
    
    # 创建邻接矩阵 [batch_size, max_len, 2*max_len]
    adjacency = np.zeros((batch_size, max_len, 2 * max_len))
    
    for i, session in enumerate(sessions):
        for j in range(len(session) - 1):
            if session[j] != 0 and session[j+1] != 0:  # 忽略padding
                # 出边
                adjacency[i][j][j+1] = 1
                # 入边
                adjacency[i][j+1][max_len + j] = 1
    
    return adjacency

class SRGNNRecommender:
    """SR-GNN推荐器"""
    
    def __init__(self, data_path, hidden_size=100, step=1, nonhybrid=False,
                 learning_rate=0.001, batch_size=32, epochs=10):
        self.data_path = data_path
        self.hidden_size = hidden_size
        self.step = step
        self.nonhybrid = nonhybrid
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 数据
        self.api_list = []
        self.api_to_id = {}
        self.id_to_api = {}
        self.train_sessions = []
        self.test_sessions = []
        self.api_popularity = {}
        
        # 模型
        self.model = None
        
        # 加载数据并训练模型
        self.load_data()
        self.train_model()
        
    def load_data(self):
        """加载数据"""
        # 加载API列表
        with open(os.path.join(self.data_path, 'used_api_list.json'), 'r', encoding='utf-8') as f:
            self.api_list = json.load(f)
        
        # 构建API映射
        self.api_to_id = {api: idx + 1 for idx, api in enumerate(self.api_list)}  # 从1开始，0用于padding
        self.id_to_api = {idx + 1: api for idx, api in enumerate(self.api_list)}
        
        # 加载训练数据
        with open(os.path.join(self.data_path, 'train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            if isinstance(train_data, dict) and 'sequences' in train_data:
                sequences = train_data['sequences']
            else:
                sequences = train_data
            
            # 将API ID转换为API名称
            self.train_sessions = []
            for session in sequences:
                if len(session) >= 2:
                    api_session = []
                    for api_id in session:
                        if api_id < len(self.api_list):
                            api_session.append(self.api_list[api_id])
                    if len(api_session) >= 2:
                        self.train_sessions.append(api_session)
        
        # 加载测试数据
        with open(os.path.join(self.data_path, 'test.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            if isinstance(test_data, dict) and 'sequences' in test_data:
                sequences = test_data['sequences']
            else:
                sequences = test_data
            
            # 将API ID转换为API名称
            self.test_sessions = []
            for session in sequences:
                if len(session) >= 2:
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
    
    def train_model(self):
        """训练模型"""
        # 创建数据集
        train_dataset = SRGNNDataset(self.train_sessions, self.api_to_id)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
        
        # 创建模型
        n_node = len(self.api_list) + 1  # +1 for padding
        self.model = SRGNNModel(
            n_node=n_node,
            hidden_size=self.hidden_size,
            step=self.step,
            nonhybrid=self.nonhybrid
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("开始训练SR-GNN模型...")
        
        # 训练循环
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                masks = batch['masks'].to(self.device)
                adjacency = batch['adjacency'].to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                hidden = self.model(inputs, adjacency)
                scores = self.model.compute_scores(hidden, masks)
                
                # 计算损失
                loss = criterion(scores, targets - 1)  # targets从1开始，需要减1
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        print("模型训练完成!")
    
    def recommend(self, session_sequence, k=20):
        """为给定会话序列生成推荐"""
        if not session_sequence:
            return self.get_popular_recommendations(k)
        
        self.model.eval()
        with torch.no_grad():
            # 转换为ID
            input_ids = []
            for api in session_sequence:
                if api in self.api_to_id:
                    input_ids.append(self.api_to_id[api])
            
            if len(input_ids) < 1:
                return self.get_popular_recommendations(k)
            
            # 准备输入
            inputs = torch.LongTensor([input_ids]).to(self.device)
            masks = torch.LongTensor([[1] * len(input_ids)]).to(self.device)
            
            # 构建邻接矩阵
            adjacency = build_adjacency_matrix([input_ids])
            adjacency = torch.FloatTensor(adjacency).to(self.device)
            
            # 前向传播
            hidden = self.model(inputs, adjacency)
            scores = self.model.compute_scores(hidden, masks)
            
            # 获取推荐
            scores = scores.cpu().numpy()[0]
            
            # 排除已经在会话中的API
            exclude_ids = set(input_ids) - {0}  # 排除padding
            for api_id in exclude_ids:
                if api_id - 1 < len(scores):  # api_id从1开始
                    scores[api_id - 1] = -float('inf')
            
            # 获取top-k推荐
            top_indices = np.argsort(scores)[::-1][:k]
            recommendations = []
            
            for idx in top_indices:
                api_id = idx + 1  # 转换回原始ID
                if api_id in self.id_to_api:
                    recommendations.append(self.id_to_api[api_id])
            
            # 如果推荐不足，用流行度补充
            if len(recommendations) < k:
                popular_apis = self.get_popular_recommendations(k - len(recommendations), 
                                                              exclude=set(session_sequence + recommendations))
                recommendations.extend(popular_apis)
            
            return recommendations[:k]
    
    def get_popular_recommendations(self, k, exclude=None):
        """基于流行度的推荐"""
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
        """评估模型性能"""
        print("开始评估...")
        
        results = {}
        num_evaluated = 0
        
        for top_k in top_k_list:
            ndcg_scores = []
            recall_scores = []
            precision_scores = []
            ap_scores = []
            
            for session in self.test_sessions:
                if len(session) < 2:
                    continue
                
                # 使用前面的API作为输入，最后一个作为目标
                input_sequence = session[:-1]
                target_api = session[-1]
                
                # 生成推荐
                recommendations = self.recommend(input_sequence, k=top_k)
                
                # 计算指标
                if target_api in recommendations:
                    target_pos = recommendations.index(target_api) + 1
                    
                    # 修正指标函数调用 - 只传入target和pred两个参数
                    ndcg_scores.append(ndcg([target_api], recommendations))
                    recall_scores.append(recall([target_api], recommendations))
                    precision_scores.append(precision([target_api], recommendations))
                    ap_scores.append(ap([target_api], recommendations))
                else:
                    ndcg_scores.append(0.0)
                    recall_scores.append(0.0)
                    precision_scores.append(0.0)
                    ap_scores.append(0.0)
                
                num_evaluated += 1
            
            # 计算平均值
            results[top_k] = {
                'NDCG': np.mean(ndcg_scores) if ndcg_scores else 0.0,
                'Recall': np.mean(recall_scores) if recall_scores else 0.0,
                'Precision': np.mean(precision_scores) if precision_scores else 0.0,
                'MAP': np.mean(ap_scores) if ap_scores else 0.0
            }
            
            print(f"Top-{top_k}: NDCG={results[top_k]['NDCG']:.4f}, "
                  f"Recall={results[top_k]['Recall']:.4f}, "
                  f"Precision={results[top_k]['Precision']:.4f}, "
                  f"MAP={results[top_k]['MAP']:.4f}")
        
        # 添加评估的会话数量
        results['num_evaluated'] = num_evaluated
        
        return results
    
    def save_results(self, results, output_path):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式
        json_file = os.path.join(output_path, f"srgnn_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式
        csv_file = os.path.join(output_path, f"srgnn_results_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Top-K,NDCG,Recall,Precision,MAP\n")
            for k in sorted(results.keys()):
                if isinstance(k, int):
                    f.write(f"{k},{results[k]['NDCG']:.6f},{results[k]['Recall']:.6f},"
                           f"{results[k]['Precision']:.6f},{results[k]['MAP']:.6f}\n")
        
        # 保存详细报告
        report_file = os.path.join(output_path, f"srgnn_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SR-GNN基线算法评估报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"评估会话数: {results.get('num_evaluated', 'N/A')}\n")
            f.write(f"隐藏层大小: {self.hidden_size}\n")
            f.write(f"GNN步数: {self.step}\n")
            f.write(f"批次大小: {self.batch_size}\n")
            f.write(f"训练轮数: {self.epochs}\n")
            f.write(f"学习率: {self.learning_rate}\n")
            f.write("\n性能指标:\n")
            f.write("-" * 30 + "\n")
            
            for k in sorted(results.keys()):
                if isinstance(k, int):
                    f.write(f"Top-{k:2d}: NDCG={results[k]['NDCG']:.6f}, "
                           f"Recall={results[k]['Recall']:.6f}, "
                           f"Precision={results[k]['Precision']:.6f}, "
                           f"MAP={results[k]['MAP']:.6f}\n")
        
        print(f"结果已保存到: {output_path}")
        return json_file, csv_file, report_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SR-GNN基线算法')
    parser.add_argument('--data_path', type=str, 
                       default='e:/data/merge/SR_GNN_NEW/data',
                       help='数据文件夹路径')
    parser.add_argument('--output_path', type=str,
                       default='e:/data/merge/基线算法/srgnn_results',
                       help='输出文件夹路径')
    parser.add_argument('--hidden_size', type=int, default=100,
                       help='隐藏层大小')
    parser.add_argument('--step', type=int, default=1,
                       help='GNN传播步数')
    parser.add_argument('--nonhybrid', action='store_true',
                       help='是否使用非混合模式')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    print("SR-GNN基线算法开始运行...")
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    
    # 创建推荐器
    recommender = SRGNNRecommender(
        data_path=args.data_path,
        hidden_size=args.hidden_size,
        step=args.step,
        nonhybrid=args.nonhybrid,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # 评估
    results = recommender.evaluate()
    
    # 保存结果
    recommender.save_results(results, args.output_path)
    
    print("SR-GNN基线算法运行完成!")

if __name__ == "__main__":
    main()