# -*- conding: utf-8 -*-
"""
MTFM + SRGNN 融合模型 (基于MTFM.py修改)
融合SRGNN的会话图嵌入与MTFM的U_sc、U_fic嵌入
cd ~/learn/MTFM-main
python -m model.MTFM
"""
import sys
import os
import math

# 获取当前脚本的绝对路径 (MTFM.py)
current_script_path = os.path.abspath(__file__)
# 向上两级目录获取项目根目录 (MTFM)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将根目录添加到Python路径
sys.path.insert(0, project_root)

from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from tools.dataset_class import *
from tools.metric import metric
from tools.utils import *


# ==================== SRGNN组件 ====================
class GNN(nn.Module):
    """图神经网络模块，用于处理API序列的图结构"""
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        # GRU门控参数
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))
        
        # 边特征处理
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
    def GNNCell(self, A, hidden):
        """GNN核心计算单元"""
        # 入边和出边的信息聚合
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        
        # 门控机制
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


class SessionEncoder(nn.Module):
    """会话编码器，使用GNN提取API序列特征"""
    def __init__(self, num_api, hidden_size, step=1):
        super(SessionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_api = num_api
        
        # API嵌入层 (API ID -> hidden_size维向量)
        self.embedding = nn.Embedding(self.num_api + 1, self.hidden_size, padding_idx=0)
        
        # GNN层
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
    
    def compute_session_embedding(self, hidden, mask):
        """计算会话的混合嵌入（全局+局部）"""
        # 局部嵌入：最后一个有效API
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        
        # 计算注意力权重
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        
        # 全局嵌入：加权聚合
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        # 混合嵌入（全局+局部）
        session_emb = self.linear_transform(torch.cat([a, ht], 1))
        return session_emb
    
    def forward(self, items, A, mask):
        """
        Args:
            items: [batch_size, max_seq_len] API ID序列
            A: [batch_size, max_n_node, 2*max_n_node] 邻接矩阵
            mask: [batch_size, max_seq_len] 掩码
        Returns:
            session_emb: [batch_size, hidden_size] 会话嵌入
        """
        # API嵌入
        hidden = self.embedding(items)
        
        # GNN传播
        hidden = self.gnn(A, hidden)
        
        # 计算会话级混合嵌入
        session_emb = self.compute_session_embedding(hidden, mask)
        
        return session_emb


# ==================== 配置类 ====================
class MTFMConfig(object):
    def __init__(self, ds_config):
        self.model_name = 'MTFM_SRGNN_Fusion'
        
        # MTFM相关参数
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.feature_dim = 8
        self.num_kernel = 256
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        
        # SRGNN相关参数
        self.use_srgnn = True  # 是否启用SRGNN组件
        self.gnn_hidden_size = 100  # SRGNN隐藏层维度
        self.gnn_step = 1  # GNN传播步数
        
        # 训练参数
        self.lr = 1e-3
        self.batch_size = 128
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')


class MTFM(nn.Module):
    """融合MTFM和SRGNN的API推荐模型"""
    def __init__(self, config):
        super(MTFM, self).__init__()
        self.config = config

        # ============ MTFM文本嵌入层 ============
        # 嵌入层：将输入的文本描述（词索引序列）转换为稠密词向量
        if config.embed is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, 
                                         padding_idx=config.vocab_size - 1)

        # ============ 语义组件 (Semantic Component) ============
        # 使用多尺度CNN提取文本的语义特征
        self.sc_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config.embed_dim,
                         out_channels=config.num_kernel,
                         kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=config.max_doc_len - h + 1))
            for h in config.kernel_size
        ])
        self.sc_fcl = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                               out_features=config.num_api)

        # ============ 特征交互组件 (Feature Interaction Component) ============
        # 建模Mashup和API之间的特征交互
        self.fic_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                               out_features=config.feature_dim)
        self.fic_api_feature_embedding = nn.Parameter(torch.rand(config.feature_dim, config.num_api))
        self.fic_mlp = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.feature_dim),
            nn.Linear(config.feature_dim, 1),
            nn.Tanh()
        )
        self.fic_fcl = nn.Linear(config.num_api * 2, config.num_api)

        # ============ SRGNN组件 ============
        if config.use_srgnn:
            # 会话编码器：从API序列中提取图结构特征
            self.session_encoder = SessionEncoder(
                num_api=config.num_api,
                hidden_size=config.gnn_hidden_size,
                step=config.gnn_step
            )
            # 投影层：将SRGNN的hidden_size映射到num_api维度
            self.srgnn_projection = nn.Linear(config.gnn_hidden_size, config.num_api)

        # ============ 融合层 ============
        # 根据是否使用SRGNN决定融合维度
        if config.use_srgnn:
            # 三路融合: U_sc + U_fic + U_srgnn -> U_mmf
            self.fusion_layer = nn.Linear(config.num_api * 3, config.num_api)
        else:
            # 两路融合: U_sc + U_fic -> U_mmf (原MTFM)
            self.fusion_layer = nn.Linear(config.num_api * 2, config.num_api)

        # ============ 任务层 ============
        # API推荐任务层
        self.api_task_layer = nn.Linear(config.num_api, config.num_api)

        # ============ 激活函数和正则化 ============
        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def init_weight(self):
        nn.init.kaiming_normal_(self.fic_api_feature_embedding)

    def forward(self, mashup_des, api_seq=None, api_adj=None, api_mask=None):
        """
        前向传播：融合MTFM和SRGNN的嵌入
        
        Args:
            mashup_des: [batch_size, max_doc_len] Mashup文本描述（词索引序列）
            api_seq: [batch_size, max_seq_len] API序列（可选，用于SRGNN）
            api_adj: [batch_size, max_n_node, 2*max_n_node] 邻接矩阵（可选）
            api_mask: [batch_size, max_seq_len] 序列掩码（可选）
            
        Returns:
            [batch_size, num_api] API推荐分数
        """
        # ======== MTFM部分：提取U_sc和U_fic ========
        
        # 1. 语义组件 (Semantic Component) -> U_sc
        embed = self.embedding(mashup_des)  # [batch_size, max_doc_len, embed_dim]
        embed = embed.permute(0, 2, 1)  # [batch_size, embed_dim, max_doc_len]
        
        # 多尺度卷积
        e = [conv(embed) for conv in self.sc_convs]  # 列表，每个元素 [batch_size, num_kernel, 1]
        e = torch.cat(e, dim=2)  # [batch_size, num_kernel, len(kernel_size)]
        e = e.view(e.size(0), -1)  # [batch_size, num_kernel * len(kernel_size)]
        
        u_sc = self.sc_fcl(e)  # [batch_size, num_api] 语义嵌入
        
        # 2. 特征交互组件 (Feature Interaction Component) -> U_fic
        u_sc_trans = self.fic_fc(e)  # [batch_size, feature_dim]
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)  # [batch_size, num_api]
        
        # MLP交互
        u_concate = []
        for u_sc_single in u_sc_trans:
            u_concate_single = torch.cat(
                (u_sc_single.repeat(self.fic_api_feature_embedding.size(1), 1),
                 self.fic_api_feature_embedding.t()),
                dim=1)
            u_concate.append(self.fic_mlp(u_concate_single).squeeze())
        u_mlp = torch.cat(u_concate).view(u_mm.size(0), -1)  # [batch_size, num_api]
        
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp), dim=1))  # [batch_size, num_api]
        u_fic = self.tanh(u_fic)  # 特征交互嵌入
        
        # ======== SRGNN部分：提取U_srgnn (如果启用) ========
        if self.config.use_srgnn and api_seq is not None:
            # 3. 会话图编码器 -> session_emb
            session_emb = self.session_encoder(api_seq, api_adj, api_mask)  # [batch_size, gnn_hidden_size]
            
            # 投影到num_api维度
            u_srgnn = self.srgnn_projection(session_emb)  # [batch_size, num_api]
            u_srgnn = self.tanh(u_srgnn)  # SRGNN嵌入
            
            # 三路融合: U_sc + U_fic + U_srgnn -> U_mmf
            u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic, u_srgnn), dim=1))
        else:
            # 两路融合: U_sc + U_fic -> U_mmf (退化为原MTFM)
            u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))
        
        # ======== 下游任务 ========
        # Dropout正则化
        u_mmf = self.dropout(u_mmf)
        
        # API推荐任务层
        y_m = self.api_task_layer(u_mmf)  # [batch_size, num_api]
        
        return self.logistic(y_m)


# ==================== 数据处理工具函数 ====================
def build_session_graph(api_sequence, num_api):
    """
    将API序列转换为会话图的邻接矩阵
    
    Args:
        api_sequence: [seq_len] API ID列表（从1开始编号）
        num_api: API总数
        
    Returns:
        items: unique API列表（包含padding后的节点ID）
        A: 邻接矩阵 [n_node, 2*n_node]
        mask: 序列掩码（用于标识有效位置）
    """
    # 过滤掉0（padding），API ID从1开始
    api_sequence = [api_id for api_id in api_sequence if api_id > 0]
    
    if len(api_sequence) == 0:
        # 返回空会话：单个padding节点
        return [0], np.zeros((1, 2)), [0]
    
    # 获取唯一节点（保持顺序）
    node = np.unique(api_sequence)
    items = node.tolist()
    n_node = len(node)
    
    # 构建邻接矩阵（有向图）
    A = np.zeros((n_node, n_node))
    
    for i in range(len(api_sequence) - 1):
        # 找到当前API和下一个API在唯一节点列表中的位置
        u = np.where(node == api_sequence[i])[0][0]
        v = np.where(node == api_sequence[i + 1])[0][0]
        A[u][v] = 1
    
    # 归一化入边（列归一化）
    u_sum_in = np.sum(A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    A_in = np.divide(A, u_sum_in)
    
    # 归一化出边（行归一化）
    u_sum_out = np.sum(A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    A_out = np.divide(A.transpose(), u_sum_out)
    
    # 拼接入边和出边矩阵 [n_node, 2*n_node]
    A = np.concatenate([A_in, A_out]).transpose()
    
    # 创建序列级别的掩码（对应原始序列长度）
    mask = [1] * len(api_sequence)
    
    return items, A, mask


def collate_fn_with_graph(batch, num_api):
    """
    自定义collate函数，处理变长序列并构建图
    
    数据对齐策略：
    - 从Mashup的used_api字段提取API序列
    - API ID需要+1，因为SRGNN中0是padding
    - 构建每个batch的邻接矩阵和掩码
    
    Args:
        batch: list of (index, des, category, used_api, des_len, category_token)
        num_api: API总数
        
    Returns:
        (indices, descriptions, categories, used_apis, items_tensor, A_tensor, mask_tensor)
    """
    indices, descriptions, categories, used_apis, des_lens, category_tokens = zip(*batch)
    
    # 转换为tensor (MTFM部分)
    indices = torch.stack(indices)
    descriptions = torch.stack(descriptions)
    categories = torch.stack(categories)
    used_apis = torch.stack(used_apis)
    
    # 构建API序列和图结构 (SRGNN部分)
    batch_items = []
    batch_A = []
    batch_mask = []
    
    max_n_node = 0
    max_seq_len = 0
    
    for used_api in used_apis:
        # 获取使用的API索引列表（从one-hot编码）
        api_indices = used_api.nonzero().squeeze()
        
        # 处理单个API和多个API的情况
        if api_indices.dim() == 0:  # 单个API
            api_ids = [api_indices.item() + 1]  # +1因为SRGNN中0是padding
        else:  # 多个API
            api_ids = [idx.item() + 1 for idx in api_indices]
        
        # 如果没有API，使用空列表
        if len(api_ids) == 0:
            api_ids = []
        
        # 构建会话图
        items, A, mask = build_session_graph(api_ids, num_api)
        
        batch_items.append(items)
        batch_A.append(A)
        batch_mask.append(mask)
        
        max_n_node = max(max_n_node, len(items))
        max_seq_len = max(max_seq_len, len(mask))
    
    # 确保至少有一个节点（避免空batch问题）
    if max_n_node == 0:
        max_n_node = 1
    if max_seq_len == 0:
        max_seq_len = 1
    
    # Padding
    padded_items = []
    padded_A = []
    padded_mask = []
    
    for items, A, mask in zip(batch_items, batch_A, batch_mask):
        # Pad items (节点列表)
        padded_item = items + [0] * (max_n_node - len(items))
        padded_items.append(padded_item)
        
        # Pad adjacency matrix [max_n_node, 2*max_n_node]
        padded_a = np.zeros((max_n_node, max_n_node * 2))
        if A.shape[0] > 0 and A.shape[1] > 0:
            padded_a[:A.shape[0], :A.shape[1]] = A
        padded_A.append(padded_a)
        
        # Pad mask (序列掩码)
        padded_m = mask + [0] * (max_seq_len - len(mask))
        padded_mask.append(padded_m)
    
    # 转换为tensor
    items_tensor = torch.LongTensor(padded_items)
    A_tensor = torch.FloatTensor(padded_A)
    mask_tensor = torch.LongTensor(padded_mask)
    
    return indices, descriptions, categories, used_apis, items_tensor, A_tensor, mask_tensor


# ==================== 训练类 ====================
class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, case_iter, log, input_ds,
                 model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.api_cri = torch.nn.BCELoss()
        # 移除类别损失
        # self.cate_cri = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.epoch = 100
        self.top_k_list = list(range(1, 31))
        self.log = log
        self.ds = input_ds
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(project_root, 'checkpoint', f'{self.config.model_name}.pth')
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)

    def train(self):
        """训练方法：融合MTFM和SRGNN的训练过程"""
        data_iter = self.train_iter
        self.model.train()
        print('开始训练融合模型...')
        print(f'  - 使用SRGNN: {self.config.use_srgnn}')
        print(f'  - 设备: {self.config.device}')

        for epoch in range(self.epoch):
            api_loss = []

            for batch_idx, batch_data in enumerate(data_iter):
                # 解包数据：包含MTFM和SRGNN所需的输入
                # batch_data: (index, des, category, used_api, items, A, mask)
                index, des, category, api_target, items, A, mask = batch_data
                
                # 移动到设备
                des = des.to(self.config.device)
                api_target = api_target.float().to(self.config.device)
                items = items.to(self.config.device)
                A = A.to(self.config.device)
                mask = mask.to(self.config.device)

                # 前向传播
                self.optim.zero_grad()
                if self.config.use_srgnn:
                    # 融合模式：同时传入文本和API序列
                    api_pred = self.model(des, items, A, mask)
                else:
                    # 原MTFM模式：仅使用文本
                    api_pred = self.model(des)

                # 计算损失
                api_loss_ = self.api_cri(api_pred, api_target)
                api_loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())

            api_loss = np.average(api_loss)

            info = '[Epoch:%s] ApiLoss:%s' % (epoch + 1, api_loss.round(6))
            print(info)
            self.log.write(info + '\n')
            self.log.flush()
            
            val_loss = self.evaluate()
            self.early_stopping(float(val_loss), self.model)

            if self.early_stopping.early_stop:
                print("早停触发")
                break

    def evaluate(self, test=False):
        """评估方法：在验证集或测试集上评估融合模型的性能"""
        if test:
            data_iter = self.test_iter
            label = 'Test'
            print('开始测试...')
        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        
        self.model.eval()

        # 仅计算 API 指标
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        api_loss = []

        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                # 解包数据
                index, des, category, api_target, items, A, mask = batch_data
                
                # 移动到设备
                des = des.to(self.config.device)
                api_target_gpu = api_target.float().to(self.config.device)
                items = items.to(self.config.device)
                A = A.to(self.config.device)
                mask = mask.to(self.config.device)

                # 前向传播
                if self.config.use_srgnn:
                    api_pred = self.model(des, items, A, mask)
                else:
                    api_pred = self.model(des)
                    
                api_loss_ = self.api_cri(api_pred, api_target_gpu)
                api_loss.append(api_loss_.item())

                api_pred = api_pred.cpu().detach()

                # 计算评估指标（使用CPU上的api_target）
                ndcg_, recall_, ap_, pre_ = metric(api_target, api_pred, top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_

        api_loss = np.average(api_loss)
        ndcg_a /= num_batch
        recall_a /= num_batch
        ap_a /= num_batch
        pre_a /= num_batch

        info = '[%s] ApiLoss:%s\n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s' % (
                   label, api_loss.round(6), ndcg_a.round(6), ap_a.round(6), pre_a.round(6), recall_a.round(6))

        print(info)
        self.log.write(info + '\n')
        self.log.flush()
        return api_loss

        # category_performance
        ndcg_c = np.zeros(len(self.top_k_list))
        recall_c = np.zeros(len(self.top_k_list))
        ap_c = np.zeros(len(self.top_k_list))
        pre_c = np.zeros(len(self.top_k_list))

        api_loss = []
        category_loss = []

        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)
                api_pred, category_pred = self.model(des)
                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.cate_cri(category_pred, category_target)
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

                api_pred = api_pred.cpu().detach()
                category_pred = category_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred.cpu(), top_k_list=self.top_k_list)
                ndcg_c += ndcg_
                recall_c += recall_
                ap_c += ap_
                pre_c += pre_

        api_loss = np.average(api_loss)
        category_loss = np.average(category_loss)

        ndcg_a /= num_batch
        recall_a /= num_batch
        ap_a /= num_batch
        pre_a /= num_batch
        ndcg_c /= num_batch
        recall_c /= num_batch
        ap_c /= num_batch
        pre_c /= num_batch

        info = '[%s] ApiLoss:%s CateLoss:%s\n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' \
               'NDCG_C:%s\n' \
               'AP_C:%s\n' \
               'Pre_C:%s\n' \
               'Recall_C:%s' % (
                   label, api_loss.round(6), category_loss.round(6), ndcg_a.round(6), ap_a.round(6), pre_a.round(6),
                   recall_a.round(6), ndcg_c.round(6), ap_c.round(6), pre_c.round(6), recall_c.round(6))

        print(info)
        self.log.write(info + '\n')
        self.log.flush()
        return api_loss + category_loss

    def case_analysis(self):  # 案例分析方法：对模型的预测结果进行案例分析，保存预测结果和真实标签
        case_dir = os.path.join(project_root, 'case')
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        case_path = os.path.join(case_dir, f'{config.model_name}.json')
        a_case = open(case_path, mode='w')
        api_case = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.case_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                api_target = []
                for api_data in batch_data[3]:
                    if isinstance(api_data.nonzero().squeeze().tolist(), list):
                        api_target.append(api_data.nonzero().squeeze().tolist())
                    else:
                        api_target.append([api_data.nonzero().squeeze().tolist()])
                # 仅保留 API 预测
                api_pred_ = self.model(des)
                api_pred_ = api_pred_.cpu().argsort(descending=True)[:, :5].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred_)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    api_case.append((name, target, pred))
        json.dump(api_case, a_case)
        a_case.close()


if __name__ == '__main__':
    print('='*60)
    print('MTFM + SRGNN 融合模型')
    print('='*60)
    
    # 加载数据集
    print('加载数据集...')
    start_time = time.time()
    now = time.time()
    ds = TextDataset()
    print('数据集加载完成，耗时: ', get_time(now))

    # 获取训练、验证、测试和案例分析的索引
    train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)

    # 初始化模型配置和模型
    config = MTFMConfig(ds)
    print(f'\n模型配置:')
    print(f'  - 模型名称: {config.model_name}')
    print(f'  - 使用SRGNN: {config.use_srgnn}')
    print(f'  - GNN隐藏层维度: {config.gnn_hidden_size}')
    print(f'  - API数量: {config.num_api}')
    print(f'  - 设备: {config.device}')
    
    model = MTFM(config)
    model.to(config.device)

    # 创建数据加载器（使用自定义collate函数）
    train_iter = DataLoader(
        ds.mashup_ds, 
        batch_size=config.batch_size, 
        sampler=SubsetRandomSampler(train_idx),
        collate_fn=lambda batch: collate_fn_with_graph(batch, config.num_api)
    )
    val_iter = DataLoader(
        ds.mashup_ds, 
        batch_size=len(val_idx), 
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=lambda batch: collate_fn_with_graph(batch, config.num_api)
    )
    test_iter = DataLoader(
        ds.mashup_ds, 
        batch_size=len(test_idx), 
        sampler=SubsetRandomSampler(test_idx),
        collate_fn=lambda batch: collate_fn_with_graph(batch, config.num_api)
    )
    case_iter = DataLoader(
        ds.mashup_ds, 
        batch_size=len(ds.mashup_ds),
        collate_fn=lambda batch: collate_fn_with_graph(batch, config.num_api)
    )

    # 创建带时间戳的日志文件
    log_dir = os.path.join(project_root, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f'{config.model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    log_path = os.path.join(log_dir, log_filename)
    log = open(log_path, mode='w', encoding='utf-8')
    log.write(f'日志开始于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    log.write(f'模型: {config.model_name}\n')
    log.write(f'使用SRGNN: {config.use_srgnn}\n')
    log.flush()

    # 初始化训练类
    # model_path = 'checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=config.device))
    train_func = Train(input_model=model,
                       input_config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       case_iter=case_iter,
                       log=log,
                       input_ds=ds)
    # 训练
    train_func.train()

    # 测试
    print('\n' + '='*60)
    print('最终测试')
    print('='*60)
    train_func.evaluate(test=True)

    # 案例分析
    print('\n生成案例分析...')
    train_func.case_analysis()
    
    log.close()
    print('\n训练完成！')
    print(f'日志保存在: {log_path}')
