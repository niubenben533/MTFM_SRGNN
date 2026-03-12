import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):    
    def __init__(self, hidden_size, step=1):
        """
        定义 GNN 传播的步数（step）。
        定义节点的嵌入维度（hidden_size）。
        初始化 GRU 的参数（w_ih, w_hh, b_ih, b_hh）。
        定义 GNN 计算的线性变换层（linear_edge_in, linear_edge_out, linear_edge_f）。

        """
        # 调用父类的构造函数进行初始化
        super(GNN, self).__init__()
        
        # 初始化步长和隐藏层大小
        self.step = step
        self.hidden_size = hidden_size #100
        
        # 计算输入大小200和门的大小100
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        # 初始化权重和偏置参数，范围是 [-1/sqrt(hidden_size), 1/sqrt(hidden_size)]
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))# 输入门
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))# 隐藏门
        self.b_ih = Parameter(torch.Tensor(self.gate_size))# 门控偏置项
        self.b_hh = Parameter(torch.Tensor(self.gate_size))# 门控偏置项
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))# 输入边偏置项
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))# 输出边偏置项
        
        # 初始化线性层，用于处理输入和输出边的特征
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        
        # 门控机制
    def GNNCell(self, A, hidden):
        """
        通过矩阵计算，这里通过图的邻接矩阵（A）和节点嵌入（hidden），完成信息的聚合与更新,更新节点的特征。GNN核心步骤
        
        参数:
        A: 图的邻接矩阵，用于表示节点之间的连接关系
        hidden: 节点的当前嵌入状态   所有商品的初始嵌入向量
        
        返回:
        hy: 更新后的节点嵌入状态
        """
        # 计算输入门（input gate），通过邻接矩阵的不同部分和隐藏状态的线性变换进行矩阵乘法
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah # A[:, :, :A.shape[1]] 表示入边信息。
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah # A[:, :, A.shape[1]: 2 * A.shape[1]] 表示从当前节点出发的边信息。
        # 将输入门和输出门的结果在特征维度上进行拼接
        inputs = torch.cat([input_in, input_out], 2) # 100*6*200
        # 计算门控机制中的输入门和隐藏门的线性部分
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # 将线性部分的结果拆分重置门（reset gate）、更新门（update gate）、候选状态门（new gate）。
        # i_r, i_i, i_n：分别表示 输入的重置门、更新门、候选状态。
        # h_r, h_i, h_n：分别表示 隐藏状态的重置门、更新门、候选状态。
        i_r, i_i, i_n = gi.chunk(3, 2) # 将 gi 沿着第二个维度（即特征维度）分割成 3 个部分，每个部分的形状为 (batch_size, n_node, hidden_size)。
        h_r, h_i, h_n = gh.chunk(3, 2) # 将 gh 沿着第二个维度（即特征维度）分割成 3 个部分，每个部分的形状为 (batch_size, n_node, hidden_size)。
        # 计算重置门，决定哪些过去的隐藏状态需要被保留
        resetgate = torch.sigmoid(i_r + h_r)  # 重置门（式3）
        # 计算更新门，决定哪些输入门的信息需要被用于更新隐藏状态
        inputgate = torch.sigmoid(i_i + h_i)  # 更新门（式4）
        # 计算候选状态门，结合当前输入和过去的隐藏状态（经过重置门调整）来生成候选隐藏状态
        newgate = torch.tanh(i_n + resetgate * h_n)  # 候选状态（式5）
        # 最终状态计算，结合候选状态和过去的隐藏状态，通过更新门来决定最终的隐藏状态
        hy = newgate + inputgate * (hidden - newgate)  # 最终状态（式6）
        return hy
    def forward(self, A, hidden):
        """
        执行前向传播过程，在图神经网络中更新节点的隐藏状态。
    
        参数:
        A (torch.Tensor): 邻接矩阵，表示节点之间的连接关系。
        hidden (torch.Tensor): 初始隐藏状态，用于开始信息传播过程。
    
        返回:
        torch.Tensor: 经过多个步骤信息传播后的最终隐藏状态。
        """
        # 通过多个步骤更新隐藏状态，每个步骤都使用GNNCell进行信息传播
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        # 返回经过所有步骤后的最终隐藏状态
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        """
        初始化SessionGraph类的构造函数。
        
        参数:
        - opt: 包含模型配置的参数对象，如隐藏层大小、批次大小等。
        - n_node: 图中节点的数量。
        
        此函数负责初始化模型的各种组件和参数，包括嵌入层、图神经网络（GNN）层、线性变换层、损失函数、优化器和学习率调度器。
        """
        # 初始化父类
        super(SessionGraph, self).__init__()        
        # 设置隐藏层大小
        self.hidden_size = opt.hiddenSize       
        # 设置节点数量
        self.n_node = n_node        
        # 设置批次大小
        self.batch_size = opt.batchSize        
        # 设置是否为非混合模式
        self.nonhybrid = opt.nonhybrid        
        # 初始化节点嵌入层
        # 例如，inputs是100*6，embedding后是100*6*100，才能输入到模型中
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)      
        # 初始化图神经网络（GNN）
        self.gnn = GNN(self.hidden_size, step=opt.step)       
        # 初始化线性变换层
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)        
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)        
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)       
        # 初始化用于转换连接的线性变换层
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)        
        # 初始化损失函数
        self.loss_function = nn.CrossEntropyLoss()       
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)       
        # 重置模型参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化或重置模型参数。
        该方法主要用于初始化模型的参数，以确保模型在训练开始时具有合适权重。
        """
        # 计算权重初始化的推荐值，基于隐藏层的大小。
        stdv = 1.0 / math.sqrt(self.hidden_size)
        # 遍历模型的所有可训练参数，进行统一的权重初始化。
        for weight in self.parameters():
            # 在均匀分布中随机采样权重，范围为[-stdv, stdv]。
            weight.data.uniform_(-stdv, stdv)
    #  注意力机制计算
        # 注意力机制计算
    def compute_scores(self, hidden, mask):
        """
        计算注意力分数。
        
        参数:
        hidden: 隐藏状态序列，形状为 (batch_size, seq_length, latent_size)。
        mask: 掩码，形状为 (batch_size, seq_length)，用于区分有效和填充部分。
        
        返回:
        scores: 注意力分数，用于推荐系统中的物品评分预测。
        """
        # 提取每个样本的最后一个有效隐藏状态，作为局部嵌入向量Sl。
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size 最后一个节点,表示局部嵌入向量Sl

        # 计算全局偏好，通过线性变换将最后一个隐藏状态转换为全局偏好表示。
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size 全局偏好

        # 计算当前兴趣，对每个隐藏状态进行线性变换，得到当前兴趣表示。
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size 当前兴趣

        # alpha 表示注意力权重，a 是聚合后的序列全局表示。
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)# 全局嵌入Sg

        # 使用线性层 linear_transform 将全局序列表示与最终节点的特征结合。
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))  # 加权表示与最终节点结合 全局和局部结合

        # 提取所有节点的嵌入，去除占位符节点。
        b = self.embedding.weight[1:]                         # 节点嵌入矩阵

        # 计算最终的注意力分数，通过全局表示和节点嵌入的矩阵乘法得到。
        scores = torch.matmul(a, b.transpose(1, 0))           # 计算分数分布
        return scores

    def forward(self, inputs, A):
        """
        前向传播函数
    
        参数:
        inputs: 输入数据，代表物品的索引序列
        A: 邻接矩阵，描述物品之间的关系图
    
        返回:
        hidden: 经过图神经网络处理后的物品嵌入向量
        """
        # 将物品索引映射到其对应的嵌入向量
        hidden = self.embedding(inputs)
        
        # 使用图神经网络对邻接矩阵A和物品嵌入向量进行处理
        hidden = self.gnn(A, hidden)
        
        # 返回经过图神经网络处理后的嵌入向量
        return hidden



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    """
    定义模型的前向传播过程。
    
    参数:
    - model: 使用的模型实例。
    - i: 当前批次的数据索引。
    - data: 包含训练数据的对象。
    
    返回:
    - targets: 真实标签。
    - compute_scores的结果: 模型的预测分数。
    """
    # 从数据对象中获取当前批次的数据
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    
    # 将数据转换为CUDA张量并传递到设备上
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    # A = trans_to_cuda(torch.tensor(np.array(A), dtype=torch.float32))
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    
    # 从 hidden 中提取 alias_inputs 对应的隐藏状态，并组合成 seq_hidden，用于后续 GNN 计算。
    # 将items和A输入模型，得到隐藏状态、节点嵌入
    hidden = model(items, A)

    # 定义一个lambda函数来获取隐藏状态中与别名输入对应的序列隐藏状态
    get = lambda i: hidden[i][alias_inputs[i]]
    
    # 使用torch.stack将每个别名输入对应的隐藏状态堆叠起来，形成序列隐藏状态张量
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    # 调用模型的compute_scores方法，计算并返回序列的分数
    return targets, model.compute_scores(seq_hidden, mask)


def calculate_ndcg(scores, target, k):
    # 计算NDCG@k
    best_score = 1.0
    actual_score = 0.0
    if len(np.where(scores[:k] == target - 1)[0]) != 0:
        rank = np.where(scores[:k] == target - 1)[0][0] + 1
        actual_score = 1.0 / np.log2(rank + 1)
    return actual_score / best_score

def calculate_precision(scores, target, k):
    # 计算Precision@k
    hits = np.isin(target - 1, scores[:k])
    return hits.sum() / k

def calculate_map(scores, target, k):
    # 计算MAP@k
    if not np.isin(target - 1, scores[:k]):
        return 0
    rank = np.where(scores[:k] == target - 1)[0][0] + 1
    return 1.0 / rank

def train_test(model, train_data, test_data, top_k):
    # 在每个训练 epoch 开始之前，先调用 scheduler.step() 更新学习率，确保优化器使用的是最新的学习率。
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    # 将模型设置为训练模式。
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    # 训练模型
    # i：用于从 train_data 中提取当前批次的数据。
    # j：用于记录当前是第几个批次，可以用于日志记录、调试或其他需要批次编号的场景。
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()  # 清空梯度
        targets, scores = forward(model, i, train_data)  # 前向传播，计算得分
        targets = trans_to_cuda(torch.Tensor(targets).long())  # 目标值转换为 CUDA
        loss = model.loss_function(scores, targets - 1)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        model.optimizer.step()  # 更新参数
        total_loss += loss  # 累加损失

    # 测试模型
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, ndcg, precision, map_score = [], [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    # 预测和评估
    for i in slices:
        targets, scores = forward(model, i, test_data)  # 前向传播
        sub_scores = scores.topk(top_k)[1]  # 取出 top-k 推荐结果
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        # 计算指标
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))  # 计算命中率
            precision.append(calculate_precision(score, target, top_k))
            map_score.append(calculate_map(score, target, top_k))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
                ndcg.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))  # 计算 MRR
                ndcg.append(calculate_ndcg(score, target, top_k))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    ndcg = np.mean(ndcg) * 100
    precision = np.mean(precision) * 100
    map_score = np.mean(map_score) * 100
    return hit, mrr, ndcg, precision, map_score, top_k
    # hit@10 就是“Top-10 命中率”，在单标签推荐任务下，数值等同于 recall@10，但更准确的说法是“命中率”。
    # 单标签推荐任务下，MRR@10 和 MAP@10 数值完全一致。
    # 由于每个 session 只有一个相关物品，所以即使模型表现很好，Precision@10 的理论上限也只有 0.1（即10%），
    # 因为最多只有1个相关物品能被命中，其余9个都是无关的。


