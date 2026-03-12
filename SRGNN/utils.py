import networkx as nx
import numpy as np
import os
import sys




# 构建一个有向加权图，用于表示商品之间的点击顺序和权重。
def build_graph(train_data):# 输入：train_data，一个列表，每个元素是用户点击的商品序列。
    graph = nx.DiGraph() 
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph # 输出：图对象 graph，其中节点表示商品，边表示商品之间的点击顺序，权重表示转移频率的归一化值。


# 填充用户点击序列，使其长度一致，并生成相应的掩码。
def data_masks(all_usr_pois, item_tail):
    """
    输入：all_usr_pois：用户点击的商品序列。
          item_tail：填充的商品（通常是 [0]）。
    输出：填充后的序列 us_pois 掩码 us_msks，标识填充部分。
          最大序列长度 len_max。
    """
    # 计算每个用户点击序列的长度。
    us_lens = [len(upois) for upois in all_usr_pois]
    # 找出所有用户点击序列中的最大长度。
    len_max = max(us_lens)
    # 对每个用户点击序列进行填充，使其长度与最大长度一致。
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # 生成掩码，对实际序列部分标记为1，对填充部分标记为0。
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    # 返回填充后的序列、掩码和最大序列长度。
    return us_pois, us_msks, len_max

# 
# 按照比例划分数据集为训练部分和验证部分。
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)
# 将划分好的训练集（包含输入和目标两部分，组成一个元组）和验证集（同样包含输入和目标两部分，组成一个元组）


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        """
        初始化 DataLoader 类。
        
        参数:
        - data: 包含输入数据和目标数据的元组，用于生成批次数据。
        - shuffle: 布尔值，如果为 True，数据会在每次生成批次时被随机打乱，增加训练过程的随机性。
        - graph: 图结构，用于后续构建会话的邻接矩阵。
        """
        # 提取输入数据
        inputs = data[0]
        # 使用 data_masks 函数对输入数据进行处理，生成掩码和最大长度
        inputs, mask, len_max = data_masks(inputs, [0])
        # 将处理后的输入数据转换为 numpy 数组
        self.inputs = np.asarray(inputs)
        # 将生成的掩码转换为 numpy 数组
        self.mask = np.asarray(mask)
        # 保存最大长度
        self.len_max = len_max
        # 将目标数据转换为 numpy 数组
        self.targets = np.asarray(data[1])
        # 计算并保存输入数据的数量
        self.length = len(inputs)
        # 保存是否打乱数据的设置
        self.shuffle = shuffle
        # 保存图结构
        self.graph = graph

    def generate_batch(self, batch_size):
        """
        根据输入的 batch_size 参数计算需要的批次数量。
        
        本函数首先检查是否需要对数据进行洗牌。如果需要，它会对数据集进行随机打乱。
        然后，根据数据集的长度和 batch_size 计算出批次的数量。
        最后，根据计算出的批次数量，将数据集分割成相应数量的批次，并返回这些批次的索引。
        
        参数:
        - batch_size: 每个批次的数据量。
        
        返回:
        - slices: 批次索引，用于提取数据。
        """
        # 检查是否需要对数据进行洗牌
        if self.shuffle:
            # 生成数据索引数组并对其进行洗牌
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # 根据洗牌后的索引重新排列输入数据、mask 和目标数据
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        
        # 根据数据集长度和 batch_size 计算批次数量
        n_batch = int(self.length / batch_size)
        # 如果数据集长度不是 batch_size 的整数倍，增加一个批次以包含剩余数据
        if self.length % batch_size != 0:
            n_batch += 1
        
        # 根据批次数量和 batch_size 计算总数据量，并分割成相应数量的批次
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # 调整最后一个批次的大小，以确保其包含所有剩余数据
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        
        # 返回批次索引
        return slices

    # 提取指定批次的输入、掩码、目标商品等数据。
    def get_slice(self, i):
        """
        根据批次索引i，提取批次输入数据、掩码、目标商品等信息。
        此函数主要用于处理图神经网络中的数据预处理，包括构建节点特征、邻接矩阵等。        
        参数:
        i: 批次索引，用于获取特定批次的数据。        
        返回:
        alias_inputs：重新映射 item ID，方便索引。
        A：邻接矩阵，用于 GNN 计算。
        items：session 内唯一 item 的列表。
        mask：掩码矩阵，用于标记有效 item（填充 0）。
        targets：真实标签（即 session 的下一个 item）。
        """
        # 初始化批次数据
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []        
        # 计算每个输入序列的唯一节点数
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)       
        # 构建每个输入序列的节点列表、邻接矩阵和别名输入序列
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))            
            # 构建邻接矩阵
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1               
            # 计算输入和输出邻接矩阵
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)           
            # 构建别名输入序列
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])     
        # 返回处理后的数据
        # print(f"A的维度: {np.array(A).shape}")
        return alias_inputs, A, items, mask, targets
        # 返回当前批次的索引转换信息 alias_inputs、邻接矩阵列表 A、节点集合列表 items、掩码列表 mask 和目标数据列表 targets。

