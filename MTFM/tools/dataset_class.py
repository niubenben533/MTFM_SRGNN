# cd /home/sxc/learn/MTFM-main
# python -m tools.dataset_class

from typing import List
import random

import json
import os
import sys
from random import randint, choice

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torchtext.vocab import GloVe, build_vocab_from_iterator

from tools.utils import tokenize

curPath = os.path.abspath(os.path.dirname(__file__))
# 向上两级目录获取项目根目录 (MTFM+SRGNN)
rootPath = os.path.dirname(os.path.dirname(curPath))
sys.path.append(rootPath)

# 加载处理Mashup的数据
class MashupDataset(Dataset):
    def __init__(self, all_api=False):
        super().__init__()
        with open(rootPath + '/data/mashup_name.json', 'r', encoding='utf-8') as f:
            self.name = json.load(f)
        
        with open(rootPath + '/data/mashup_description.json', 'r', encoding='utf-8') as f:
            self.description = json.load(f)
        
        with open(rootPath + '/data/mashup_category.json', 'r', encoding='utf-8') as f:
            self.category = json.load(f)
        
        with open(rootPath + '/data/mashup_used_api.json', 'r', encoding='utf-8') as f:
            self.used_api = json.load(f)
        
        with open(rootPath + '/data/used_api_list.json', 'r', encoding='utf-8') as f:
            self.used_api_list = json.load(f)
            
        # 添加 category_list 的加载
        with open(rootPath + '/data/category_list.json', 'r', encoding='utf-8') as f:
            category_list = json.load(f)

        if all_api: 
            with open(rootPath + '/data/api_name.json', 'r', encoding='utf-8') as f:
                api_list = json.load(f)
        else:
            with open(rootPath + '/data/used_api_list.json', 'r', encoding='utf-8') as f:
                api_list = json.load(f)

        self.num_api = len(api_list)
        self.num_category = len(category_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([api_list])
        self.des_lens = []
        self.category_token = []
        for des in self.description:     # 处理文本长度（截断或填充至50个词）
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    # 返回 Mashup 数据集的大小
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):  # torch.is_tensor()如果传递的对象是PyTorch张量，则方法返回True
            index = index.tolist()  # 返回列表或者数字
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.used_api[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])
        return torch.tensor(index).long(), torch.tensor(
            description).long(), category_tensor, used_api_tensor, des_len, category_token

# 加载处理API的数据
class ApiDataset(Dataset):
    def __init__(self, all_api=False):
        super().__init__()
        with open(rootPath + '/data/api_name.json', 'r', encoding='utf-8') as f:
            name = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r', encoding='utf-8') as f:
            description = json.load(f)
        with open(rootPath + '/data/api_category.json', 'r', encoding='utf-8') as f:
            category = json.load(f)  # 局部变量
        with open(rootPath + '/data/category_list.json', 'r', encoding='utf-8') as f:
            category_list = json.load(f)  # 局部变量
        with open(rootPath + '/data/mashup_name.json', 'r', encoding='utf-8') as f:
            self.mashup = json.load(f)
        with open(rootPath + '/data/used_api_list.json', 'r', encoding='utf-8') as f:
            used_api_list = json.load(f)
        
        if all_api:
            self.name = name
            self.description = description
            self.category = category
            self.used_api = []
            for api in self.name:
                self.used_api.append([api])
        else:
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []
            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        self.num_category = len(category_list)
        self.num_api = len(used_api_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([used_api_list])
        self.des_lens = []
        self.category_token = []
        for des in self.description:       # 处理文本长度（截断或填充至50个词）
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        # 将索引封装成向量形式
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.name[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])

        return torch.tensor(index).long(), \
            torch.tensor(description).long(), \
            category_tensor, \
            used_api_tensor, \
            des_len, category_token

# 用于贝叶斯个性化排名任务，生成正负样本对
class BPRDataset(Dataset):
    def __init__(self, sample_indices, neg_num):
        super(BPRDataset, self).__init__()
        self.ds = TextDataset()
        self.sample_indices = sample_indices
        self.triplet = None
        self.neg_num = neg_num  # 一个正例对应需要采样的负例数量
        self.create_triplet()

    # 创建三元组样本，包括用户、正例API和负例API
    def create_triplet(self):
        pairs = []
        triplet = []
        neg_list = list(range(len(self.ds.api_ds)))
        for sample in self.sample_indices:
            pos_indices = self.ds.mashup_ds[sample][3].nonzero().flatten().tolist()
            for pos in pos_indices:
                pairs.append([sample, pos])
        for pair in pairs:
            break_point = 0
            while True:
                ch = choice(neg_list)
                if break_point == self.neg_num:
                    break
                elif ch != pair[1]:
                    triplet.append((pair[0], pair[1], ch))
                    break_point += 1

        self.triplet = triplet

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        mashup = self.ds.mashup_ds[sample[0]]
        api_i = self.ds.api_ds[sample[1]]
        api_j = self.ds.api_ds[sample[2]]
        return mashup, api_i, api_j           # Mashup样本, 正API样本, 负API样本


# class NNRDataset(Dataset):
#     def __init__(self, nn_num):
#         super(NNRDataset, self).__init__()
#         self.tds = TextDataset()
# 
#         # self.sample_indices = sample_indices
#         self.nn_num = nn_num  # 近邻mashup数量
#         self.sim_matrix = torch.zeros(self.nn_num, self.tds.embed_dim)
#         self.mashup_feature = torch.self.tds.embed[ds.mashup_ds.description].sum(dim=1)
#         self.sim_cal()
#
#     def sim_cal(self):
#         for i, des_list in enumerate(range(self.nn_num)):
#             tmp_ = torch.zeros(300)
#
#             m_feature[i] = torch.nn.functional.normalize(tmp_, dim=0)
#         self.sim_matrix = m_feature.mm(m_feature.t()).argsort(dim=1, descending=True)
#
#     def __len__(self):
#         return len(self.mds)
#
#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         main_mashup = self.tds.mashup_ds[index]
#         nn_mashup_des = torch.zeros(self.nn_num, 50)
#         for count, i in enumerate(self.sim_matrix[index, 1:self.nn_num+1]):
#             nn_mashup_des[count] = self.tds.mashup_ds[i][1]
#
#         return main_mashup, nn_mashup_des.long()

# 整合Mashup和API的数据，构建词汇表和词嵌入矩阵，将文本描述和类别信息转换为数值特征
class TextDataset:
    def __init__(self):
        cache = rootPath + '/vector_cache'    # 创建词向量缓存目录
        if not os.path.exists(cache):
            os.mkdir(cache)

        self.mashup_ds = MashupDataset()
        self.api_ds = ApiDataset()

        self.max_vocab_size = 10000    # 词汇表最大容量
        self.max_doc_len = 50          # 文本截断/填充长度
        self.random_seed = 2020        # 随机种子（当前未使用）

        self.num_category = self.mashup_ds.num_category
        self.num_mashup = len(self.mashup_ds)
        self.num_api = len(self.api_ds)

        all_text = list()
        all_text.extend(self.mashup_ds.description)    # 合并Mashup描述
        all_text.extend(self.api_ds.description)       # 合并API描述

        print('build_vocab...')
        self.vocab = self.build_vocab(all_text, self.max_vocab_size)     # 生成词汇表


        print('\nload Glove word vectors...')
        self.vectors = GloVe(name='6B', dim=300, cache=cache)    # 加载 6B 语料版本的300维GloVe

        print('extract word vectors...')        # 抽取词的子嵌入矩阵
        self.embed = self.vectors.get_vecs_by_tokens(self.vocab.lookup_tokens([i for i in range(len(self.vocab))]))  # 将词汇表映射到对应词向量
        self.embed[0] = torch.mean(self.embed, dim=0)  # 用均值初始化<unk>

        '''output the dimensionality of embeddings'''
        self.vocab_size = self.embed.shape[0]
        self.embed_dim = self.embed.shape[1]
        self.des_lens = []
        self.word2id()         # 文本→索引序列
        self.tag2feature()     # 标签→特征向量

        # print(f"合并文本完成，总样本数: {len(all_text)}")
        # print(f"词向量矩阵形状: {self.embed.shape}")
        # print(f"\n词汇表验证：")
        # print(f"- 总词数: {self.vocab_size}")
        # print(f"- 示例词: {self.vocab.lookup_tokens([0,1,42,100])}")  # 检查特殊符号处理
        # print("\n数据统计：")
        # print(f"Mashup数量: {self.num_mashup} (示例: {self.mashup_ds.name[0]})")
        # print(f"API数量: {self.num_api} (示例: {self.api_ds.name[0]})")

    @staticmethod
    def build_vocab(all_text, max_vocab_size):  # 从所有文本描述中构建词汇表
        def yield_tokens(all_text_):
            for text_ in all_text_:
                # 添加预处理步骤
                tokens = tokenize(str(text_).lower())  # 统一小写
                tokens = [t for t in tokens if t.isalpha()]  # 过滤非字母字符
                yield tokens

        vocab = build_vocab_from_iterator(yield_tokens(all_text), specials=['<unk>'], max_tokens=max_vocab_size) # 词汇表构建，按词频从高到低排序
        vocab.set_default_index(vocab['<unk>'])  # 未登录词统一映射到<unk>
        return vocab

    def word2id(self):  # 将文本描述中的词转换为对应的词汇表id
        print('word2id...')  
        # 处理 Mashup 描述
        for i, des in enumerate(self.mashup_ds.description):
            # Step 1: 词到索引的转换
            tokens = [self.vocab[x] for x in des]    # 核心转换逻辑

            # Step 2: 处理空序列
            if not tokens:
                tokens = [0]     # 使用 <unk> 代替空文本

            # Step 3: 序列长度标准化
            if len(tokens) < self.max_doc_len:
                # 填充 pad 符号（索引为1）
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                # 截断超长文本
                tokens = tokens[:self.max_doc_len]
                # print(tokens)

            # Step 4: 回写处理后的序列
            self.mashup_ds.description[i] = tokens

        # 处理 API 描述
        for i, des in enumerate(self.api_ds.description):
            tokens = [self.vocab[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.api_ds.description[i] = tokens

    def tag2feature(self):  # 将类别信息转换为固定长度的数值特征序列
        print('tag2feature...')
        for i, category in enumerate(self.mashup_ds.category):
            tokens = [self.vocab[x] for x in tokenize(' '.join(category))]
            # tokenize是tools中utils中的函数，其目的是进行词的替换，如缩写，变形
            # vacab中装的是表述api和mashup，中的description
            # print(tokens)

            # 空序列 → [0]（使用 <unk>）
            if not tokens:
                tokens = [0]

            # 长度不足10 → 填充 <pad>（索引1）
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))

            # 超长 → 截断前10个词
            else:
                tokens = tokens[:10]
            self.mashup_ds.category_token.append(tokens)

        for i, category in enumerate(self.api_ds.category):
            tokens = [self.vocab[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.api_ds.category_token.append(tokens)


# class F3RMDataset(Dataset):
#     def __init__(self, nn_num=10):
#         super(F3RMDataset, self).__init__()
#         cache = '.vec_cache'
#         if not os.path.exists(cache):
#             os.mkdir(cache)
#         self.tds = TextDataset()   # 加载文本数据集
#         # self.sample_indices = sample_indices
#         self.nn_num = nn_num  # 近邻mashup数量

#          # 初始化存储近邻描述的3D张量 (总样本数, 近邻数, 文本长度)
#         self.neighbor_mashup_des = torch.zeros(len(self.tds.mashup_ds), self.nn_num, self.tds.max_doc_len)

#          # 计算Mashup特征向量（词向量求和的归一化结果）
#         self.mashup_feature = torch.nn.functional.normalize(self.tds.embed[self.tds.mashup_ds.description].sum(dim=1))

#          # 计算相似度矩阵（余弦相似度）
#         self.sim = torch.nn.functional.normalize(torch.mm(self.mashup_feature, self.mashup_feature.t()))

#         # 获取每个Mashup的前nn_num个最邻近索引
#         self.neighbor_mashup_index = self.sim.argsort(descending=True)[:, :self.nn_num]

#         # 填充近邻描述张量（存在问题）
#         # for i in range(len(self.tds.mashup_ds)):
#         #     for j, index in enumerate(range(self.nn_num)):
#         #         self.neighbor_mashup_des[i, j] = self.tds.mashup_ds[index][1]
#         for i in range(len(self.tds.mashup_ds)):
#             for j in range(self.nn_num):
#                 neighbor_idx = self.neighbor_mashup_index[i][j].item()  # 获取真实邻居索引
#                 self.neighbor_mashup_des[i, j] = self.tds.mashup_ds[neighbor_idx][1]

#     def __len__(self):
#         return len(self.tds.mashup_ds)

#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         main_mashup = self.tds.mashup_ds[index]
#         n_mashup_des = self.neighbor_mashup_des[index]
#         return main_mashup, n_mashup_des.long()
class F3RMDataset(Dataset):
    def __init__(self, nn_num=10):
        super(F3RMDataset, self).__init__()
        # 确保os模块已导入
        import os  
        cache = '.vec_cache'
        os.makedirs(cache, exist_ok=True)  # 自动创建目录
        
        self.tds = TextDataset()
        self.nn_num = nn_num
        
        # 计算余弦相似度
        features = torch.nn.functional.normalize(
            self.tds.embed[self.tds.mashup_ds.description].sum(dim=1), 
            dim=1
        )
        self.sim = torch.mm(features, features.t())  # 余弦相似度矩阵
        
        # 获取近邻索引（排除自身）
        self.neighbor_indices = []
        for i in range(len(self.tds.mashup_ds)):
            sim_scores, indices = torch.topk(self.sim[i], self.nn_num+1)  # 包含自身
            valid_indices = [idx for idx in indices if idx != i][:self.nn_num]
            self.neighbor_indices.append(valid_indices)

    def __len__(self):
        return len(self.tds.mashup_ds)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        main_mashup = self.tds.mashup_ds[index]
        # 动态获取邻居描述
        neighbor_des = [
            self.tds.mashup_ds[idx][1].long() 
            for idx in self.neighbor_indices[index]
        ]
        neighbor_des = torch.stack(neighbor_des)
        
        return main_mashup, neighbor_des


# class FCDataset(Dataset):
#     def __init__(self, sample_indices, is_training=True):
#         super(FCDataset, self).__init__()
#         self.ds = TextDataset()
#         self.triplet = []
#         if is_training:
#             self.neg_num = 14  # 一个正例对应需要采样的负例数量
#             for indice in sample_indices:  # 样本指数
#                 pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
#                 # nonzero 不是0的坐标输出为两维数组 第一行为行标，第二行为列表（返回非0元素目录，返回值元组）
#                 # flatten 是将多维数据降成一维
#                 for pos in pos_indices:
#                     self.triplet.append([indice, pos, 1])
#                 for idx in range(self.neg_num):
#                     r = randint(0, 1646)
#                     if r not in pos_indices:
#                         self.triplet.append([indice, r, -1])
#         else:
#             for indice in sample_indices:
#                 pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
#                 for idx in range(len(self.ds.api_ds)):
#                     if idx in pos_indices:
#                         self.triplet.append([indice, idx, 1])
#                     else:
#                         self.triplet.append([indice, idx, -1])

#     def __len__(self):
#         return len(self.triplet)

#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         sample = self.triplet[index]
#         mashup = self.ds.mashup_ds[sample[0]]
#         api = self.ds.api_ds[sample[1]]
#         label = sample[2]
#         return mashup, api, label
class FCDataset(Dataset):
    def __init__(self, sample_indices: List[int], is_training: bool = True):
        super(FCDataset, self).__init__()
        self.ds = TextDataset()
        self.triplet = []
        self.num_apis = len(self.ds.api_ds)
        
        if is_training:
            self._init_train(sample_indices)
        else:
            self._init_test(sample_indices)

    def _init_train(self, sample_indices):
        """训练集：为每个正样本生成负样本"""
        all_apis = set(range(self.num_apis))
        for mashup_idx in sample_indices:
            # 获取正样本API
            pos_apis = self.ds.mashup_ds[mashup_idx][3].nonzero().flatten().tolist()
            pos_apis_set = set(pos_apis)
            
            # 添加正样本
            for api_idx in pos_apis:
                self.triplet.append((mashup_idx, api_idx, 1))
            
            # 高效负采样
            neg_candidates = list(all_apis - pos_apis_set)
            if not neg_candidates:
                continue
                
            for _ in range(14):  # 可参数化 self.neg_num
                neg_api = random.choice(neg_candidates)
                self.triplet.append((mashup_idx, neg_api, -1))

    def _init_test(self, sample_indices):
        """测试集：生成所有可能的配对（动态计算）"""
        # 不预生成，在__getitem__中动态处理
        self.sample_indices = sample_indices
        self.total_samples = len(sample_indices) * self.num_apis

    def __len__(self):
        return self.total_samples if not self.triplet else len(self.triplet)

    def __getitem__(self, index):
        if self.triplet:  # 训练模式
            mashup_idx, api_idx, label = self.triplet[index]
        else:             # 测试模式（动态计算）
            mashup_idx = self.sample_indices[index // self.num_apis]
            api_idx = index % self.num_apis
            pos_apis = self.ds.mashup_ds[mashup_idx][3].nonzero().flatten().tolist()
            label = 1 if api_idx in pos_apis else -1
        
        mashup = self.ds.mashup_ds[mashup_idx]
        api = self.ds.api_ds[api_idx]
        return mashup, api, torch.tensor(label, dtype=torch.float)


if __name__ == '__main__':
    # mashup_ds = MashupDataset()
    # mashup_ds.__getitem__()
    # api_ds = ApiDataset()
    # ds = F3RMDataset()
    ds = TextDataset()
    # print(ds.build_vocab())
    print(ds.vocab)
    # print(ds.word2id().tokens)

    # # 查看嵌入维度
    # print(f"词向量维度: {ds.embed.shape}")  # 预期输出 (vocab_size, 300)
    # # 示例词向量查询
    # word_idx = ds.vocab["social"]
    # print(f"social 的向量: {ds.embed[word_idx][:5]}")  # 显示前5个维度