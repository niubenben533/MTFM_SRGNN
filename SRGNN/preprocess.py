import json
import pickle
import os
import pandas as pd
import pickle
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import build_graph, Data, split_validation

# 文件路径
api_list_path = 'data/used_api_list.json'
mashup_used_api_path = 'data/mashup_used_api.json'

# 读取API列表
with open(api_list_path, 'r', encoding='utf-8') as f:
    api_list = json.load(f)

# 创建API名称到ID的映射（从1开始编号）
api_to_id = {api: idx+1 for idx, api in enumerate(api_list)}

# 保存API到ID的映射，方便后续使用
with open('data/api_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(api_to_id, f, indent=2)

print(f"已为{len(api_list)}个API创建从1开始的编号映射")

# 读取mashup使用的API列表
with open(mashup_used_api_path, 'r', encoding='utf-8') as f:
    mashup_used_api = json.load(f)

# 创建二维列表，用编号表示API
mashups_df = []
for api_list in mashup_used_api:
    # 将API名称转换为ID
    api_ids = [api_to_id[api] for api in api_list if api in api_to_id]
    # 只保留至少有两个API的mashup
    if len(api_ids) >= 2:
        mashups_df.append(api_ids)

print(f"处理后的Mashup数量: {len(mashups_df)}")

# 创建DataFrame以便于处理
train_df = pd.DataFrame({
    'mashups_name': [f'mashup_{i}' for i in range(len(mashups_df))],
    'API_ID': mashups_df
})

# 数据集拆分
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# 为原始训练集和测试集添加原始mashup ID（在筛选之前）
train_df = train_df.reset_index(drop=False).rename(columns={'index': 'original_mashup_id'})
test_df = test_df.reset_index(drop=False).rename(columns={'index': 'original_mashup_id'})

# 只对训练集进行重排列组合
x = 6
print(f"对训练集中API数量小于等于{x}个的mashup进行重新排列组合")
small_train = train_df[train_df['API_ID'].apply(len) <= x]

# 生成新mashup时保持原始mashup ID的追踪
new_mashups = []
for idx, row in small_train.iterrows():
    api_ids = row['API_ID']
    original_id = row['original_mashup_id']
    perms = list(itertools.permutations(api_ids))
    for i, perm in enumerate(perms[1:], 1):
        new_mashups.append({
            'mashups_name': f"{row['mashups_name']}_perm{i}",
            'API_ID': list(perm),
            'original_mashup_id': original_id
        })

print(f"生成了{len(new_mashups)}个新的训练集mashup组合")

# 将新生成的mashup添加到训练集
new_mashups_df = pd.DataFrame(new_mashups)
train_df = pd.concat([train_df, new_mashups_df], ignore_index=True)

print(f"训练集扩充后的大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

train_seqs = train_df['API_ID'].tolist()
test_seqs = test_df['API_ID'].tolist()

# ---------------------------------------------------------------
# 处理session输入序列和标签
# 将训练集和测试集组合成all_seqs
all_seqs = train_df['API_ID'].tolist() + test_df['API_ID'].tolist()

# 遍历 all_seqs，求出n_node
max_api_id = max(api_id for seq in all_seqs for api_id in seq)
print(f"n_node: {max_api_id + 1}")


def process_seqs(iseqs):
    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs, ids

def process_seqs_with_mashup_id(iseqs, mashup_ids):
    out_seqs = []
    labs = []
    seq_ids = []
    mashup_id_list = []
    
    for seq_id, (seq, mashup_id) in enumerate(zip(iseqs, mashup_ids)):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs.append(tar)
            out_seqs.append(seq[:-i])
            seq_ids.append(seq_id)
            mashup_id_list.append(mashup_id)  # 保持mashup ID追踪
    
    return out_seqs, labs, seq_ids, mashup_id_list

# 获取mashup ID列表（修复tolist()方法调用）
train_mashup_ids = train_df['original_mashup_id'].tolist()
test_mashup_ids = test_df['original_mashup_id'].tolist()

# 处理序列并保持mashup ID追踪
tra_seqs, tra_labs, tra_seq_ids, tra_mashup_ids = process_seqs_with_mashup_id(train_seqs, train_mashup_ids)
tes_seqs, tes_labs, tes_seq_ids, tes_mashup_ids = process_seqs_with_mashup_id(test_seqs, test_mashup_ids)

tra_data=(tra_seqs, tra_labs)
tes_data=(tes_seqs, tes_labs)

# 保存数据
# pickle.dump(tra_data, open('data/train.txt', 'wb'))
# pickle.dump(tes_data, open('data/test.txt', 'wb'))
# pickle.dump(train_seqs, open('data/all_train_seq.txt', 'wb'))

# json格式保存
# 保存包含mashup ID的训练和测试数据
with open('data/train.json', 'w', encoding='utf-8') as f:
    json.dump({
        'sequences': tra_seqs, 
        'labels': tra_labs,
        'mashup_ids': tra_mashup_ids  # 添加mashup ID追踪
    }, f, separators=(',', ':'), ensure_ascii=False)

with open('data/test.json', 'w', encoding='utf-8') as f:
    json.dump({
        'sequences': tes_seqs, 
        'labels': tes_labs,
        'mashup_ids': tes_mashup_ids  # 添加mashup ID追踪
    }, f, separators=(',', ':'), ensure_ascii=False)

# 保存mashup ID到原始mashup信息的映射
mashup_info_mapping = {}
for idx, row in pd.concat([train_df, test_df]).iterrows():
    original_id = row['original_mashup_id']
    if original_id not in mashup_info_mapping:
        mashup_info_mapping[original_id] = {
            'original_name': row['mashups_name'].split('_perm')[0],  # 去除排列后缀
            'api_sequence': row['API_ID']
        }

with open('data/mashup_info_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(mashup_info_mapping, f, separators=(',', ':'), ensure_ascii=False)

with open('data/all_train_seq.json', 'w', encoding='utf-8') as f:
    json.dump(train_seqs, f, separators=(',', ':'), ensure_ascii=False)

print("数据处理完成，已保存为JSON格式。")