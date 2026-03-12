import argparse
import pickle
import time
import json
import os
import numpy as np
from datetime import datetime
from utils import build_graph, Data, split_validation
from model import *
from preprocess import *


# 通过命令行传入参数，控制数据集选择、模型超参数和训练细节。
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--topk', type=int, default=5, help='top-k 推荐数量')
parser.add_argument('--output_path', type=str, default='results', help='输出文件夹路径')
opt = parser.parse_args()
print(opt)

# train_data = pickle.load(open('data/train.txt', 'rb'))
# test_data = pickle.load(open('data/test.txt', 'rb'))
# all_data = pickle.load(open('data/all_train_seq.txt', 'rb'))

# 读取JSON格式文件
with open('data/train.json', 'r', encoding='utf-8') as f:
    train_data_dict = json.load(f)
    train_data = (train_data_dict['sequences'], train_data_dict['labels'])

with open('data/test.json', 'r', encoding='utf-8') as f:
    test_data_dict = json.load(f)
    test_data = (test_data_dict['sequences'], test_data_dict['labels'])

with open('data/all_train_seq.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

train_data = Data(train_data, shuffle=True)
test_data = Data(test_data, shuffle=False)

n_node = max_api_id + 1

def evaluate_model(model, test_data, top_k_list=list(range(1, 31))):
    """评估模型在不同top-k值下的性能"""
    print("开始评估模型性能...")
    
    results = {}
    
    for top_k in top_k_list:
        print(f"评估 Top-{top_k}...")
        hit, mrr, ndcg, precision, map_score, _ = train_test(model, None, test_data, top_k, train=False)
        
        results[top_k] = {
            'Recall': hit,
            'MRR': mrr,
            'NDCG': ndcg,
            'Precision': precision,
            'MAP': map_score
        }
        
        print(f"Top-{top_k}: NDCG={ndcg:.4f}, Recall={hit:.4f}, Precision={precision:.4f}, MAP={map_score:.4f}")
    
    return results

def save_results(results, output_path, opt):
    """保存结果到多种格式的文件"""
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON格式
    json_file = os.path.join(output_path, f"srgnn_main_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式
    csv_file = os.path.join(output_path, f"srgnn_main_results_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Top-K,NDCG,Recall,Precision,MAP,MRR\n")
        for k in sorted(results.keys()):
            if isinstance(k, int):
                f.write(f"{k},{results[k]['NDCG']:.6f},{results[k]['Recall']:.6f},"
                       f"{results[k]['Precision']:.6f},{results[k]['MAP']:.6f},{results[k]['MRR']:.6f}\n")
    
    # 保存详细报告
    report_file = os.path.join(output_path, f"srgnn_main_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SR-GNN主程序评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"批次大小: {opt.batchSize}\n")
        f.write(f"隐藏层大小: {opt.hiddenSize}\n")
        f.write(f"训练轮数: {opt.epoch}\n")
        f.write(f"学习率: {opt.lr}\n")
        f.write(f"L2正则化: {opt.l2}\n")
        f.write(f"GNN步数: {opt.step}\n")
        f.write(f"耐心值: {opt.patience}\n")
        f.write("\n性能指标:\n")
        f.write("-" * 30 + "\n")
        
        for k in sorted(results.keys()):
            if isinstance(k, int):
                f.write(f"Top-{k:2d}: NDCG={results[k]['NDCG']:.6f}, "
                       f"Recall={results[k]['Recall']:.6f}, "
                       f"Precision={results[k]['Precision']:.6f}, "
                       f"MAP={results[k]['MAP']:.6f}, "
                       f"MRR={results[k]['MRR']:.6f}\n")
    
    print(f"结果已保存到: {output_path}")
    return json_file, csv_file, report_file

def main():
    # 将模型转移到 GPU 上
    model = trans_to_cuda(SessionGraph(opt, n_node))

    # 初始化训练相关的变量
    start = time.time()
    best_result = [0, 0, 0, 0, 0]  # 扩展为5个指标：hit, mrr, ndcg, precision, map_score
    best_epoch = [0, 0, 0, 0, 0]
    bad_counter = 0
    
    print("开始训练模型...")
    for epoch in range(opt.epoch):
        print('------------------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, ndcg, precision, map_score, top_k = train_test(model, train_data, test_data, opt.topk)
        flag = 0

        # 更新最佳性能
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if ndcg >= best_result[2]:
            best_result[2] = ndcg
            best_epoch[2] = epoch
            flag = 1
        if precision >= best_result[3]:
            best_result[3] = precision
            best_epoch[3] = epoch
            flag = 1
        if map_score >= best_result[4]:
            best_result[4] = map_score
            best_epoch[4] = epoch
            flag = 1
    
        
        # 打印最佳指标
        print('best result:')
        print(f'Recall@{top_k}: {best_result[0]:.2f}% (Epoch: {best_epoch[0]})')
        print(f'MRR@{top_k}: {best_result[1]:.2f}% (Epoch: {best_epoch[1]})')
        print(f'NDCG@{top_k}: {best_result[2]:.2f}% (Epoch: {best_epoch[2]})')
        print(f'Precision@{top_k}: {best_result[3]:.2f}% (Epoch: {best_epoch[3]})')
        print(f'MAP@{top_k}: {best_result[4]:.2f}% (Epoch: {best_epoch[4]})')
    
        # 早停机制：
        # 如果连续 patience 次训练轮次中性能未提升，则终止训练。
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    
    print('--------------------------------------------------------------')
    print("训练完成，开始全面评估...")
    
    # 在训练完成后，评估所有top-k值（1到30）
    evaluation_results = evaluate_model(model, test_data, list(range(1, 31)))
    
    # 保存评估结果
    save_results(evaluation_results, opt.output_path, opt)
    
    end = time.time()
    minutes = int((end - start) // 60)
    seconds = int((end - start) % 60)
    print("Run time: {} min {} s".format(minutes, seconds))
    print("评估完成，结果已保存！")


if __name__ == '__main__':
    main()
