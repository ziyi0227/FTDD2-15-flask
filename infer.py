'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import torch
import numpy as np
import flask
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)

    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    print("start infering ...")

    model.load_state_dict(torch.load('./weights/model_jd.ckpt', map_location='cuda:0'))
    model.eval()

    entity_gcn_emb, user_gcn_emb = model.generate()
    user = user_gcn_emb[4439, :]
    item = entity_gcn_emb
    rate = torch.matmul(user, item.t())

    item_dict = {}
    f = open("./data/jd/item_list.txt", "r", encoding='utf-8')
    row = f.readlines()
    for each_row in row:
        item_dict[each_row.split(",")[0]] = each_row.split(",")[1]
    f.close()

    # 设置要获取的 top-k 值的数量
    k = 5
    # 计算 top-k 值及其对应的索引
    top_values, top_indices = torch.topk(rate, k)

    for job in top_indices:
        print(item_dict[str(job.item())])

    l = [8673, 14151, 330, 5138, 13833, 9466, 7064]
    print("Groud Truth")
    for x in l :
        print(item_dict[str(x)])

# 运行：python infer.py --dataset jd --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --node_dropout True --node_dropout_rate 0 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3

