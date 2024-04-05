import random
import torch
import numpy as np
import json
from flask import Flask, request, jsonify
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
import pymysql
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求
# MySQL数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'ft_demo',
    'cursorclass': pymysql.cursors.DictCursor  # 返回字典形式的结果集
}
# 创建数据库连接
connection = pymysql.connect(**DB_CONFIG)

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
args = None
device = None
model = None
item_dict = None


def load_model():
    global args, device, model, item_dict, user_gcn_emb, entity_gcn_emb
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """set args"""
    args = parse_args()
    args.dataset = 'jd'
    args.dim = 64
    args.lr = 0.0001
    args.sim_regularity = 0.0001
    args.batch_size = 1024
    args.node_dropout = True
    args.node_dropout_rate = 0
    args.mess_dropout = True
    args.mess_dropout_rate = 0.1
    args.gpu_id = 0
    args.context_hops = 3

    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

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
    item_dict = {}
    f = open("./data/jd/item_list.txt", "r", encoding='utf-8')
    row = f.readlines()
    for each_row in row:
        item_dict[each_row.split(",")[0]] = each_row.split(",")[1]
    f.close()


@app.route('/recommend-job', methods=['GET'])
def recommend_job():
    global args, device, model, item_dict
    user_id = int(request.args.get('user_id', default=4439))
    user = user_gcn_emb[user_id, :]
    item = entity_gcn_emb
    rate = torch.matmul(user, item.t())

    # 设置要获取的 top-k 值的数量
    k = 5
    # 计算 top-k 值及其对应的索引
    top_values, top_indices = torch.topk(rate, k)

    recommended_items = []
    for job in top_indices:
        recommended_items.append(job.item())

    # return json.dumps(recommended_items, ensure_ascii=False)
    try:
        with connection.cursor() as cursor:
            job_results = {}
            for index, job_id in enumerate(recommended_items):
                sql = "select * from job_table where ordered_id = %s"
                cursor.execute(sql, (job_id,))
                result = cursor.fetchone()
                if result:
                    job_results[index] = result
        # 使用 jsonify 函数将 Python 对象转换为 JSON 格式并返回
        return jsonify(job_results)
    except Exception as e:
        print("数据库查询出错:", e)
        return jsonify([])


@app.route('/recommend-seeker', methods=['GET'])
def recommend_seeker():
    global args, device, model, item_dict
    # item_id = int(request.args.get('item_id', default=4439))
    item_id = int(request.args.get('jobId', default=4439))
    # item_id = request.args.get('item_id')
    user = user_gcn_emb
    item = entity_gcn_emb[item_id, :]
    rate = torch.matmul(user, item.t())

    # 设置要获取的 top-k 值的数量
    k = 5
    # 计算 top-k 值及其对应的索引
    top_values, top_indices = torch.topk(rate, k)

    recommended_items = []
    for user in top_indices.squeeze():
        recommended_items.append(user.item())

    try:
        # 创建游标对象
        with connection.cursor() as cursor:
            resume_results = []
            # 执行 SQL 查询
            for user_id in recommended_items:
                sql = "SELECT * FROM Resume WHERE id = %s"
                cursor.execute(sql, (user_id,))
                # 获取查询结果
                result = cursor.fetchall()
                resume_results.append(result)
        return jsonify(resume_results)
    except Exception as e:
        print("数据库查询出错:", e)
        return jsonify([])
    # return json.dumps(recommended_items, ensure_ascii=False)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
