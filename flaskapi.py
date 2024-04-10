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
import re
from fuzzywuzzy import fuzz
import pymysql

mydb = pymysql.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="root",
    database="ft_demo"
)

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


def snake_to_camel(name):
    # 将蛇形命名转换为驼峰命名
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)


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

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * from Resume where id =%s", (user_id,))
    result_a = mycursor.fetchall()

    mycursor.execute("SELECT * FROM Resume where id!=%s", (user_id,))
    result_b = mycursor.fetchall()

    for item_a in result_a:
        max_ratio = 0
        matched_item = ""
        for item_b in result_b:
            ratio = fuzz.ratio(item_a[9], item_b[9])
            ratio1 = fuzz.ratio(str(item_a[2]), str(item_b[2]))
            ratio2 = fuzz.ratio(str(item_a[10]), str(item_b[10]))
            ratio3 = fuzz.ratio(str(item_a[11]), str(item_b[11]))
            ratio4 = fuzz.ratio(str(item_a[13]), str(item_b[13]))
            ratio5 = fuzz.ratio(str(item_a[15]), str(item_b[16]))
            ratio6 = fuzz.ratio(str(item_a[17]), str(item_b[17]))
            sum = ratio6 + ratio5 + ratio4 + ratio3 + ratio2 + ratio1 + ratio
            if sum > max_ratio:
                max_ratio = sum
            matched_item = item_b[0]

            # print(matched_item)
            user = user_gcn_emb[matched_item, :]
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
                    # 将字段名转换为驼峰模式
                    result_camel = {snake_to_camel(key): value for key, value in result.items()}
                    job_results[index] = result_camel
            return jsonify(job_results)
    except Exception as e:
        print("数据库查询出错:", e)
        return jsonify([])


@app.route('/recommend-seeker', methods=['GET'])
def recommend_seeker():
    global args, device, model, item_dict
    item_id = request.args.get('jobId')

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * from job_table where id =%s", (item_id,))
    result_a = mycursor.fetchall()

    mycursor.execute("SELECT * FROM job_table where id!=%s", (item_id,))
    result_b = mycursor.fetchall()

    for item_a in result_a:
        max_ratio = 0
        matched_item1 = ""
        for item_b in result_b:
            ratio = fuzz.ratio(item_a[1], item_b[1])
            ratio2 = fuzz.ratio(item_a[4], item_b[4])
            ratio3 = fuzz.ratio(str(item_a[6]), str(item_b[6]))
            ratio4 = fuzz.ratio(str(item_a[7]), str(item_b[7]))
            ratio5 = fuzz.ratio(str(item_a[12]), str(item_b[12]))
            ratio6 = fuzz.ratio(str(item_a[13]), str(item_b[13]))
            ratio7 = fuzz.ratio(str(item_a[14]), str(item_b[14]))
            ratio8 = fuzz.ratio(str(item_a[15]), str(item_b[15]))
            ratio9 = fuzz.ratio(str(item_a[16]), str(item_b[16]))
            sum = ratio5 + ratio4 + ratio3 + ratio2 + ratio + ratio9 + ratio8 + ratio7 + ratio6
            if sum > max_ratio:
                max_ratio = sum
                # matched_item = item_b[-1]
                matched_item1 = item_b[16]

    print(matched_item1)
    # item_id = request.args.get('item_id')
    user = user_gcn_emb
    item = entity_gcn_emb[matched_item1, :]
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
                # 将字段名转换为驼峰模式
                result_camel = [{snake_to_camel(key): value for key, value in item.items()} for item in result]
                resume_results.append(result_camel)
            return jsonify(resume_results)
    except Exception as e:
        print("数据库查询出错:", e)
        return jsonify([])
    # return json.dumps(recommended_items, ensure_ascii=False)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
