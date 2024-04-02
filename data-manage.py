import pandas as pd


def check_continuous_ids(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file, encoding='gbk')

    # 获取ID列并转换为列表
    ids = df['ID'].tolist()

    # 检查ID是否连续
    for i in range(len(ids) - 1):
        if ids[i + 1] - ids[i] != 1:
            print(f"ID {ids[i]} 和 ID {ids[i + 1]} 之间不连续！")

    print("检查完成：ID连续。")


# CSV文件路径
csv_file = 'example.csv'

# 执行检查
check_continuous_ids(csv_file)
