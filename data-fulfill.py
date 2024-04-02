import csv

# 读取 CSV 文件并处理每行数据
with open('example.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    rows = list(reader)

# 处理每行数据，删除后两列文字前的逗号
for row in rows:
    for i in range(-2, 0):  # 后两列
        if row[i]:  # 如果数据不为空
            if row[i].startswith('"') and row[i].endswith('"'):  # 如果数据被引号括起来
                row[i] = '"' + row[i][1:].lstrip(' ,').strip()  # 删除前面的逗号并保留引号
            else:
                row[i] = row[i].lstrip(' ,')  # 否则，删除前面的逗号

# 将处理后的数据写回 CSV 文件
with open('example.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print("已成功删除每行数据中后两列文字前的逗号。")
