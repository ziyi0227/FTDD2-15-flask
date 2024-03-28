import pandas as pd
f = open("./data/jd/kg_final.txt", "a")
df = pd.read_csv("./data/jd/job.csv")
for index , row in df.iterrows():
    r1 = row["品质"][1:-1].split(",")
    r2 = row["工作要求"][1:-1].split(",")
    r3 = row["工作要求.1"][1:-1].split(",")

    for x in r1:
        if r1[0] == "":
            continue
        f.write(str(index)+"\t"+str(5)+"\t"+str(int(x) + 1445)+"\n")
    for x in r2:
        if r2[0] == "":
            continue
        f.write(str(index)+"\t"+str(6)+"\t"+str(int(x) + 1445)+"\n")

    for x in r3:
        if r3[0] == "":
            continue

        f.write(str(index)+"\t"+str(7)+"\t"+str(int(x) + 1445)+"\n")


# print(min(s1), max(s1))         #0 19377
# print(min(s2), max(s2))         #1000 1444
