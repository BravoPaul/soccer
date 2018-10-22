import numpy as np
import pandas as pd
import jieba.posseg as pseg
#
#
# result = pseg.cut("也是现在的贱女人太多")
#
# for w in result:
#     print(w)
#
# # print(result)
#
# # print(np.random.rand(128))
# touzi = [300,500,1000,2000,3000,3000,3000,3000]

touzi = np.random.randint(300,8000,8)
touzi = [80 for i in range(200)]
# touzi = [300,500,800,1400,2200,3200,4800,7000]



#  计算概率和赔率
ps = [1.5+i*0.01 for i in range(100)]
for p in ps:
    profits = []
    for i, value in enumerate(touzi):
        once_profit = value*(p-1)
        profit = once_profit - np.array(touzi[0:i]).sum()
        profits.append(profit)
        # print("投资第%d次:本次汇报为%f,净赚为:%f,日均净赚为:%f。"%((i+1),once_profit,profit,profit/(i+1)))
    g_ws = [0 + i * 0.01 for i in range(100)]
    for g_w in g_ws:
        g_l = 1-g_w
        e = 0
        for i, value in enumerate(profits):
            e += value*pow(g_l,i)
        e = e*g_w-np.array(touzi).sum()*pow(g_l,len(profits))
        if e>0:
            # print(p,"    ",g_w,"    ",p/g_w)
            break



# 计算能挣多少钱
touzie = 1000
gailv = 0.68
peilv = 1.9
profit = gailv*touzie*(peilv-1)-(touzie)*(1-gailv)
print(profit)



