import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import openpyxl
import sklearn.svm
from sklearn import neighbors
import xlsxwriter
import pandas.io.formats.excel
import scipy
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from random import seed
from random import randrange
from numpy import *
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pywt
import math
color_map = ['c', 'b', 'g', 'm', 'y', 'k', 'r']

def adress_change(str_:str): #把地址的\改成/
    str_pro = str()
    for i in str_:
        if i == '\\':
            str_pro += '/'
        elif i != '\\':
            str_pro += i
    return str_pro

sheet1 = pd.read_excel(adress_change(r"C:\Users\luokun\Desktop\数据文件.xlsx"),index_col="day")
# print(sheet1)

for y_in in sheet1.columns:
    plt.figure(dpi=200)
    plt.xlabel("day")
    plt.ylabel("in - out")
    plt.plot(sheet1.index, sheet1[y_in])
    plt.title(f"{y_in}")
    plt.savefig(adress_change(f'D:\校赛\图片/{y_in}.png'))
    # plt.show()
    plt.close()

# data = np.array(sheet1)
# print(data)

#模块调用

data = sheet1

list_data_index = []
for i in range(len(data.index)):
    list_data_index.append(i)
# print(list_data_index)


list_data = []
for i in sheet1.columns:
    list_data.append(np.array(sheet1[i]))




DBSCAN
DBSCAN_result = []
a = []
for i in range(len(sheet1.columns)):
    target = np.array(sheet1[sheet1.columns[i]]).reshape(-1, 1)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(target).labels_
    plt.figure(dpi=600)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]
        plt.xlabel("day")
        plt.ylabel("in - out")
        plt.scatter(sheet1.index[j], sheet1.loc[j+1][y],marker = '*',c=color_map[clustering[j]] )
        plt.title(f"{y}_DBSCAN")
        if clustering[j]==-1 :
            a.append(j)
    plt.savefig(adress_change(f'D:\校赛\DBSCAN/{y}.png'))
    DBSCAN_result.append(a)
    a = []
    # plt.show()
    plt.close()
# print(DBSCAN_result)

#
#
from scipy import stats
# z-score
# target = np.array(sheet1[sheet1.columns[0]]).reshape(-1, 1)
# target = sheet1[sheet1.columns[0]]
out = []
label = []
b = []
b_index = []
Zscore_result = []
def ZRscore_outlier(df):
    med = np.median(df)
    ma = stats.median_abs_deviation(df)
    for i in range(len(df)):
        z = (0.6745*(df[i]-med))/ (np.median(ma))
        if np.abs(z) > 3:
            out.append(i)
            label.append(-1)
            b_index.append(i)
        else:
            label.append(0)
    # print(b_index)
    Zscore_result.append(b_index)



def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in range(len(df)):
        z = (df[i]-m)/sd
        if np.abs(z) > 2:
            out.append(i)
            label.append(-1)
            b_index.append(i)
        else:
            label.append(0)
    Zscore_result.append(b_index)


for i in range(len(sheet1.columns)):
    target = np.array(sheet1[sheet1.columns[i]]).reshape(-1, 1)
    b_index = []
    Zscore_outlier(target)
    plt.figure(dpi=600)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]

        plt.xlabel("day")
        plt.ylabel("in - out")
        plt.scatter(sheet1.index[j], sheet1.loc[j+1][y],marker = '*',c=color_map[label[j]] )
        plt.title(f"{y}_Zscore")
    plt.savefig(adress_change(f'D:\校赛\Z-SCORE/{y}.png'))
    b.append(label)
    label = []
    # plt.show()
    plt.close()
# print(Zscore_result)
#
#
# 画箱线图
c = []
d = []
box_result = []
box_index = []
for i in range(len(sheet1.columns)):
    target = np.array(sheet1[sheet1.columns[i]]).reshape(-1, 1)
    fig, ax = plt.subplots(dpi=600)
    ax.boxplot(target)
        # plt.savefig(adress_change(f'D:\校赛\图片/{y_in}.png'))
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]
        if sheet1.loc[j+1][y] < lower_bound or sheet1.loc[j+1][y] > upper_bound:
            c.append(j)
            d.append(-1)
        else:
            d.append(0)
    box_result.append(c)
    c = []
    box_index.append(d)
    d = []
    plt.savefig(adress_change(f'D:\校赛\BOX_PLOT/{y}.png'))
    # plt.show()
    plt.close()
# print(box_result)
#
for i in range(len(sheet1.columns)):
     target = np.array(sheet1[sheet1.columns[i]]).reshape(-1, 1)
     plt.figure(dpi=600)
     for j in range(len(sheet1.index)):
         y = sheet1.columns[i]
         plt.xlabel("day")
         plt.ylabel("in - out")
         plt.scatter(sheet1.index[j], sheet1.loc[j+1][y],marker = '*',c=color_map[box_index[i][j]])
         plt.title(f"{y}_box")
     plt.savefig(adress_change(f'D:\校赛\BOX/{y}.png'))
     # plt.show()
     plt.close()
#
# DBSCAN_result = np.array(DBSCAN_result, dtype=object).reshape(-1,1)
# print(DBSCAN_result)
# Zscore_result = np.array(Zscore_result, dtype=object).reshape(-1,1)
# print(Zscore_result)
# box_result = np.array(box_result, dtype=object).reshape(-1,1)
# print(box_result)
# outlier = []
# outlier2 = []
# e = []
# for i in range(len(sheet1.columns)):
#     DBSCAN_result_hashable = [tuple(x) for x in DBSCAN_result[i]]
#     for element in set(DBSCAN_result_hashable):
#         set_D = set(element)
#
#     Zscore_result_hashable = [tuple(x) for x in Zscore_result[i]]
#     for element in set(Zscore_result_hashable):
#         set_Z = set(element)
#
#     box_result_hashable = [tuple(x) for x in box_result[i]]
#     for element in set(box_result_hashable):
#         set_b = set(element)
#
#     outlier.append(set_D & set_Z | set_D & set_b | set_Z & set_b)
#     outlier2.append(set_D & set_Z & set_b)
# outlier_list = []
# for i in outlier:
#     outlier_list.append(list(i))
# for i in range(len(sheet1.columns)):
#     target = np.array(sheet1[sheet1.columns[i]]).reshape(-1, 1)
#     plt.figure(dpi=600)
#     for j in range(len(sheet1.index)):
#         y = sheet1.columns[i]
#
#         plt.xlabel("day")
#         plt.ylabel("in - out")
#         for la_list in outlier_list[i]:
#             if j == la_list:
#                 plt.scatter(sheet1.index[j], sheet1.loc[j + 1][y], marker='*', c=color_map[-1])
#                 break
#
#             else:
#                 plt.scatter(sheet1.index[j], sheet1.loc[j + 1][y], marker='*', c=color_map[0])
#         plt.title(f"{y}_outlier")
#     plt.savefig(adress_change(f'D:\校赛/3并/{sheet1.columns[i]}.png'))
#     # plt.show()
#     plt.close()
#
#
# import xlsxwriter
# writer_valus = pd.ExcelWriter(adress_change('D:\校赛/异常值.xlsx'),engine='xlsxwriter')
# # tegether1.to_excel(writer_kt_valus,sheet_name='纹饰')
# # tegether3.to_excel(writer_kt_valus,sheet_name='颜色')
# # tegether4.to_excel(writer_kt_valus,sheet_name='表面风化情况')
# # writer_kt_valus.close()
#
#
# print(DBSCAN_result[0][0])
# df1 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
# for i in range(len(sheet1.columns)):
#     for j in range(len(DBSCAN_result[i][0])):
#         df1.iloc[j][i] = DBSCAN_result[i][0][j]
# df2 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
# for i in range(len(sheet1.columns)):
#     for j in range(len(Zscore_result[i][0])):
#         df2.iloc[j][i] = Zscore_result[i][0][j]
# df3 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
# for i in range(len(sheet1.columns)):
#     for j in range(len(box_result[i][0])):
#         df3.iloc[j][i] = box_result[i][0][j]
# df4 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
# for i in range(len(sheet1.columns)):
#     for j in range(len(outlier_list[i])):
#         df4.iloc[j][i] = outlier_list[i][j]
#
# df1.to_excel(writer_valus,sheet_name='DBSCAN')
# df2.to_excel(writer_valus,sheet_name='Zscore')
# df3.to_excel(writer_valus,sheet_name='box_result')
# df4.to_excel(writer_valus,sheet_name='all')
# writer_valus.close()

from scipy.signal import savgol_filter

# Generate some noisy data
# Apply SG filter
window_size = 11
poly_order = 2
smooth_data = savgol_filter(list_data, window_size, poly_order)
print(smooth_data)

for y_in in range(len(sheet1.columns)):

    plt.figure(dpi=200)
    plt.xlabel("day")
    plt.ylabel("in - out")
    plt.plot(sheet1.index, smooth_data[y_in])
    plt.title(f"SG{y_in+1}")
    plt.savefig(adress_change(f'D:\校赛\降噪/降噪{y_in}.png'))
    # plt.show()
    plt.close()

# DBSCAN
DBSCAN_result = []
a = []
for i in range(len(sheet1.columns)):
    target = np.array(smooth_data[i]).reshape(-1, 1)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(target).labels_
    plt.figure(dpi=600)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]
        plt.xlabel("day")
        plt.ylabel("in - out")
        plt.scatter(sheet1.index[j], target[j],marker = '*',c=color_map[clustering[j]] )
        plt.title(f"{y}_DBSCAN")
        if clustering[j]==-1 :
            a.append(j)
    plt.savefig(adress_change(f'D:\校赛\降噪散点\DBSCAN/db{y}.png'))
    DBSCAN_result.append(a)
    a = []
    # plt.show()
    plt.close()


from scipy import stats
# z-score
# target = np.array(sheet1[sheet1.columns[0]]).reshape(-1, 1)
# target = sheet1[sheet1.columns[0]]
out = []
label = []
b = []
b_index = []
Zscore_result = []
def ZRscore_outlier(df):
    med = np.median(df)
    ma = stats.median_abs_deviation(df)
    for i in range(len(df)):
        z = (0.6745*(df[i]-med))/ (np.median(ma))
        if np.abs(z) > 3:
            out.append(i)
            label.append(-1)
            b_index.append(i)
        else:
            label.append(0)
    # print(b_index)
    Zscore_result.append(b_index)



def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in range(len(df)):
        z = (df[i]-m)/sd
        if np.abs(z) > 2:
            out.append(i)
            label.append(-1)
            b_index.append(i)
        else:
            label.append(0)
    Zscore_result.append(b_index)


for i in range(len(sheet1.columns)):
    target = np.array(smooth_data[i]).reshape(-1, 1)
    b_index = []
    Zscore_outlier(target)
    plt.figure(dpi=600)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]

        plt.xlabel("day")
        plt.ylabel("in - out")
        plt.scatter(sheet1.index[j], target[j],marker = '*',c=color_map[label[j]] )
        plt.title(f"{y}_Zscore")
    plt.savefig(adress_change(f'D:\校赛\降噪散点\zscore/zs{y}.png'))
    b.append(label)
    label = []
    # plt.show()
    plt.close()


# 画箱线图
c = []
d = []
box_result = []
box_index = []
for i in range(len(sheet1.columns)):
    target = np.array(smooth_data[i]).reshape(-1, 1)
    fig, ax = plt.subplots(dpi=600)
    ax.boxplot(target)
        # plt.savefig(adress_change(f'D:\校赛\图片/{y_in}.png'))
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]
        if target[j] < lower_bound or target[j] > upper_bound:
            c.append(j)
            d.append(-1)
        else:
            d.append(0)
    box_result.append(c)
    c = []
    box_index.append(d)
    d = []
    plt.savefig(adress_change(fr'D:\校赛\降噪散点\box\BOX_PLOT/{y}.png'))
    # plt.show()
    plt.close()
# print(box_result)
#
for i in range(len(sheet1.columns)):
     target = np.array(smooth_data[i]).reshape(-1, 1)
     plt.figure(dpi=600)
     for j in range(len(sheet1.index)):
         y = sheet1.columns[i]
         plt.xlabel("day")
         plt.ylabel("in - out")
         plt.scatter(sheet1.index[j], target[j],marker = '*',c=color_map[box_index[i][j]])
         plt.title(f"{y}_box")
     plt.savefig(adress_change(rf'D:\校赛\降噪散点\box/BOX/{y}.png'))
     # plt.show()
     plt.close()


DBSCAN_result = np.array(DBSCAN_result, dtype=object).reshape(-1,1)
print(DBSCAN_result)
Zscore_result = np.array(Zscore_result, dtype=object).reshape(-1,1)
print(Zscore_result)
box_result = np.array(box_result, dtype=object).reshape(-1,1)
print(box_result)
outlier = []
outlier2 = []
e = []
for i in range(len(sheet1.columns)):
    DBSCAN_result_hashable = [tuple(x) for x in DBSCAN_result[i]]
    for element in set(DBSCAN_result_hashable):
        set_D = set(element)

    Zscore_result_hashable = [tuple(x) for x in Zscore_result[i]]
    for element in set(Zscore_result_hashable):
        set_Z = set(element)

    box_result_hashable = [tuple(x) for x in box_result[i]]
    for element in set(box_result_hashable):
        set_b = set(element)

    outlier.append(set_D & set_Z | set_D & set_b | set_Z & set_b)
    outlier2.append(set_D & set_Z & set_b)
outlier_list = []
for i in outlier:
    outlier_list.append(list(i))
for i in range(len(sheet1.columns)):
    target = np.array(smooth_data[i]).reshape(-1, 1)
    plt.figure(dpi=600)
    for j in range(len(sheet1.index)):
        y = sheet1.columns[i]

        plt.xlabel("day")
        plt.ylabel("in - out")
        for la_list in outlier_list[i]:
            if j == la_list:
                plt.scatter(sheet1.index[j], target[j], marker='*', c=color_map[-1])
                break

            else:
                plt.scatter(sheet1.index[j], target[j], marker='*', c=color_map[0])
        plt.title(f"{y}_outlier")
    plt.savefig(adress_change(fr'D:\校赛\降噪散点\all/{sheet1.columns[i]}.png'))
    # plt.show()
    plt.close()


import xlsxwriter
writer_valus = pd.ExcelWriter(adress_change('D:\校赛/异常值降噪.xlsx'),engine='xlsxwriter')
# tegether1.to_excel(writer_kt_valus,sheet_name='纹饰')
# tegether3.to_excel(writer_kt_valus,sheet_name='颜色')
# tegether4.to_excel(writer_kt_valus,sheet_name='表面风化情况')
# writer_kt_valus.close()


print(DBSCAN_result[0][0])
df1 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
for i in range(len(sheet1.columns)):
    for j in range(len(DBSCAN_result[i][0])):
        df1.iloc[j][i] = DBSCAN_result[i][0][j]
df2 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
for i in range(len(sheet1.columns)):
    for j in range(len(Zscore_result[i][0])):
        df2.iloc[j][i] = Zscore_result[i][0][j]
df3 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
for i in range(len(sheet1.columns)):
    for j in range(len(box_result[i][0])):
        df3.iloc[j][i] = box_result[i][0][j]
df4 = pd.DataFrame(columns=sheet1.columns,index=sheet1.index)
for i in range(len(sheet1.columns)):
    for j in range(len(outlier_list[i])):
        df4.iloc[j][i] = outlier_list[i][j]

df1.to_excel(writer_valus,sheet_name='DBSCAN')
df2.to_excel(writer_valus,sheet_name='Zscore')
df3.to_excel(writer_valus,sheet_name='box_result')
df4.to_excel(writer_valus,sheet_name='all')
writer_valus.close()



