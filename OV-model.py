import pandas as pd
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from joblib import dump


#data载入及预处理
warnings.filterwarnings("ignore")
file1=pd.read_csv('tcga_expr_index_149.txt',sep='\t')
gene=list(file1.iloc[:,0])
file1.set_index('gene', inplace=True)
print(file1)
print(type(file1))
file2=pd.read_csv('tcga_sample 分型.txt',sep='\t')
file2['Mixture'] = file2['Mixture'].str.replace('-', '.')
dict={}
for i in range(file2.shape[0]):
    dict[file2.iloc[i,0]]=file2.iloc[i,1]
file2.set_index('Mixture', inplace=True)
sample=[]
print(file1)
y_values=[]
for i in dict.keys():
    if dict[i]!='subtype2':
        sample.append(i)
        y_values.append(dict[i])
file1=file1.loc[:,sample]
file2=pd.DataFrame({'sample':y_values})
file2['sample'] = file2['sample'].replace({'subtype1':0, 'subtype3': 1})

#X,y
X=file1.values.T
y=file2.values


# RFECV 特征基因筛选
svc = SVC(kernel='linear',probability=True)
model = LogisticRegression()

rfecv = RFECV(estimator=svc,          # 学习器
              min_features_to_select=2, # 最小选择的特征数量
              step=4,                 # 移除特征个数
              cv=StratifiedKFold(16),  # 交叉验证次数
              scoring='accuracy',     # 学习器的评价标准
              verbose = 0,#详细程度控制。默认为 0，不输出进度信息。
              n_jobs = 1#并行运行的作业数。默认为 1，表示不使用并行化。
              ).fit(X, y)
print(rfecv)
X_RFECV = rfecv.transform(X)
#print(X_RFECV)
print(len(rfecv.ranking_))
print("RFECV特征选择结果——————————————————————————————————————————————————")
print("有效特征个数 : %d" % rfecv.n_features_)
print("全部特征等级 : %s" % list(rfecv.ranking_))

# 保存模型
dump(rfecv, 'rfecv.joblib')
ranking=list(rfecv.ranking_)
importgene=[]
for i in range(len(ranking)):
    if ranking[i]==1:
        importgene.append(gene[i])
print(importgene)
#输出特征基因
output_file = 'gene_import.txt'
with open(output_file, 'w') as file:
    file.write("gene\n")  # 写入表头 "gene"
    for item in importgene:
        file.write(item + '\n')
filex=file1.loc[importgene,:]
print(filex)
X_import=filex.values.T
print(X_import)
print(len(X_import))

svc = SVC(kernel='poly',probability=True,C=0.45)
svc.fit(X_import, y)

y_pred_prob = svc.predict_proba(X_import)

# 查看预测结果
print(y_pred_prob)

# 将预测概率转换为类别标签
y_pred = np.argmax(y_pred_prob, axis=1)
print(y_pred)

proper_tcga = pd.DataFrame(y_pred_prob, columns=['subtype1', 'subtype3'])
class_tcga = list(y_pred)
print(class_tcga)
print(proper_tcga)
#保存模型
dump(svc, 'svc_model.joblib')


# #自身检验
# from sklearn.metrics import roc_auc_score
#
# #subtype1
# subtype1=[]
# for i  in sample:
#     if dict[i]=='subtype1':
#         subtype1.append(1)
#     if dict[i]!='subtype1':
#         subtype1.append(0)
# expr_one = list(proper_tcga.iloc[:,0])
# roc_auc1 = roc_auc_score(subtype1, expr_one)
# print('subtype1的AUC:'+str(roc_auc1))
#
# #subtype3
# subtype3=[]
# for i  in sample:
#     if dict[i]=='subtype3':
#         subtype3.append(1)
#     if dict[i]!='subtype3':
#         subtype3.append(0)
# expr_two = list(proper_tcga.iloc[:,1])
# roc_auc2 = roc_auc_score(subtype3, expr_two)
# print('subtype3的AUC:'+str(roc_auc2))


