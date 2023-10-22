import pandas as pd
import numpy as np
import warnings
from joblib import load
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC

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
#file2['sample'] = file2['sample'].replace({'subtype1':1, 'subtype3':3})
X=file1.values.T
y=file2.values

rfecv = load('rfecv.joblib')
print(rfecv)
# 打印选择的特征数量和排名信息
print("Selected features:", rfecv.n_features_)
print("Feature rankings:", rfecv.ranking_)
# # RFECV
# from sklearn.svm import SVC
# svc = SVC(kernel='linear',probability=True)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# model = LogisticRegression()
#
# rfecv = RFECV(estimator=svc,          # 学习器
#               min_features_to_select=2, # 最小选择的特征数量
#               step=4,                 # 移除特征个数
#               cv=StratifiedKFold(16),  # 交叉验证次数
#               scoring='accuracy',     # 学习器的评价标准
#               verbose = 0,#详细程度控制。默认为 0，不输出进度信息。
#               n_jobs = 1#并行运行的作业数。默认为 1，表示不使用并行化。
#               ).fit(X, y)
# print(rfecv)
# X_RFECV = rfecv.transform(X)
# #print(X_RFECV)
# print(len(rfecv.ranking_))
# print("RFECV特征选择结果——————————————————————————————————————————————————")
# print("有效特征个数 : %d" % rfecv.n_features_)
# print("全部特征等级 : %s" % list(rfecv.ranking_))
# # 保存模型
# from joblib import dump
# dump(rfecv, 'rfecv.joblib')
ranking=list(rfecv.ranking_)
importgene=[]
for i in range(len(ranking)):
    if ranking[i]==1:
        importgene.append(gene[i])
print(importgene)
# output_file = 'gene_import.txt'
# with open(output_file, 'w') as file:
#     file.write("gene\n")  # 写入表头 "gene"
#     for item in importgene:
#         file.write(item + '\n')
filex=file1.loc[importgene,:]
print(filex)
X_import=filex.values.T
print(X_import)
print(len(X_import))

X_import_df = pd.DataFrame(X_import)
dict1={}
for i in range(X_import_df.shape[0]):
    dict1[X_import_df.iloc[i,0]]=sample[i]
print(dict1)
print(X_import)
print(len(X_import))
print(y)


#五折交叉验证

svc = SVC(kernel='poly',probability=True,C=0.015)
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # n_splits 表示将数据集分成几份
x=0
sample_all=[]
result_matrix = np.empty((0, 2))
for train_index, test_index in kf.split(X_import, y):
    x=x+1
    sample1=[]
    X_train, X_test = X_import[train_index], X_import[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_test_df = pd.DataFrame(X_test)
    for i in  range(X_test_df.shape[0]):
        sample1.append(dict1[X_test_df.iloc[i,0]])
        sample_all.append(dict1[X_test_df.iloc[i,0]])
    # print(X_train)
    # print(y_train)
    # print(x)
    # # 进行训练和测试
    svc.fit(X_train, y_train)
    y_pred_prob = svc.predict_proba(X_test)
    proper_tcga = pd.DataFrame(y_pred_prob, columns=['subtype1', 'subtype3'])
    print(proper_tcga)
    result_matrix = np.vstack((result_matrix, y_pred_prob))
    result_matrix = pd.DataFrame(result_matrix, columns=['subtype1', 'subtype3'])
    #dump(svc, str(x)+'.joblib')

print(result_matrix)


print(sample_all)
#subtype1
subtype1=[]
for i  in sample_all:
    if dict[i]=='subtype1':
        subtype1.append(1)
    if dict[i]!='subtype1':
        subtype1.append(0)
expr_one = list(result_matrix.iloc[:,0])
roc_auc1 = roc_auc_score(subtype1, expr_one)
print('subtype1的AUC:'+str(roc_auc1))
fpr1, tpr1, thresholds1 = roc_curve(subtype1, expr_one)

#subtype3
subtype3=[]
for i  in sample_all:
    if dict[i]=='subtype3':
        subtype3.append(1)
    if dict[i]!='subtype3':
        subtype3.append(0)
expr_two = list(result_matrix.iloc[:,1])
roc_auc2 = roc_auc_score(subtype3, expr_two)
print('subtype3的AUC:'+str(roc_auc2))
fpr2, tpr2, thresholds2 = roc_curve(subtype3, expr_two)


plt.plot(fpr1, tpr1, color='red', label='TCGA 5-fold cross-validation(AUC = %0.3f)' % roc_auc1)
# 设置图形属性
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 显示图形并保存为文件
plt.savefig('roc_curve_50cross.pdf', dpi=300)
plt.show()
