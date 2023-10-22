from joblib import load
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix




#载入预测特征基因
rfecv = load('rfecv.joblib')
print(rfecv)
# 打印选择的特征数量和排名信息
print("Selected features:", rfecv.n_features_)
print("Feature rankings:", rfecv.ranking_)
#载入GEO数据及处理
warnings.filterwarnings("ignore")
file1=pd.read_csv('geo_expr_index_149.txt',sep='\t')
gene=list(file1.iloc[:,0])
file1.set_index('gene', inplace=True)
print(file1)
print(type(file1))
file2=pd.read_csv('geo_sample分型.txt',sep='\t')
sample=[]
y_values=[]
dict={}
for i in range(file2.shape[0]):
    dict[file2.iloc[i,0]]=file2.iloc[i,1]
for i in dict.keys():
    if dict[i]!='subtype2':
        sample.append(i)
        y_values.append(dict[i])
print(sample)
print(len(sample))
file2.set_index('sample', inplace=True)
file2['type'] = file2['type'].replace({'subtype1':0, 'subtype3':1})

ranking=list(rfecv.ranking_)
importgene=[]
for i in range(len(ranking)):
    if ranking[i]==1:
        importgene.append(gene[i])
print(len(importgene))
filex=file1.loc[importgene,:]
filex=filex.loc[:,sample]
print(filex)
X_import=filex.values.T
print(X_import)
print(len(X_import))
file2=pd.DataFrame({'sample':y_values})
#y=file2.values
print(filex)
print(file2)

#载入模型
svc = load('svc_model.joblib')
print(svc)
y_pred_prob = svc.predict_proba(X_import)
# 查看预测结果
print(y_pred_prob)

# 将预测概率转换为类别标签
y_pred = np.argmax(y_pred_prob, axis=1)
print(type(y_pred))
proper_tcga = pd.DataFrame(y_pred_prob, columns=['subtype1', 'subtype3'])
proper_tcga1=pd.DataFrame({'sample':sample}).join(proper_tcga)
sub=[]
for i in range(proper_tcga1.shape[0]):
    if proper_tcga1.iloc[i,1]>proper_tcga1.iloc[i,2]:
        sub.append('subtype1')
    else:
        sub.append('subtype3')
proper_tcga2=pd.DataFrame({'subtype':sub})
proper_tcga2=proper_tcga1.join(proper_tcga2)

class_tcga = list(y_pred)
print(class_tcga)
print(proper_tcga)


#ROC曲线判断拟合程度
# Subtype1
subtype1 = []
for i in sample:
    if dict[i] == 'subtype1':
        subtype1.append(1)
    else:
        subtype1.append(0)
expr_one = list(proper_tcga.iloc[:,0])
roc_auc1 = roc_auc_score(subtype1, expr_one)
fpr1, tpr1, thresholds1 = roc_curve(subtype1, expr_one)
print('subtype1的AUC:'+str(roc_auc1))
print(fpr1,tpr1)
# Subtype3
subtype3 = []
for i in sample:
    if dict[i] == 'subtype3':
        subtype3.append(1)
    else:
        subtype3.append(0)
expr_two = list(proper_tcga.iloc[:, 1])
roc_auc2 = roc_auc_score(subtype3, expr_two)
fpr2, tpr2, thresholds2 = roc_curve(subtype3, expr_two)
print('subtype3的AUC:'+str(roc_auc2))



# 绘制 ROC 曲线
plt.plot(fpr1, tpr1, color='red', label='GEO (AUC = %0.3f)' % roc_auc1)
#plt.plot(fpr2, tpr2, color='blue', label='Subtype3 (AUC = %0.3f)' % roc_auc2)
#plt.plot(fpr3, tpr3, color='blue', label='Subtype3 (AUC = %0.2f)' % roc_auc3)

# 设置图形属性
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# 显示图形并保存为文件
plt.savefig('roc_curve_geo.pdf', dpi=300)
plt.show()

