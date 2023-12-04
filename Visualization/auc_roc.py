from sklearn import metrics
import pylab as plt
import warnings; warnings.filterwarnings('ignore')
import numpy as np


y_test = np.array([1] * 135 + [0] * 378, dtype='float32')  # human独立测试数据


list1 = []
with open('NetNGlyc_hm') as f:
    content = f.readlines()
    for i in content:
        list1.append(float(i))

list2 = []
with open('GlycoPP_BPP') as f:
    content = f.readlines()
    for i in content:
        list2.append(float(i))

list3 = []
with open('GlycoPP_CPP') as f:
    content = f.readlines()
    for i in content:
        list3.append(float(i))

list4 = []
with open('GlycoPP_BPP+ASA') as f:
    content = f.readlines()
    for i in content:
        list4.append(float(i))

list5 = []
with open('GlycoEP_BPP') as f:
    content = f.readlines()
    for i in content:
        list5.append(float(i))

list6 = []
with open('GlycoEP_CPP') as f:
    content = f.readlines()
    for i in content:
        list6.append(float(i))

list7 = []
with open('GlycoEP_BPP+ASA') as f:
    content = f.readlines()
    for i in content:
        list7.append(float(i))

list8 = []
with open('HM_ind_Result.txt') as f:
    content = f.readlines()
    for i in content:
        list8.append(float(i))



# 画图部分
fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, list1)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, list2)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc2 = metrics.auc(fpr2, tpr2)

fpr3, tpr3, threshold3 = metrics.roc_curve(y_test, list3)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc3 = metrics.auc(fpr3, tpr3)

fpr4, tpr4, threshold4 = metrics.roc_curve(y_test, list4)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc4 = metrics.auc(fpr4, tpr4)

fpr5, tpr5, threshold5 = metrics.roc_curve(y_test, list5)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc5 = metrics.auc(fpr5, tpr5)

fpr6, tpr6, threshold6 = metrics.roc_curve(y_test, list6)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc6 = metrics.auc(fpr6, tpr6)

fpr7, tpr7, threshold7 = metrics.roc_curve(y_test, list7)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc7 = metrics.auc(fpr7, tpr7)

fpr8, tpr8, threshold8 = metrics.roc_curve(y_test, list8)       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
roc_auc8 = metrics.auc(fpr8, tpr8)


plt.figure(figsize=(6,6))
# plt.title('HM_model')
plt.plot(fpr1, tpr1, '#2DABB9', label = 'NetNGlyc AUC=%0.3f' % roc_auc1)
plt.plot(fpr2, tpr2, '#EFBB89', label = 'GlycoPP_BPP AUC=%0.3f' % roc_auc2)
plt.plot(fpr3, tpr3, '#D584BE', label = 'GlycoPP_CPP AUC=%0.3f' % roc_auc3)
plt.plot(fpr4, tpr4, '#D1D097', label = 'GlycoPP_BPP+ASA AUC=%0.3f' % roc_auc4)
plt.plot(fpr5, tpr5, '#C4B6D0', label = 'GlycoEP_BPP AUC=%0.3f' % roc_auc5)
plt.plot(fpr6, tpr6, '#B4C8E1', label = 'GlycoEP_CPP AUC=%0.3f' % roc_auc6)
plt.plot(fpr7, tpr7, '#A0D494', label = 'GlycoEP_BPP+ASA AUC=%0.3f' % roc_auc7)
plt.plot(fpr8, tpr8, '#BF3D3D', label = 'N-GlycoPred AUC=%0.3f' % roc_auc8)



plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("HM_ind.png")
plt.show()
plt.close()


