from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from do_apcluster import do_apcluster
from sklearn import datasets
from sklearn.cluster import KMeans
from demo_function import normalize
iris = datasets.load_iris()
print('>> shape of data:',iris.data.shape)
feature_names = iris.feature_names
class_names = iris.target_names
X = iris.data
y = iris.target
# 划分训练集、测试集  7:3
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train=X
y_train=y
n_sample = X_train.shape[0]
for i in range(4):
    X_train[:,i] = normalize(X_train[:,i])
dism = np.ones((n_sample, n_sample))-1
for i in range(n_sample):
    for j in range(i+1, n_sample):
        dism[i,j]=np.sqrt(np.sum((X_train[i,:]-X_train[j,:])**2))
        dism[j,i] = dism[i,j]
sim = -1*dism
n_clusters, y_pred, cluster_centers_indices=do_apcluster(sim,dism,'iris',minc=2,maxc=10,recal=True,auto=True,kin=None)
yy = KMeans(n_clusters=3, init='k-means++').fit_predict(X_train)
import matplotlib.pyplot as plt
y_pred = y_pred
y_pred[np.where(y_pred<0)]+=3
print(classification_report(y_train, np.array(y_pred)))

yy = yy
yy[np.where(yy>2)]-=3
print(classification_report(y_train, np.array(yy)))
plt.plot(y_train)
plt.plot(y_pred)
#plt.plot(yy)
plt.show()