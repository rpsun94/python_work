from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from do_apcluster import do_apcluster
from do_spectral import do_spec
from demo_function import normalize
import numpy as np
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
# plt.show()
n_sample = Xmoon.shape[0]
for i in range(2):
    Xmoon[:,i] = normalize(Xmoon[:,i])
dism = np.ones((n_sample, n_sample))-1
for i in range(n_sample):
    for j in range(i+1, n_sample):
        dism[i,j]=np.sqrt(np.sum((Xmoon[i,:]-Xmoon[j,:])**2))
        dism[j,i] = dism[i,j]
sim = -1*dism
n_clusters, y_pred, cluster_centers_indices=do_apcluster(sim,dism,'iris',minc=2,maxc=10,recal=True,auto=True,kin=None)
c = ['r','g','b','y','k','gray','steelblue','orange','pink','purpule']
for i in range(n_sample):
    plt.scatter(Xmoon[i, 0], Xmoon[i, 1], c=c[y_pred[i]])
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=8).fit(Xmoon)
y_pred = gmm.predict(Xmoon)
print(y_pred)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1], c=y_pred,cmap='viridis')
plt.show()
n_clusters, y_pred, cluster_centers_indices=do_spec(sim,dism,'iris',minc=2,maxc=2,recal=True,auto=True,kin=None)
for i in range(n_sample):
    plt.scatter(Xmoon[i, 0], Xmoon[i, 1], c=c[y_pred[i]])
plt.show()