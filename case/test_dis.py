import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw_distance
from demo_function import normalize
from do_apcluster import do_apcluster
lines = [
    np.array([[0.5,0],[1,0.025],[1.5,0.075],[2,0.2],[2.5,0.5],[3,1],[3.5,2],[4,3.5],[4.5,4.5]]),
]
X = np.mean(lines[0][:,0])
Y = np.mean(lines[0][:,1])
lines.append(lines[0][::-1]+0.1)
lines.append(np.array([[2*X-lines[0][i,0],2*Y-lines[0][i,1]] for i in range(len(lines[0]))]))
lines.append(np.array([[5.5,0],[6,0.05],[6.5,0.12],[7,0.25],[7.5,0.5],[7,0.75],[6.5,0.87],[6,0.94]]))
lines.append(np.array([[5.5,0],[6,0.05],[6.5,0.12],[7,0.25],[6.5,0.87],[7,0.75],[7.5,0.5],[6,0.94]]))
para=[]
for line in lines:
    line_ = np.array(line)
    plt.plot(line_[:,0],line_[:,1],linewidth=2.5)
    plt.scatter(line_[0,0],line_[0,1],s=15)
    X = np.mean(line_[:,0])
    Y = np.mean(line_[:,1])
    VX = np.mean((line_[:,0]-X)**2)
    VY = np.mean((line_[:,1]-Y)**2)
    VXY = np.mean((line_[:,0]-X)*(line_[:,1]-Y))
    alpha = np.mean([np.arctan(((line_[i+1,1]-line_[i,1])/(line_[i+1,0]-line_[i,0])-(line_[i+1,1]-line_[i,1])/(line_[i+1,0]-line_[i,0]))/1+((line_[i+1,1]-line_[i,1])/(line_[i+1,0]-line_[i,0]))*((line_[i,1]-line_[i-1,1])/(line_[i,0]-line_[i-1,0]))) for i in range(1,len(line_)-1)])
    print(X,Y,VX,VY,VXY,alpha)
    para.append(np.array([X,Y,VX,VY,VXY,alpha]))
para = np.array(para)
for i in range(6):
    para[:,i] = normalize(para[:,i])

dism_c = np.ones((5,5))-1
dism_d = np.ones((5,5))-1
for i in range(5):
    for j in range(i+1,5):
        dism_d[i,j] = dtw_distance(lines[i], lines[j], method='ed')
        dism_d[j,i] = dism_d[i,j]
        dism_c[i,j] = np.mean((para[i]-para[j])**2)
        dism_c[j,i] = dism_c[i,j]
dism_d_ = (dism_d-np.min(dism_d))
dism_d_/=np.max(dism_d_)
dism_c_ = (dism_c-np.min(dism_c))
dism_c_/=np.max(dism_c_)
print(dism_c_)
print(dism_d_)
plt.savefig('orignal_path.png',dpi=300,bbox_inches='tight')
plt.close()
sim_c = -1*dism_c
sim_d = -1*dism_d
n_clusters_c, y_pred_c, cluster_centers_indices=do_apcluster(sim_c,dism_c,'iris',minc=2,maxc=10,recal=True,auto=True,kin=None,m1=0.1)
n_clusters_d, y_pred_d, cluster_centers_indices=do_apcluster(sim_d,dism_d,'iris',minc=2,maxc=10,recal=True,auto=True,kin=None,m1=0.1)
print(n_clusters_c,n_clusters_d)
print(y_pred_c,y_pred_d)
c=['r','g','b','gray']
for i in range(5):
    line_ = lines[i]
    plt.plot(line_[:,0],line_[:,1],c=c[y_pred_c[i]],linewidth=2.5)
    plt.scatter(line_[0,0],line_[0,1],s=15,c=c[y_pred_c[i]])
plt.savefig('cluster_ori.png',dpi=300,bbox_inches='tight')
plt.close()
for i in range(5):
    line_ = lines[i]
    plt.plot(line_[:,0],line_[:,1],c=c[y_pred_d[i]],linewidth=2.5)
    plt.scatter(line_[0,0],line_[0,1],s=15,c=c[y_pred_d[i]])
plt.savefig('cluster_dtw.png',dpi=300,bbox_inches='tight')
plt.close()
