import json
import numpy
from demo_function import paint_timeseries
from demo_function import mkdir
from sklearn.cluster import spectral_clustering
import numpy as np


def CHS(dism, cluster_centers_indices, labels):
    n_clusters = len(cluster_centers_indices)
    if n_clusters > 1:
        where_mindis = np.where(np.mean(dism**2,axis=0)==np.min(np.mean(dism**2,axis=0)))[0][0]
        c=0
        h=0
        for k in range(n_clusters):
            ncl = len(np.where(labels==k)[0])
            class_members = labels == k
            c += np.sum((dism[cluster_centers_indices[k],:][class_members])**2)
            h += (dism[where_mindis,:][cluster_centers_indices[k]])**2*ncl
        chk = (h/c)*(dism.shape[0]-n_clusters)/(n_clusters-1)
        print(chk, n_clusters)
        return chk
    elif n_clusters <= 1:
        return 0


def silhoueffe_score(dism, labels):
    n = dism.shape[0]
    n_c = np.max(labels)+1
    if n_c > 1:
        ss = np.ones((n))-1
        for i in range(n):
            which = labels[i]
            class_members = (labels==which)
            ai = np.mean(dism[i,:][class_members])
            bi_l = np.ones((n_c-1))-1
            ki = -1
            for k in range(n_c):
                if k != labels[i]:
                    ki+=1
                    which = k
                    class_members = labels==which
                    bi_l[ki] = np.mean(dism[i,:][class_members])
            bi = np.min(bi_l)
            si = (bi-ai)/max(ai,bi)
            ss[i] = si
        return ss
    elif n_c <=1:
        return 0

def decide_center(dism,labels):
    # n = dism.shape[0]
    n_c = np.max(labels)+1
    nn = np.arange(len(labels))
    if n_c > 1:
        centers = np.ones((n_c))-1
        for i in range(n_c):
            class_members = nn[labels==i]
            which=class_members[0]
            print(which,dism.shape)
            dis = np.mean(dism[which,:][class_members])
            for member in class_members:
                ai = np.mean(dism[member,:][class_members])
                if ai<dis:
                    dis = ai
                    which = member
            centers[i]=which
        return centers
    elif n_c <=1:
        return 0

def do_spec(sim,dism,name,minc=2,maxc=10,recal=True,auto=True,kin=None):
    sim = np.exp(-1*dism/np.std(dism))
    if recal:
        krange = numpy.arange(minc,maxc+1)
        # ch = []
        kr = []
        sc = []
        clus=[]
        for k in krange:
            labels = spectral_clustering(sim, n_clusters=int(k)) # preference=-50,preference=numpy.min(sim),
            #cluster_centers_indices = af.cluster_centers_indices_
            y_pred = labels
            n_clusters = int(k)
            if n_clusters>maxc:
                print(n_clusters)
            if n_clusters>=minc and n_clusters<=maxc:
                # chk = CHS(dism, cluster_centers_indices, y_pred)
                # ch.append(chk)
                si = np.mean(silhoueffe_score(dism, y_pred))
                sc.append(si)
                kr.append(k)
                clus.append(n_clusters)
        # cha=np.array(ch)
        # cha/=np.max(cha)
        # cha*=np.max(sc)
        # cha = list(cha)
        # js_dic = {
        #     #'ch': ch,
        #     'sc': sc,
        #     'kr': kr,
        #     'clus': clus,
        #     }
        # # r_c = np.arange(min(clus), max(clus)+1, 1)
        # # r_ch = numpy.ones((len(r_c)))-1
        # # for iii in range(len(ch)):
        # #     if ch[iii]>r_ch[clus[iii]-min(clus)]:
        # #         r_ch[clus[iii]-min(clus)] = ch[iii]
        # path_save = 'prepare'
        # mkdir(path_save)
        # file_save = 'judge_'+name+'.json'
        # with open(path_save + '\\' + file_save,'w') as outfile :
        #     json.dump(js_dic,outfile)
        dic_y = {#'0':
        #          {'y': np.array(cha),
        #           'color': 'b',
        #           },
                 '1':
                 {'y': np.array(sc),
                  'color': 'r',
                  }
                 }
        paint_timeseries(np.array(kr), dic_y, 'preference', 'Index', 'prepare/'+'judge_'+name+'.png')
        # dic_y = {'0':
        #          {'y': r_ch,
        #           'color': 'b',
        #           },
        #          }
        # paint_timeseries(r_c, dic_y, 'clusters', 'Index', 'prepare/'+'judge2_'+name+'.png')


    if not recal:
        fid = open('prepare/judge_'+name+'.json')
        jc_dic = json.load(fid)
        fid.close()
        kr = jc_dic['kr']
        #ch = jc_dic['ch']
        sc = jc_dic['sc']
    if auto:
        judge = numpy.array(sc)
        judge=list(judge)
        k=kr[judge.index(max(judge))]
    elif not auto:
        k=kin
    labels = spectral_clustering(sim, n_clusters=int(k))  # preference=-50,preference=numpy.min(sim),
    cluster_centers_indices = decide_center(dism, labels)
    y_pred = labels
    n_clusters = k
    #os.system('pause')
    return n_clusters, y_pred, cluster_centers_indices
