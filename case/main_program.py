import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats
from minisom import MiniSom
from sklearn import datasets

from demo_function import (extend_figure, getcolorbar, interp2, link_figure,
                           normalize, paint_figure, make_dism, classify)
from do_apcluster import do_apcluster
from sklearn.cluster import KMeans
import h5py


def paint_heatmap(y_pred, plist, colors, heatmap, texts, save_name):
    for index in range((len(y_pred))):
        point = plist[index]
        plt.plot(point[1], point[0], 'o', markerfacecolor=colors[y_pred[index]],
                 markeredgecolor=colors[y_pred[index]], markersize=5, markeredgewidth=2)
    plt.imshow(heatmap, cmap='bone_r')
    plt.colorbar()
    for index, ndata in np.ndenumerate(texts):
        px, py = index
        plt.text(px, py,  '%d' % (ndata),
                 color='black', fontdict={'weight': 'bold',  'size': 15},
                 va='center', ha='center')
    plt.savefig(save_name+'U-matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


#  =====================define_param=========================================================
script_name, filename, v7, val_name, ncase, d = sys.argv
v7 = True if v7 == '1' else False
ncase = int(ncase)
d = float(d)
# filename = r'D:/data/casedata_500.mat'  # casedata_lvbo_lag-2.mat casedata_lvbo_lag-3.mat
# v7 = False  # mat 保存的 v7.3版需要用別的方法讀取, 是該版時這裡改為True
# val_name = 'casedata_500'  # 'lag__2lvbo','lag__3lvbo'
# ncase = 235  # 236 236
# d = 1  # resolution
nx = 360
ny = 91
lat_range = np.arange(90, -0.5, -1)
lon_range = np.arange(-180, 180.5, 1)  # 經緯度範圍
lon0 = [180, 360]
lat0 = [0, 71]  # 起始位置
nla = int(lat0[1]-lat0[0])
nlo = int(lon0[1]-lon0[0])  # 根據格點數使用雙線性插值或降分辨率
nla = int((nla-1)/d+1)
nlo = int((nlo-1)/d+1)
print(nla, nlo)
relearn = False  # 由於學習一次時間較長，學習過一次后僅修改圖片格式時可以將這裡改為False
save_name = val_name  # lag2 lag3 # 保存前綴
save_name += '_r='+str(d)+'_'
# -180-180, 0-90
# ============================================================================================

matfile = filename

if v7:
    data = h5py.File(matfile)
    print(data.keys())
    dd = data[val_name][:]/9.8
elif not v7:
    data = scipy.io.loadmat(matfile)
    print(data.keys())
    dd = data[val_name].T/9.8  # 轉化為位勢高度

mean_dd = np.mean(dd, axis=0)[lat0[0]:lat0[1], lon0[0]:lon0[1]]
X_train = np.ones((ncase, nla*nlo))-1
dd_a = np.ones((ncase, nla, nlo))-1
for i in range(ncase):
    # dd[180:360,0:71,i]#
    dd_a[i, :, :] = interp2(
        dd[i, lat0[0]:lat0[1], lon0[0]:lon0[1]], (nla, nlo))
#dd = dd_a
for index, ndata in np.ndenumerate(dd_a):
    case, lat, lon = index
    X_train[case, nlo*lat+lon] = ndata
max_train = np.ones([ X_train.shape[1]])-1
min_train = np.ones([ X_train.shape[1]])-1
for ii in range(X_train.shape[1]):
    max_train[ii] = np.max(X_train[:, ii])
    min_train[ii] = np.min(X_train[:, ii])
    X_train[:, ii] = normalize(X_train[:, ii])
range_train = max_train-min_train
#print(range_train)
#print(min_train)
N = X_train.shape[0]  # 样本数量
M = X_train.shape[1]  # 维度/特征数量
'''
设置超参数
'''
size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸
print("训练样本个数:{}".format(N))
print("输出网格最佳边长为:", size)

max_iter = int(N*20)

# Initialization and training
if relearn:
    som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
                  neighborhood_function='bubble')  # sigma=3 學習半徑為周圍兩圈
    som.pca_weights_init(X_train)
    som.train_batch(X_train, max_iter, verbose=False)
    heatmap = som.distance_map()  # 生成U-Matrix
    winners = []
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        winners.append([w[1], w[0]])  # x,y
    #plt.axis([0, size, 0, size])
    winners = np.array(winners)
    texts = np.ones((size, size))-1
    W = som.get_weights()
    for item in winners:
        texts[int(item[0]), int(item[1])] += 1  # 每個節點對應的樣本數
    np.savez(save_name+'heatmap.npz', heatmap=heatmap,
             winners=winners, texts=texts, W=W)
elif not relearn:
    dic = np.load(save_name+'heatmap.npz')
    heatmap = dic['heatmap']
    winners = dic['winners']
    texts = dic['texts']
    W = dic['W']
# ap聚類
# print(W.shape)
ww = W.reshape(size*size, M)
dism, plist = make_dism(ww, size)
# print(dism)
sim = -1*dism
clusters = []
n_clusters, y_pred, cluster_centers_indices = do_apcluster(
    sim, dism, save_name, minc=2)
for ii in range(len(X_train)):
    point = [winners[ii][0], winners[ii][1]]
    clusters.append(y_pred[plist.index(point)])
# ====繪製heatmap=====
colors = ['r', 'b', 'g', 'y', 'steelblue', 'gray', 'black',
          'purple', 'orange', 'cyan']  # 按該順序在U-matrix圖上顯示類別
paint_heatmap(y_pred, plist, colors, heatmap, texts, save_name)
# ====繪製合成場====
cmap = getcolorbar('GMT_no_green')
basemap_option = {
    'projection': 'cyl',
    # 'boundinglat': 20,
    # 'lon_0': 180,
    # 'resolution': 'l',
    # 'round': True,
    'llcrnrlat': 20,
    'urcrnrlat': 90,
    'llcrnrlon': 0,
    'urcrnrlon': 180,  # 指定地圖投影和邊界
}
lon = lon_range[lon0[0]:lon0[1]]
lat = lat_range[lat0[0]:lat0[1]]
nn = np.arange(0, ncase, 1)
clusters = np.array(clusters)
contourf_data0 = {
    '0': {
        'data': mean_dd,
        'cmap': cmap,
        'levels': np.linspace(-150, 150, 41),
    },
}
contour_data0 = {
    '0': {
        'data': mean_dd,
        'colors': 'k',
        'levels': np.linspace(-150, 150, 21),
        'linewidths': 0.5,
    },
}
paint_figure(lat, lon, basemap_option, save_name+'_mean.png', save=True,
                contourf=True, data_contourf=contourf_data0,
                contour=True, data_contour=contour_data0,
                # plot=True,data_plot=data_plot,
                rotation=False, tight=False)
for i in range(n_clusters):
    whichs = nn[np.where(clusters == i)]  # 哪些個例屬於某一類
    percent = len(whichs)/ncase
    dd_w = np.ones((dd.shape[1], dd.shape[2], len(whichs)))
    for ii in range(len(whichs)):
        dd_w[:, :, ii] = dd[int(whichs[ii]), :, :]  # 該類中每個個例的要素場
    weights = np.array([ heatmap[winners[ii,0], winners[ii,1]] for ii in whichs])
    wstd = np.std(weights)
    wm = np.mean(weights)
    print(wm)
    print(wstd)
    whichs_ = whichs[np.where(weights<(wm+wstd))]
    print(len(whichs),len(whichs_))
    dd_w_ = np.ones((dd.shape[1], dd.shape[2], len(whichs_)))
    for ii in range(len(whichs_)):
        dd_w_[:, :, ii] = dd[int(whichs_[ii]), :, :]  # 該類中每個個例的要素場
    bmuf = np.mean(dd_w_, axis=2)[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    # bmu = winners[whichs[weights.index(min(weights))],0]*size+winners[whichs[weights.index(min(weights))],1]
    # print(bmu)
    # print(texts[winners[whichs[weights.index(min(weights))],0],winners[whichs[weights.index(min(weights))],1]])
    # bmuf = ww[bmu,:]*range_train+min_train
    # bmuf = bmuf.reshape([nla,nlo])
    p = np.ones((dd.shape[1], dd.shape[2]))-1
    p2 = np.ones((dd.shape[1], dd.shape[2]))-1
    for index, ndata in np.ndenumerate(p):
        iy, ix = index
        # means = np.array([mean_dd[iy, ix]]*ncase)
        p[index] = scipy.stats.ttest_ind(dd_w[iy, ix, :], dd[:, iy, ix], axis=0)[
            1]  # 對合成場做顯著性檢驗,但圖中並無體現
        p2[index] = scipy.stats.ttest_ind(dd_w_[iy, ix, :], dd[:, iy, ix], axis=0)[
            1]  # 對合成場做顯著性檢驗,但圖中並無體現
    # print(np.min(p))
    dd_d = (np.mean(dd_w, axis=2)-np.mean(dd, axis=0))[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    dd_d_ = (np.mean(dd_w_, axis=2)-np.mean(dd, axis=0))[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    mask = np.ones(p.shape)
    hatches = np.ones(p.shape)-1
    hatches2 = np.ones(p2.shape)-1
    mask[np.where(p <= 0.1)] = 0
    hatches[np.where(p <= 0.05)] = 99
    hatches2[np.where(p2 <= 0.05)] = 99
    paint_data = np.mean(dd_w, axis=2)
    mask = mask[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    paint_data = paint_data[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    p = p[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    hatches = hatches[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    hatches2 = hatches2[lat0[0]:lat0[1], lon0[0]:lon0[1]]
    std_ca = np.std(paint_data)
    ppd = paint_data
    hats = hatches
    hats2 = hatches2
    ss = 150 if val_name == 'casedata_500' else 250
    contourf_data = {
        '0': {
            'data': ppd,
            'cmap': cmap,
            'levels': np.linspace(-1*ss, 1.*ss, 41),
            'data_hatches': hats,
            'hatches': ['x'],  # 若需要顯著性檢驗的陰影區,將這兩行解除註釋
        },
    }
    contour_data = {
        '0': {
            'data': ppd,
            'colors': 'k',
            'levels': np.linspace(-1*ss, 1.*ss, 21),
            'linewidths': 0.5,
        },
    }
    contourf_data2 = {
        '0': {
            'data': bmuf,
            'cmap': cmap,
            'levels': np.linspace(-1*ss, 1.*ss, 41),
            'data_hatches': hats2,
            'hatches': ['x'],  # 若需要顯著性檢驗的陰影區,將這兩行解除註釋
        },
    }
    contour_data2 = {
        '0': {
            'data': bmuf,
            'colors': 'k',
            'levels': np.linspace(-1*ss, 1.*ss, 21),
            'linewidths': 0.5,
        },
    }
    contourf_data_d = {
        '0': {
            'data': dd_d,
            'cmap': cmap,
            'levels': np.linspace(-1*ss, 1.*ss, 41),
            'data_hatches': hats,
            'hatches': ['x'],  # 若需要顯著性檢驗的陰影區,將這兩行解除註釋
        },
    }
    contour_data_d = {
        '0': {
            'data': dd_d,
            'colors': 'k',
            'levels': np.linspace(-1*ss, 1.*ss, 21),
            'linewidths': 0.5,
        },
    }
    contourf_data_d2 = {
        '0': {
            'data': dd_d_,
            'cmap': cmap,
            'levels': np.linspace(-1*ss, 1.*ss, 41),
            'data_hatches': hats2,
            'hatches': ['x'],  # 若需要顯著性檢驗的陰影區,將這兩行解除註釋
        },
    }
    contour_data_d2 = {
        '0': {
            'data': dd_d_,
            'colors': 'k',
            'levels': np.linspace(-1*ss, 1.*ss, 21),
            'linewidths': 0.5,
        },
    }
    paint_figure(lat, lon, basemap_option, save_name+'%.1f' % (percent*100)+'cluster_'+str(i)+'.png', save=True,
                 contourf=True, data_contourf=contourf_data,
                 contour=True, data_contour=contour_data,
                 # plot=True,data_plot=data_plot,
                 rotation=False, tight=False)
    paint_figure(lat, lon, basemap_option, save_name+'%.1f' % (percent*100)+'cluster_'+str(i)+'_bmu.png', save=True,
                 contourf=True, data_contourf=contourf_data2,
                 contour=True, data_contour=contour_data2,
                 # plot=True,data_plot=data_plot,
                 rotation=False, tight=False)
    paint_figure(lat, lon, basemap_option, save_name+'%.1f' % (percent*100)+'cluster_'+str(i)+'_dif.png', save=True,
                 contourf=True, data_contourf=contourf_data_d,
                 contour=True, data_contour=contour_data_d,
                 # plot=True,data_plot=data_plot,
                 rotation=False, tight=False)
    paint_figure(lat, lon, basemap_option, save_name+'%.1f' % (percent*100)+'cluster_'+str(i)+'_bmu_dif.png', save=True,
                 contourf=True, data_contourf=contourf_data_d2,
                 contour=True, data_contour=contour_data_d2,
                 # plot=True,data_plot=data_plot,
                 rotation=False, tight=False)
