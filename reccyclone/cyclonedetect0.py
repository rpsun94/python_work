import numpy
import netCDF4 as nc
import scipy.ndimage
from demo_function import datelist
from demo_function import make_global_extend
from elev import elev
from const import grav, R_earth
import numba
import os
from demo_function import paint_figure, interp2
import matplotlib.pyplot as plt


# @numba.jit(parallel=True, nopython=False)
def use_detect(cyclone_num, cyclone_pres, cyclone_lat, cyclone_lon,
               cyclone_lap,
               nummap, filename, jstart, jend, istart, iend, lat,
               lon, cbl, cbl2, cbl3, bl, op,
               Di, Dj, ELEV):
    try:
        temp_dic = numpy.load('prepare/temp/inum.npz')
        stnum = temp_dic['inum']+1
        for inum in numba.prange(0, stnum):
            temp_dic = numpy.load('prepare/temp/'+str(inum)+'.npz')
            cyclone_num[inum] = temp_dic['cyclone_num']
            cyclone_pres[inum] = temp_dic['cyclone_pres']
            cyclone_lat[inum] = temp_dic['cyclone_lat']
            cyclone_lon[inum] = temp_dic['cyclone_lon']
    except:
        stnum = 0
    # inum.npz 用于存放当前程序的进程，以防止程序意外退出时不得不重新进行。
    # 若不存在inum.npz 则从头开始运行程序。
    for inum in numba.prange(stnum, nummap):
        ds = nc.Dataset(filename[inum][0])
        varxy = ds.variables['msl'][filename[inum][1], 0:Nj, 0:Ni]
        varxy, nlo, nla = make_global_extend(varxy, lon, lat, derta=derta,
                                             vector=False, north=True,
                                             south=False)
        for index, ndata in numpy.ndenumerate(nlo):
            if ndata < 0:
                nlo[index] += 360
            elif ndata > 360:
                nlo[index] -= 360
        # 这里将经度转化为0-360°
        ds.close()
        all_point, varreg = detect_potential_point1(
            cyclone_num, cyclone_pres, cyclone_lat,
            cyclone_lon, cyclone_lap, nummap, jstart, jend,
            istart, iend, nla, nlo, cbl,
            cbl2, cbl3, bl, op, Di, Dj, ELEV, varxy)
        rr=4
        v4 = interp2(varreg,(int(varreg.shape[0]/(rr/Dj)),int(varreg.shape[1]/(rr/Di))))
        # 把mslp投影到rr°*rr° 分辨率的网格中，以计算rr°*rr° 分辨率的laplacian（P）
        Laplacian_p = scipy.ndimage.filters.laplace(v4, mode='wrap')/(rr**2)
        Laplacian_p = interp2(Laplacian_p,varreg.shape)

        unique_point = numpy.unique(all_point, axis=0)
        # 排序 找出所有不重复的点
        kk = len(unique_point)
        varminloc = numpy.ones((kk), dtype=numpy.float64)-10000
        lonloc = numpy.ones((kk), dtype=numpy.float64)-10000
        latloc = numpy.ones((kk), dtype=numpy.float64)-10000
        varminloc = unique_point[:, 0]
        lonloc = unique_point[:, 1]
        latloc = unique_point[:, 2]
        icyclone = len(unique_point)
        iccent, cccentpres, cccentlon, cccentlat, cccentlap = detect_potential_point2(
            cyclone_num, cyclone_pres, cyclone_lat, cyclone_lon, cyclone_lap,
            icyclone,
            varminloc, latloc, lonloc, nummap, jstart, jend, istart, iend, nla,
            nlo, cbl, cbl2, cbl3, bl, op, Di, Dj, ELEV, Laplacian_p,
            varxy,lat,lon)
        # 以下是测试，查看是否识别到异常的中心
        #=================
        # basemap_option = {
        #     'projection': 'npstere',
        #     'boundinglat': 30,
        #     'lon_0': 180,
        #     # 'resolution':'l',
        #     'round': True,
        # }
        # contour_data = {
        #     '0':{'data': var0/100,
        #           'levels': numpy.arange(980,1100,2.5),
        #           'colors': 'b',
        #           'clabel': True
        #     }
        #     }
        # contourf_data = {
        #     '0':{'data': Laplacian_p[derta::,derta:Ni+derta],
        #           'levels': numpy.arange(-2,2,0.25),
        #           'cmap': plt.get_cmap('bwr'),
        #     }
        #     }
        # latlons = numpy.ones((iccent,2))-1
        # for icc in range(iccent):
        #     latlons[icc,0], latlons[icc,1] = cccentlat[icc], cccentlon[icc]
        #     print(cccentlat[icc], cccentlon[icc], cccentlap[icc])
        # scatter_data = {'0':{
        #     'data':latlons,
        #     'colors':'k',
        #     }
        #     }
        # llo = lon[0,:]
        # lla = lat[:,0]
        # paint_figure(lla, llo, basemap_option, '1.png', save=True,contour=True,data_contour=contour_data,scatter_latlon=True,data_scatter_latlon=scatter_data,contourf=True,data_contourf=contourf_data )
        # os.system('pause')
        #=================
        # 保存数据
        cyclone_num[inum] = iccent
        for item in numba.prange(iccent):
            cyclone_pres[inum, item] = cccentpres[item]
            cyclone_lon[inum, item] = cccentlon[item]
            cyclone_lat[inum, item] = cccentlat[item]
            cyclone_lap[inum, item] = cccentlap[item]
        print(str(inum)+' is finished, potential point: '+str(iccent))
        numpy.savez('prepare/temp/'+str(inum)+'.npz',inum=inum,cyclone_num=cyclone_num[inum],cyclone_pres=cyclone_pres[inum],cyclone_lat=cyclone_lat[inum],cyclone_lon=cyclone_lon[inum],cyclone_lap=cyclone_lap[inum])
        numpy.savez('prepare/temp/inum.npz', inum=inum)
    print('all finished')
    return cyclone_num, cyclone_pres, cyclone_lat, cyclone_lon, cyclone_lap


@numba.jit(parallel=False, nopython=True)
def detect_potential_point1(cyclone_num, cyclone_pres, cyclone_lat,
                            cyclone_lon, cyclone_lap,
                            nummap, jstart, jend, istart, iend, lat,
                            lon, cbl, cbl2, cbl3, bl, op,
                            Di, Dj, ELEV, varxy):

    varxy = varxy/100
    lonreg = lon[jstart:jend, istart:iend]
    latreg = lat[jstart:jend, istart:iend]
    varreg = varxy[jstart:jend, istart:iend]
    varregdim = varreg.shape
    mm, nn = varregdim
    cnum = 0
    loop_lat = numpy.arange(int(cbl3/Dj)-1,
                            (varregdim[0]-bl/Dj-cbl3/Dj),
                            int((bl-op)/Dj))
    loop_lon = numpy.arange(int(cbl3/Di)-1,
                            (varregdim[1]-bl/Di-cbl3/Di),
                            int((bl-op)/Di))# for laplacian around the edge cannot be trust
    cpres = numpy.ones((mm*nn), dtype=numba.float64)-10000
    lonc = numpy.ones((mm*nn), dtype=numba.float64)-10000
    latc = numpy.ones((mm*nn), dtype=numba.float64)-10000
    dx = int(bl/Di)
    dy = int(bl/Dj)
    # ==========find potential center
    for i in loop_lat:
        for j in loop_lon:
            varbox = varreg[int(i+1):int(i+dy), int(j+1):int(j+dx)].copy()
            vmin = numpy.min(varbox)
            vmax = numpy.max(varbox)
            loca = numpy.where(varbox == vmin)
            ic = loca[0][0]
            jc = loca[1][0]
            if vmin < threshold and (vmax-vmin) > gradp: # 中心小于阈值并范围内气压差值大于阈值
                lonc[cnum] = lonreg[int(ic+i), int(jc+j)]
                latc[cnum] = latreg[int(ic+i), int(jc+j)]
                cpres[cnum] = vmin
                cnum = cnum+1
    # ==========delete same points
    all_point = numpy.ones((int(cnum), 3), dtype=numba.float64)-1
    for ii in numba.prange(cnum):
        all_point[ii, 0] = cpres[ii]
        all_point[ii, 1] = lonc[ii]
        all_point[ii, 2] = latc[ii]
    return all_point, varreg


@numba.jit(parallel=False, nopython=True)
def detect_potential_point2(cyclone_num, cyclone_pres, cyclone_lat,
                            cyclone_lon, cyclone_lap, icyclone, varminloc, latloc,
                            lonloc,
                            nummap, jstart, jend, istart, iend, lat,
                            lon, cbl, cbl2, cbl3, bl, op,
                            Di, Dj, ELEV, Laplacian_p, varxy,lat0,lon0):
    prescent = numpy.ones((icyclone), dtype=numba.float64)-1
    lla = lat0[:,0]
    llo = lon0[0,:]
    # ==========absorb points too near
    for ii in numba.prange(icyclone):
        for k in numba.prange(icyclone):
            if (k > ii
                and (abs(lonloc[k]-lonloc[ii]) <= cbl or abs(lonloc[k]-lonloc[ii]) >= 360-cbl)
                and (abs(latloc[k]-latloc[ii]) <= cbl)
                and (varminloc[k] >= varminloc[ii])):
                prescent[k] = 1
                #break
    # ==========strongest potential center
    icenter = 0
    loop_ct = numpy.where(prescent == 0)[0]
    lct = len(loop_ct)
    centerpres = numpy.ones((lct), dtype=numba.float64)-1
    centerlon = numpy.ones((lct), dtype=numba.float64)-1
    centerlat = numpy.ones((lct), dtype=numba.float64)-1
    for ii in loop_ct:
        centerpres[icenter] = varminloc[ii]
        centerlon[icenter] = lonloc[ii]
        centerlat[icenter] = latloc[ii]
        icenter += 1
    iccentt = 0
    ccentpres = numpy.ones((icenter), dtype=numba.float64)-1
    ccentlon = numpy.ones((icenter), dtype=numba.float64)-1
    ccentlat = numpy.ones((icenter), dtype=numba.float64)-1
    logic = numpy.ones((4), dtype=numba.float64)-1
    # east, west, north, south
    # ==========at least two potential center around strongest ones
    for ii in numba.prange(icenter):
        for k in numba.prange(icyclone):
            if ((abs(centerlon[ii]-lonloc[k]) <= cbl2 or (abs(centerlon[ii]-lonloc[k])>=360-cbl2)
               and abs(centerlat[ii]-latloc[k]) <= cbl2)):
                if centerlon[ii] < lonloc[k] or centerlon[ii]-lonloc[k]>=360-cbl2:
                    logic[0] = 1
                elif centerlon[ii] > lonloc[k] or centerlon[ii]-lonloc[k]<=cbl2-360:
                    logic[1] = 1
                if centerlat[ii] < latloc[k]:
                    logic[2] = 1
                elif centerlat[ii] > latloc[k]:
                    logic[3] = 1
        if numpy.all(logic == 1):
            ccentpres[iccentt] = centerpres[ii]
            ccentlon[iccentt] = centerlon[ii]
            ccentlat[iccentt] = centerlat[ii]
            iccentt = iccentt+1
    iccent = 0
    cccentpres = numpy.ones((int(iccentt)), dtype=numba.float64)-1
    cccentlon = numpy.ones((int(iccentt)), dtype=numba.float64)-1
    cccentlat = numpy.ones((int(iccentt)), dtype=numba.float64)-1
    cccentlap = numpy.ones((int(iccentt)), dtype=numba.float64)-1
    for ii in numba.prange(iccentt):
        x = int(numpy.where(llo==ccentlon[ii])[0][0])
        y = int(numpy.where(lla==ccentlat[ii])[0][0])
        lap_ = Laplacian_p[derta::,derta:Ni+derta]
        lap=lap_[y,x]
        if ELEV[y, x] < 1500 and lap > standard_lap:
            cccentpres[iccent] = ccentpres[ii]
            cccentlon[iccent] = ccentlon[ii]
            cccentlat[iccent] = ccentlat[ii]
            cccentlap[iccent] = lap
            iccent = iccent+1
    return iccent, cccentpres, cccentlon, cccentlat, cccentlap


# ==========================load_data=========================================
date_list = datelist((1979, 1, 1, 0), (2019, 8, 31, 18), step='hours=6',
                     remodel='date')
fmty = '%04d'
fmtm = '%02d'
head_path = r'D:\data\era_int_mslp_6h_05'  # 存放路径
nc_name = ''  # 文件名
filename = [[head_path+'\\'+nc_name+'.nc', i] for i in range(len(date_list))]
inum = len(date_list)
# The record number of the variable to be decoded
# recnum=211;
# The name of the variable to be analyzed
varname = 'MSLP'
# Display GDS or not,disgds='y' for displaying,disgds='n' for not displaying,
# usually disgds='y' when it is the first time to decode a specific grib file,
# which will make you understand the structure of the data.When you want
# to extract data from the grib file, you must set disgds='n'
disgds = 'n'

# The starting latitude
La1 = 90
# The starting longitude
Lo1 = 0
# The terminating latitude
La2 = -90
# The terminating longitude
Lo2 = 359.5
# The resolution in x-direction
Di = 0.5
# The resolution in y-direction
Dj = 0.5
# The Earth's angular velocity
omega = 7.29e-5
# The Air density
ro = 1.29
# earth's radius(km)
R = R_earth/1000
# m/s^2 % gravity
g = grav
# m/s^2 % gravity
# Define the area analyzed, area='wp' for west pacific, area='gb' for
# global
area = 'nt'
# Define the contour type,ct='pc' for pcolor,ct='cn' for contour,ct='cf' for
# contourf
# ct     = 'cn'
# number of contour levels
# cn     = 31
# The colormap used for filled contour
# concol = 'jet'
# Searching Box length, in degree
bl = 4
# Overlapping part between the two neighbouring box,in degree
op = 2*Dj
# The threshold value for defining the cyclone
threshold = 1020.
# The threshold pressure gradient for defining the cyclone
gradp = 2.
# Center search box half-length
cbl = 5
# Closed center searching box
cbl2 = 5
#
cbl3 = 0
#
derta = 40
#
standard_lap = 0.18
# --------------------------------------------------------------------------
if (area == 'wp'):
    lonmin = 0
    lonmax = 160
    latmin = 20
    latmax = 80
if (area == 'gb'):
    lonmin = 0
    lonmax = 359.5
    latmin = -90
    latmax = 90
if (area == 'nt'):
    lonmin = 0
    lonmax = 359.5
    latmin = 25
    latmax = 90
# The grid number in x-direction
Ni = int((lonmax-lonmin)/Di+1)
# The grid number in y-direction
Nj = int((latmax-latmin)/Di+1)
lon = numpy.ones([Nj, Ni])-1
lat = numpy.ones([Nj, Ni])-1
for j in range(0, Nj):
    for i in range(0, Ni):
        lon[j, i] = Lo1+Di*(i)
        lat[j, i] = La1-Dj*(j)
ELEV_temp, LAT, LON = elev(latmin, latmax, lonmin, lonmax, resolution_lon=Di,
                           resolution_lat=Dj, mode=360)
ELEV0 = ELEV_temp[::-1, :]
lla = numpy.ones((len(LAT), len(LON)))-1
llo = numpy.ones((len(LAT), len(LON)))-1
for i in range(len(LAT)):
    llo[i, :] = LON
for j in range(len(LON)):
    lla[:, j] = LAT
ELEV, nlo, nla = make_global_extend(ELEV0, llo, lla, derta=derta,
                                    vector=False, north=True,
                                    south=False)
ELEV=ELEV0
ny, nx = nla.shape
istart = 0
iend = nx
jstart = 0
jend = ny
nummap = inum
cyclone_num = numpy.ones((nummap))-10000
cyclone_pres = numpy.ones((nummap, 500))-10000
cyclone_lon = numpy.ones((nummap, 500))-10000
cyclone_lat = numpy.ones((nummap, 500))-10000
cyclone_lap = numpy.ones((nummap, 500))-10000
# ===================================compute =================================
cyclone_num, cyclone_pres, cyclone_lat, cyclone_lon,cyclone_lap = use_detect(
    cyclone_num, cyclone_pres, cyclone_lat, cyclone_lon, cyclone_lap,
    nummap, filename, jstart, jend, istart, iend, lat,
    lon, cbl, cbl2, cbl3, bl, op,
    Di, Dj, ELEV)
