import numpy
import json
from dirvar import dirvarvort
from const import R_earth,Pi#unit of R is m now
from demo_function import spherical_distance, spherical_ang
from demo_function import mkdir,datelist
import os
import matplotlib.pyplot as plt
from cyclone_explosive import cyclone_explosive,get_wind
from demo_function import paint_figure
import netCDF4 as nc
#from numpy.linalg import norm as norm
test = False
name = r'prepare/analyse/cyclone_info2019.json'
fid      = open(name)
dic      = json.load(fid)
fid.close()
locals().update(dic)
iobs=len(hour)
dt=6
#cyclone movement speed threshold in m/s
Vmax=50
#One of search radius threshold in km
Lr=500
path_save = r'prepare/analyse/tracks'
mkdir(path_save)
#Definition of original area
latthresholdsouth=-90
latthresholdnorth=90
# lonthresholdwest=230;
# lonthresholdeast=360;
lonthresholdwest=0
lonthresholdeast=360
#All cyclone are set to be newborn system before tracking precedure
#newborn(1:iobs,1:25)='y'
newborn = []
#cyclone_prest(1:100)=0;
cyclone_prest = numpy.ones([1000])-1
cyclone_lapt = numpy.ones([1000])-1
#lont(1:100)=0;
lont = numpy.ones([1000])-1
#latt(1:100)=0;
latt = numpy.ones([1000])-1
#yeart(1:100)=0;
yeart = numpy.ones([1000])-1
#montht(1:100)=0;
montht = numpy.ones([1000])-1
#dayt(1:100)=0;
dayt = numpy.ones([1000])-1
#hourt(1:100)=0;
hourt = numpy.ones([1000])-1
#cyclone_prestt(1,1:100)=0;
cyclone_prestt = []
cyclone_laptt = []
#lontt(1,1:100)=0;
lontt = []
#lattt(1,1:100)=0;
lattt  = []
#yeartt(1,1:100)=0;
yeartt  = []
#monthtt(1,1:100)=0;
monthtt  = []
#daytt(1,1:100)=0;
daytt  = []
hourtt  = []
#vortvel(1,1:100,1:2)=0;

# % if dt<=24
# %     condition(1)=1;
# % end
cyclonettnum=0;
Vnorm = numpy.ones([1000])-1
V = numpy.ones([1000],dtype = 'complex')-(1+0j)
iobs_ilter = list(range(iobs))
for i in iobs_ilter :
    #print(cyclone_num[i])
    cyclone_num_i = list(range(cyclone_num[i]))
    add_newborn = [True for num in cyclone_num_i]
    newborn.append(add_newborn)

# 去除无中心的时刻
delete     = numpy.where(numpy.array(cyclone_num)==0)
for item in delete[0] :
    iobs_ilter.remove(item)


for i in iobs_ilter :
    print('doing No.'+str(i)+', '+str(cyclonettnum))
    loc = iobs_ilter.index(i)
    cyclone_num_i = list(range(cyclone_num[i]))
    for k in cyclone_num_i :
        if newborn[i][k] :
            # 对某时刻的某个中心，当其为未使用过的中心时：
            ctime=0
            cyclone_prest[ctime]=cyclone_pres[i][k]
            cyclone_lapt[ctime] = cyclone_lap[i][k]
            lont[ctime]=cyclone_lon[i][k]
            latt[ctime]=cyclone_lat[i][k]
            yeart[ctime]=year[i]
            montht[ctime]=month[i]
            dayt[ctime]=day[i]
            hourt[ctime]=hour[i]
            newborn[i][k]= False
            stored_point = [latt[ctime],lont[ctime]]
            prev_point= [latt[ctime],lont[ctime]]
            n_stored = ctime
            for itime in iobs_ilter[loc+1::] :  # 遍历其后的所有时刻的所有中心
                condition = numpy.array([0,0,0,0])
                t = itime
                cyclones = cyclone_num[t]
                if ctime == 0 :
                    tprev = i
                cposition = numpy.ones([cyclones])-1
                distance = numpy.ones([cyclones])-1
                disvalid = numpy.ones([cyclones])-1
                V = numpy.ones([cyclones])-1
                # logic_opsite = numpy.ones([cyclones])-1
                #cnum is the number of cyclone satisfying the condition
                #1,3,4,5 in time t
                cnum=0
                search= False
                #if cyclone_num(t)~=0
                for kk in range(cyclones) :
                    if newborn[t][kk] :  # 若这个中心也是未使用过的
                        distance[kk] = spherical_distance( [latt[ctime],lont[ctime]] , [ cyclone_lat[t][kk] , cyclone_lon[t][kk]] , R = R_earth)/1000
                        Vnorm[kk] = distance[kk]*1000/((t-tprev)*dt)/3600.#the speed of cyclone,but not the real one 计算移动距离和速度
                        if ctime == 0 :
                            Rs=Lr
                            if Vnorm[kk]<Vmax :  # 若其是起始，移动速度小于阈值即可
                                condition[0] = 1
                               # 'speed less than 40'
                               # pause
                        elif ctime != 0 :
                            if Vnorm[kk] < Vmax :  # 若并非起始，移动速度小于阈值的同时，移动距离、前后移动的夹角也有要求
                                condition[0] = 1
                                Rs=max(Lr,3*dt*3600*abs(V0)/1000.)
                                A = prev_point
                                B = [latt[ctime], lont[ctime]]
                                C = [cyclone_lat[t][kk] , cyclone_lon[t][kk]]
                                if spherical_distance(A, B) == 0:
                                    A = stored_point
                                theta = 180-spherical_ang(A, B, C)/Pi*180  # 计算三点曲边三角形的夹角，以确定前后两速度的夹角
                                if theta<dirvarvort(Vnorm[kk]) :
                                    condition[3] = 1
                        if distance[kk] < Rs :
                            condition[2] = 1
                        # 判断是否继续搜索
                        if ctime > 0 :
                            if condition[0]*condition[2]*condition[3] != 0 :
                                search=True
                                cposition[cnum]=kk
                                disvalid[cnum]=distance[kk]
                                cnum=cnum+1
                        else :
                            if condition[0]*condition[2] != 0 :
                                search=True
                                cposition[cnum]=kk
                                disvalid[cnum]=distance[kk]
                                cnum=cnum+1
                if ctime-n_stored >= 4:  #当气旋中心于同一位置持续停滞1天时， 停止搜索
                    search=False
                if search :
                    # Condition 2, finding the closest cyclone
                    try :
                        dismin=numpy.min(disvalid[0:cnum])
                        ks = numpy.where(disvalid[0:cnum] == dismin)[0][0]
                    except IndexError :
                        print(disvalid[0:cnum])
                        exit()
                    ctime=ctime+1
                    cp = int(cposition[ks])
                    cyclone_prest[ctime]=cyclone_pres[t][cp]
                    cyclone_lapt[ctime] = cyclone_lap[t][cp]
                    lont[ctime]=cyclone_lon[t][cp]
                    latt[ctime]=cyclone_lat[t][cp]
                    yeart[ctime]=year[t]
                    montht[ctime]=month[t]
                    dayt[ctime]=day[t]
                    hourt[ctime]=hour[t]
                    V0=Vnorm[cp]
                    newborn[t][cp] = False
                    tprev = ctime
                    prev_point = [latt[ctime-1],lont[ctime-1]]
                    if dismin !=0:
                        stored_point = [latt[ctime-1],lont[ctime-1]]
                        n_stored = ctime
                        # 若前后两点距离为0，判断为滞留
                elif not search :
                    if ctime-n_stored>=3:
                        ctime=n_stored
                    if ctime>=4 :
                        disfinal = spherical_distance((latt[0], lont[0]),(latt[ctime], lont[ctime]))
                        if latt[0]>=30 and disfinal>=500:# 出生于30°以北并移动大于500km
                            cyclonettnum = cyclonettnum + 1
                            wind_list = [get_wind(latt[icc], lont[icc], 5,
                                                  (int(yeart[icc]),
                                                  int(montht[icc]),
                                                  int(dayt[icc]),
                                                  int(hourt[icc])))
                                         for icc in range(ctime)]
                            # 如没有风场数据， 将windlist=[]即可
                            is_explo,deepening_rate = cyclone_explosive(
                                list(cyclone_prest[0:ctime+1]),
                                list(latt[0:ctime+1]), list(lont[0:ctime+1]),
                                6,wind_list = wind_list) # 判断是否属于爆发性气旋
# 测试， 查看路径是否异常
#=============================================================================
                            # basemap_option = {
                            #     'projection': 'npstere',
                            #     'boundinglat': 30,
                            #     'lon_0': 180,
                            #     # 'resolution':'l',
                            #     'round': True,
                            # }
                            # date0 = (int(yeart[0]),int(montht[0]),int(dayt[0]),int(hourt[0]))
                            # date1 = (int(yeart[ctime]),int(montht[ctime]),int(dayt[ctime]),int(hourt[ctime]))
                            # date_list = datelist(date0,date1,step='hours=6'
                            #                       ,remodel='date')
                            # fmty='%04d'
                            # fmtm='%02d'
                            # scatter=numpy.ones((ctime,2))-1
                            # for iii in range(ctime):
                            #     scatter[iii,1],scatter[iii,0]=lont[iii], latt[iii]
                            # for iii in range(ctime):
                            #     dd = date_list[iii]
                            #     ds = nc.Dataset('D:/data/era_int_mslp_6h_05/'
                            #                     +fmty%(dd.year)+fmtm%(dd.month)+'.nc')
                            #     dy = datelist((dd.year,dd.month,1,0),
                            #                   (dd.year,dd.month,dd.day,dd.hour)
                            #                   ,step='hours=6'
                            #                       ,remodel='date')
                            #     which = len(dy)-1
                            #     mslp=ds.variables['msl'][which,0:121,0:720]
                            #     lla=numpy.arange(90,29.5,-0.5)
                            #     llo=numpy.arange(0,360,0.5)
                            #     contour_data = {
                            #                 '0':{'data': mslp/100,
                            #                       'levels': numpy.arange(980,1100,2.5),
                            #                       'colors': 'b',
                            #                       'clabel': True
                            #                 }
                            #                 }
                            #     scatter_data = {'0':{
                            #         'data':scatter,
                            #         'colors':'k',
                            #         }
                            #         }
                            #     paint_figure(lla, llo, basemap_option, str(iii)+'.png', save=True,contour=True,data_contour=contour_data,scatter_latlon=True,data_scatter_latlon=scatter_data)
                            # os.system('pause')
#==============================================================================
                            cyc_dic = {
                                'cyclonettnum'  : cyclonettnum ,
                                'cyclone_prestt': list(cyclone_prest[0:ctime+1]),
                                'cycloen_laptt' : list(cyclone_lapt[0:ctime+1]),
                                'lontt'         : list(lont[0:ctime+1]) ,
                                'lattt'         : list(latt[0:ctime+1]) ,
                                'yeartt'        : list(yeart[0:ctime+1]),
                                'monthtt'       : list(montht[0:ctime+1]),
                                'daytt'         : list(dayt[0:ctime+1]),
                                'hourtt'        : list(hourt[0:ctime+1]) ,
                                'is_exp': is_explo,
                                'wind_list':wind_list,
                                'dpr': list(deepening_rate),
                                }
                            file_save = str(cyclonettnum)+'.json'
                            with open(path_save + '\\' + file_save,'w') as outfile :
                                json.dump(cyc_dic,outfile)
                    break  # 若判定气旋截止， 则其后的时间不再遍历
