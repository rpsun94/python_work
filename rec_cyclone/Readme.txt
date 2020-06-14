1.安装配置python环境，网上有很多教程，选一个喜欢的就好；
依赖的包主要有numpy,netCDF4,scipy,numba,ecmwf-api-client，提示缺什么的话安装就好了。
2.程序需要海平面气压数据和高程数据，高程数据从https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz 下载，下载好后解压，放到elev.py里对应的路径中即可
海平面气压用download_mslp.py就可以下载，为6h间隔0.5°分辨率。但需要查一下ecmwf-api-client的使用方法，放好账户密码。在其中可以更改年分。
3.下载好数据之后，依次运行cyclonedetect0.py,make_json.py,cyclonetracking.py,make_json_tracks.py就可以获得时间段内的气旋路径了，以json格式存放。如果需要其它格式，可以自行修改，或者告诉我你想要的格式和存放方式，我来写程序。
气旋追踪的算法主要参考halt2003年的论文，刁老师、傅刚老师都用过这种算法；我的程序只是对极地和0°向周围进行了填补，以方便能正常的计算85°以北的laplacian（P）和查找中心。
detect程序中的make_global_extend主要是为了方便寻找极地的中心，如果不需要寻找极地的中心，可以把该函数中的north和south的值改为False；
detect程序中的standard_lap规定了气旋中心laplacian（P）的最小值，表征气旋中心气压场的凹陷程度，因halt的原算法中并未提及这个量，可以参照第二篇文献修改这个值。现在设置为0.1主要是考虑到小于0.1的中心大多不具有闭合等压线的特征。
这次的程序把文件读取方式更改为从一个大文件中读取数据，并添加了一些中文注释，最后搜索路径时添加了搜索爆发性气旋一项。应该时可以正常运行，如果还缺少什么和我说。