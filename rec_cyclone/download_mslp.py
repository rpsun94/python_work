from ecmwfapi import ECMWFDataServer
from demo_function import datelist
import numpy
import json


year_list = numpy.arange(1979,2020,1)
month_list = numpy.arange(1,13,1)
fmty = '%04d'
fmtm = '%02d'
fid = open('file_need.json')
cdic = json.load(fid)
for iy in year_list:
    for im in month_list:
        if not cdic[str(fmty%(iy))+str(fmtm%(im))]:
            std = 1
            iy_ = iy
            im_ = im+1
            if im_ == 13:
                iy_ = iy+1
                im_ = 1
            date_list = datelist((iy, im, 1), (iy_, im_, 1),
                                 step='day', remodel='date')
            edd = date_list[-2].day
            #print(im, edd)
            date = (str(fmty%(iy))+'-'+str(fmtm%(im))+'-'+str(fmtm%(std))
                    +'/to/'
                    +str(fmty%(iy))+'-'+str(fmtm%(im))+'-'+str(fmtm%(edd)))
            target = r"C:\data\era_int_mslp_6h_05"+'\\'+str(fmty%(iy))+str(fmtm%(im))+'.nc'
            server = ECMWFDataServer()
            server.retrieve({
                "class": "ei",
                "dataset": "interim",
                "date": date,
                "expver": "1",
                "grid": "0.5/0.5",
                "levtype": "sfc",
                "param": "151.128",
                "step": "0",
                "stream": "oper",
                "time": "00:00:00/06:00:00/12:00:00/18:00:00",
                "type": "an",
                "format": "netcdf",
                "target": target,
            })
            cdic[str(fmty%(iy))+str(fmtm%(im))] = True
            with open('file_need.json','w') as outfile :
                json.dump(cdic, outfile)
