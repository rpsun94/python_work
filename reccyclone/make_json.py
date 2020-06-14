import numpy
from demo_function import datelist ,mkdir
import json


date_list = datelist((1979, 1, 1, 0), (2019, 8, 31, 18),
                     step='hours=6', remodel='date')
N = len(date_list)
cyclone_num = []
cyclone_lap = []
cyclone_lat = []
cyclone_lon = []
cyclone_pres = []
year = []
month = []
day = []
hour = []
for i in range(N):
    temp_dic = numpy.load(r'prepare/temp/'+str(i)+'.npz')
    print('dealing No.'+str(i))
    cla = temp_dic['cyclone_lat']
    clo = temp_dic['cyclone_lon']
    clp = temp_dic['cyclone_lap']
    cpr = temp_dic['cyclone_pres']
    cnu = int(temp_dic['cyclone_num'])
    cyclone_num.append(cnu)
    cyclone_lat.append(list(cla[0:cnu]))
    cyclone_lon.append(list(clo[0:cnu]))
    cyclone_lap.append(list(clp[0:cnu]))
    cyclone_pres.append(list(cpr[0:cnu]))
    year.append(date_list[i].year)
    month.append(date_list[i].month)
    day.append(date_list[i].day)
    hour.append(date_list[i].hour)
save_dic = {
    'cyclone_num': cyclone_num,
    'cyclone_pres': cyclone_pres,
    'cyclone_lon': cyclone_lon,
    'cyclone_lat': cyclone_lat,
    'cyclone_lap': cyclone_lap,
    'hour': hour,
    'day': day,
    'month': month,
    'year': year,
    }
path_save = r'prepare\analyse'
mkdir(path_save)
file_save = 'cyclone_info2019.json'
with open(path_save + '\\' + file_save,'w') as outfile :
    json.dump(save_dic,outfile)
