import numpy
from demo_function import datelist ,mkdir
import json


N = 79811
cyclonettnum =0
cyclone_laptt = []
lattt = []
lontt = []
cyclone_prestt = []
yeartt = []
monthtt = []
daytt = []
hourtt = []
is_exp = []
wind_list = []
dpr = []
vl=['cyclone_laptt','cyclone_prestt','lattt','lontt','yeartt','monthtt',
    'daytt','hourtt','is_exp','wind_list','dpr']
vl2=['cycloen_laptt','cyclone_prestt','lattt','lontt','yeartt','monthtt',
    'daytt','hourtt','is_exp','wind_list','dpr']
count8=0
countexp=0
for i in range(1,N+1):
    print('dealing No.'+str(i))
    path = r'prepare/analyse/tracks'
    f_n  = str(i)+'.json'
    fid  = open(path+'/'+f_n)
    dic  = json.load(fid)
    fid.close()
    cyclonettnum = dic['cyclonettnum']
    for j in range(len(vl)):
        eval(vl[j]).append(dic[vl2[j]])
    if len(dic['yeartt'])>=8:
        count8+=1
    if dic['is_exp']:
        countexp+=1
print(count8,countexp)

save_dic = {}
save_dic['cyclonettnum']=cyclonettnum
for name in vl:
    save_dic[name]=eval(name)
path_save = r'prepare\analyse'
mkdir(path_save)
file_save = 'cyclone_track2019.json'
with open(path_save + '\\' + file_save,'w') as outfile :
    json.dump(save_dic,outfile)
