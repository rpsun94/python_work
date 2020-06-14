import xlrd
import json
path = r'F:\python_work\ttest'
filename = 'ttest.xlsx'
data = xlrd.open_workbook(path+'\\'+filename)
table = data.sheet_by_index(0)
#print(table.nrows,table.ncols)
#print(table.col_values(0)[2:-1])
#print(table.row_values(0)[2::])
dic={}
dic['num'] = table.col_values(0)[2:-1]
for i in range(2,table.nrows):
    dic[str(table.col_values(0)[i])]={}
    for j in range(2,table.ncols):
        dic[str(table.col_values(0)[i])][str(table.row_values(0)[j])]=table.row_values(i)[j]
#print(dic)
with open(path+'\\'+'ttest.json', 'w') as outfile:
    json.dump(dic, outfile)
