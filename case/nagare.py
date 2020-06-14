import os
os.system('cd C:/python_work/projects/case')
# script_name, filename, v7, val_name, ncase, d
for d in ['1']:
    os.system('python main_program.py D:/data/casedata_500.mat 0 casedata_500 235 '+d)
    os.system('python main_program.py D:/data/casedata_lvbo_lag-2.mat 1 lag__2lvbo 236 '+d)
    os.system('python main_program.py D:/data/casedata_lvbo_lag-3.mat 1 lag__3lvbo 236 '+d)