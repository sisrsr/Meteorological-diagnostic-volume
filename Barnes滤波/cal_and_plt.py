# @Time : 2023/11/16 14:56
# @Author : 赵新宇
# @File : barnes_study.py
# @Use:
import time
import sys
import netCDF4 as nc
from wrf import getvar,interplevel
import numpy as np
sys.path.append(r'*******')  # class:Way的位置
from Use_model import Way
fn = nc.Dataset("wrfout数据")  # wrfout输出的数据
DX = fn.DX
u = getvar(fn,'ua')[20,200:400,200:400]
# Bar = Way.Barnes_filter(np.array(u),C=16,DX=DX,NX=5)
time_start = time.time()
Barnes = Way.Conv_banres(DN=10,DX=DX/1000,C=16)
weigth = Barnes.cal_weigth()
result = Barnes.CNN(data=u,weight=weigth)
time_end = time.time()
time_sum = time_end - time_start
np.array(result).astype(np.float32).tofile('try.bin')
print(result.shape)
print('成功运行')
print(time_sum)
