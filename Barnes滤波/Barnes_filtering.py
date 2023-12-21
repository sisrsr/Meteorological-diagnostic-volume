# @Time : 2023/11/16 14:04
# @Author : 赵新宇
# @File : Way.py
# @Use:  使用Fortran或pybarnes或for循环会计算很慢，可能会出现运行内存不足
# 借用卷积的方法，使用权重矩阵，来对输入数据卷积，以达到Barnes滤波的效果
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.signal import convolve2d

class Conv_banres():
    def __init__(self,DN,DX,C):
        self.DN = DN
        self.DX = DX
        self.C = C
        pass

    def draw_R(self,lam):
        # 查看响应函数
        R0 = np.exp(-(4*np.pi**2*self.C)/lam**2)
        plt.plot(lam,R0)
        plt.savefig('R0.png')
        plt.clf()
        plt.close()

    def cal_weigth(self):
        # 计算权重
        self.w_all = np.ones((2*self.DN+1,2*self.DN+1))  # 以DN=5 为例，则w_all为10*10的
        for i in np.arange(2*self.DN+1):
            for j in np.arange(2*self.DN+1):
                self.w_all[i,j] = np.exp(-((i-self.DN)**2 +(j-self.DN)**2)*self.DX**2/(4*self.C))
        return self.w_all

    def CNN(self,data,weight):
        data_out = convolve2d(data,weight,mode='same') / self.w_all.sum()
        return data_out

