# @Time : 2023/11/16 14:04
# @Author : 赵新宇
# @File : Way.py
# @Use:
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.signal import convolve2d
@nb.jit(nopython=True)
def _barnes1(data,C,DX,NX,nx,ny,nz,data_new):
    for k in np.arange(nz):
        for j in np.arange(0 + NX, ny - NX):
            for i in np.arange(0 + NX, nx - NX):
                team_w1 = 0  # 分子
                team_w2 = 0  # 分母
                for j_n in np.arange(j - NX, j + NX):
                    for i_n in np.arange(i - NX, i + NX):
                        w = np.exp(-(((i - i_n) ** 2 + (j - j_n) ** 2) * DX ** 2) / (4 * C))  # 权重，给点了NX，那么所有的w是相同的
                        team_w2 = team_w2 + w
                        team_w1 = w * data[k, j_n, i_n] + team_w1
                data_new[k, j, i] = team_w1 / team_w2  # 对于每个格点的team_w2都是相同的。
                print(team_w2)
    return data_new

# barnes滤波 python实现barnes滤波
class Barnes_filter():
    def __init__(self,data,C,DX,NX):
        self.nz,self.ny,self.nx = np.array(data).shape
        self.data_new = np.ones(np.array(data).shape)
        self.C = np.array(C)
        self.DX = np.array(DX/1000)  # 单位Km
        self.NX = np.array(NX)
        self.data = np.array(data)

    def draw_R(self,lam):
        # 查看响应函数
        R0 = np.exp(-(4*np.pi**2*self.C)/lam**2)
        plt.plot(lam,R0)
        plt.savefig('R0.png')


    def barnes(self):
        result = _barnes1(data=self.data,C=self.C,DX=self.DX,NX=self.NX,nx=self.nx,ny=self.ny,nz=self.nz,
                         data_new=self.data_new)
        return result



    # def step_one(self):
    #     for k in np.arange(self.nz):
    #         for j in np.arange(0+self.NX,self.ny-self.NX):
    #             for i in np.arange(0+self.NX,self.nx-self.NX):
    #                 team_w1 = 0  # 分子
    #                 team_w2 = 0  # 分母
    #                 for j_n in np.arange(j-self.NX,j+self.NX):
    #                     for i_n in np.arange(i - self.NX, i + self.NX):
    #                         w = np.exp(-(((i-i_n)**2 + (j-j_n)**2)*self.DX**2)/(4*self.C))
    #                         team_w2 = team_w2 + w
    #                         team_w1 = w*self.data[k,j_n,i_n] + team_w1
    #                 self.data_new[k,j,i] = team_w2 / team_w1
    #     return self.data_new

# 利用卷积操作来实现barnes滤波
# 自定义卷积和，卷积一次就ok
# 得到的结果再除以总的权重之和就OK了

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


