# @Time : 2023/10/27 9:34
# @Author : jj—bone
# @File : cal_var.py
# @Use: 计算诊断量
#基于wrf-python，对nc数据处理
import os
import wrf
from wrf import getvar,interplevel
from wrf.g_times import get_times
import numpy as np
import netCDF4 as nc
from wrf.g_times import get_times
import metpy.calc as mpcalc

# 各种位温
class Cal_theta_var():

    def __init__(self):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5

    # 计算扰动位温（通过RPT方法）
    def theta_r_RPT(self,theta,p_b,dz,):
        """

        :param p_b: 基态气压 数据要使用z坐标系下的
        :param dz: 垂直方向的DZ（使用静力方程）
        :return:theta_r,theta_b扰动位温、基态位温
        """
        dpb_dz = np.gradient(p_b,dz,axis=0)
        rho_b = -dpb_dz/self.__g
        Tb = p_b/(rho_b*self.__R)
        theta_b = Tb*(self.__ps/100/p_b)**(self.__R/self.__cp)
        theta_r = theta - theta_b
        return theta_r,theta_b


    # 广义位温
    def cal_theta_g(self,theta, p, rh, indk):
        '''
        计算广义位温
        :param q: 水汽混合比 单位 kg/kg
        :param theta: 位温 单位 K
        :param p: 气压 单位 hPa
        :param rh: 相对湿度 单位 %
        :return: 广义位温
        '''
        L = 2.5e+6
        T = wrf.tk(p * 100, theta).values
        ba = np.where((T - 273.16) > -15, 17.2693882, 21.3745584)
        bb = np.where((T - 273.16) > -15, 35.86, 7.66)

        es = 6.11 * np.exp(ba * (T - 273.16) / (T - bb))
        e = rh / 100 * es  # ok
        e = np.where(e <= 0, 1e-20, e)

        qvs = 0.622 * es / (p - 0.378 * es)  # 饱和水汽比湿
        qv = 0.622 * e / (p - 0.378 * e)
        qv = np.where(qv > qvs, qvs, qv)

        td = (273.16 * ba - bb * np.log(e / 6.11)) / (ba - np.log(e / 6.11))
        tl1 = 338.52 - 0.24 * T + 1.24 * (td - 273.16)  # 抬升凝结温度

        beta = np.exp(L * qvs / (self.__cp * tl1) * (qv / qvs) ** indk)

        theta_g = theta * beta

        return theta_g

    # 假相当位温
    def cal_theta_se(self, T, P, rv, Tc):
        """
        计算假相当位温
        :param T: 温度，单位：K
        :param P: 气压，单位：Pa
        :param rv: 水蒸气混合比：kg/kg
        :param Tc: 抬升凝结高度处的温度（使用metpy计算的）
        :return:
        """
        a_1 = 0.2854 * (1 - 0.28 * rv)
        theta_se_1 = T * (100000 / P) ** a_1
        theta_se_2 = np.exp(rv * (1 + 0.81 * rv) * (3376 / Tc - 2.54))
        theta_se = theta_se_1 * theta_se_2
        return np.array(theta_se)

    # eta坐标系下相当位温梯度
    def dtheta_e_deta(self,MUD,theta,theta_e,QVAPOR,p,eta):
        thm = theta * (1 + 1.61 * QVAPOR)  # 计算湿位温
        ad = self.__R * thm / self.__ps / (p * 10000 / self.__ps) \
             ** (self.__cv / self.__cp)
        result = -self.__g/(ad*MUD.values)*np.gradient(theta_e,eta,axis=0)
        return result


# 焦宝峰 垂直速度位涡
class Vertical_velocity_potential_vorticity():
    def __init__(self,z_level,DZ):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.DZ = DZ
        self.z_level = z_level


    def cal_var(self,wrfout_path):

        fn = nc.Dataset(wrfout_path)
        # 各个方向梯度的DIS
        self.DX = fn.DX
        self.DY = fn.DY
        # 提取计算变量
        time = get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        U_all = getvar(fn, 'ua').values
        V_all = getvar(fn, 'va').values
        W_all = getvar(fn, 'wa').values
        P_all = getvar(fn, 'pressure').values * 100  # 计算的时候，单位是pa
        # T_all = getvar(fn, 'tk').values
        # 需要通过位温来计算温度（直接getvar温度有问题，不知道什么原因）
        theta = getvar(fn, 'th')  # √
        T_all = np.squeeze(wrf.tk(P_all, theta).values)  # 加个values，否则做除法的时候，会多出来一维
        Z = getvar(fn, 'z坐标系下').values
        # 将各个变量插值到登高面上
        U = interplevel(U_all,Z,self.z_level) 
        V = np.array(interplevel(V_all,Z,self.z_level)) 
        W = np.array(interplevel(W_all,Z,self.z_level)) 
        P = np.array(interplevel(P_all,Z,self.z_level)) 
        T = np.array(interplevel(T_all,Z,self.z_level)) 
        # 维度
        a, b, c = U.shape
        # 计算密度rho
        rho = P / (self.__R * T)  # 用焦师兄代码中的公式
        # 计算水平散度
        du_dx,dv_dy = np.gradient(U, self.DX, axis=2),np.gradient(V, self.DY, axis=1)
        div = du_dx + dv_dy
        # 计算垂直涡度
        du_dy,dv_dx = np.gradient(U, self.DY, axis=1),np.gradient(V, self.DX, axis=2)
        vor = dv_dx - du_dy
        # 绝对涡度
        vora = vor + self.__f0
        # 计算q
        dw_dx = np.gradient(W, self.DX, axis=2)
        dv_dz = np.gradient(V, self.DZ, axis=0)
        dw_dy = np.gradient(W, self.DY, axis=1)
        du_dz = np.gradient(U, self.DZ, axis=0)
        dw_dz = np.gradient(W, self.DZ, axis=0)
        q = -dw_dx * dv_dz + dw_dy * du_dz + dw_dz * vora
        # 计算q1、q2、q3、q4
        dq_dx = np.gradient(q, self.DX, axis=2)
        dq_dy = np.gradient(q, self.DY, axis=1)
        dq_dz = np.gradient(q, self.DZ, axis=0)
        q1,q2, q3= -U * dq_dx,-V * dq_dy,-W * dq_dz
        q4 = (2 * (div * div - du_dx * dv_dy + dv_dx * du_dy + dw_dx * du_dz + dw_dy * dv_dz) - vor * self.__f0) * vora

        # 计算g1
        ddp_dx_dx,ddp_dy_dy = np.gradient(np.gradient(P, self.DX, axis=2), self.DX, axis=2),np.gradient(np.gradient(P, self.DY, axis=1), self.DY, axis=1)
        ddp_dx_dz,ddp_dy_dz = np.gradient(np.gradient(P, self.DX, axis=2), self.DZ, axis=0),np.gradient(np.gradient(P, self.DY, axis=1), self.DZ, axis=0)
        g1 = ((dv_dz * ddp_dx_dz - du_dz * ddp_dy_dz +
               vora * (ddp_dx_dx + ddp_dy_dy))) / rho
        # 计算g11,g12,g13
        g11, g12,g13 = (dv_dz * ddp_dx_dz) / rho,(-du_dz * ddp_dy_dz) / rho,(vora * (ddp_dx_dx + ddp_dy_dy)) / rho

        # 计算g2
        dp_dy,dp_dx = np.gradient(P, self.DY, axis=1),np.gradient(P, self.DX, axis=2)
        rpy,rpx = dp_dy / rho,dp_dx / rho
        drpy_dz,drpx_dz = np.gradient(rpy, self.DZ, axis=0),np.gradient(rpx, self.DZ, axis=0)
        g2 = drpy_dz * dw_dx - drpx_dz * dw_dy

        q = np.array(q).reshape((1, a, b, c))
        q1 = np.array(q1).reshape((1, a, b, c))
        q2 = np.array(q2).reshape((1, a, b, c))
        q3 = np.array(q3).reshape((1, a, b, c))
        q4 = np.array(q4).reshape((1, a, b, c))
        g1 = np.array(g1).reshape((1, a, b, c))
        g2 = np.array(g2).reshape((1, a, b, c))
        g11 = np.array(g11).reshape((1, a, b, c))
        g12 = np.array(g12).reshape((1, a, b, c))
        g13 = np.array(g13).reshape((1, a, b, c))
        data_all = np.concatenate((q, q1, q2, q3, q4, g1, g2, g11, g12, g13))
        if not os.path.exists(os.path.join('Vertical_velocity_potential_vorticity_data', )):
            try:
                os.makedirs(os.path.join('./', 'Vertical_velocity_potential_vorticity_data', ))
            except OSError:
                pass
        save_name = './'+'Vertical_velocity_potential_vorticity_data/' + str(time) + '.bin'
        data_all.tofile(save_name)

    # 湿位涡

# 周括，地形追随坐标系下的垂直运动方程
class The_terrain_follows_the_equation_of_vertical_motion():
    """
    由于内存问题，在输入数据的时候，尽量对数据进行一下切片，减小数据维度
    绘图不可能整个区域都绘制。
    """
    def __init__(self,wrfout_path_all,DX,DY,DZ,dt,save_path):
        self.__g = np.array(9.8).astype(np.float32)
        self.__R = np.array(287.05).astype(np.float32)
        self.__rv = np.array(461.6).astype(np.float32)
        self.__ps = np.array(100000.0).astype(np.float32)
        self.__cp = np.array(1005.0).astype(np.float32)
        self.__cv = np.array(718.0).astype(np.float32)
        self.__f0 = np.array(4.97e-5).astype(np.float32)
        self.wrfout_path_all = wrfout_path_all
        self.DX = np.array(DX).astype(np.float32)
        self.DY = np.array(DY).astype(np.float32)
        self.DZ =np.array( DZ).astype(np.float32)
        self.dt =np.array( dt).astype(np.float32)
        self.save_path = save_path  # 注意，应该是绝对路径

    def data_save(self,var,var_name,save_path):
        if not os.path.exists(os.path.join(save_path, )):
            try:
                os.makedirs(os.path.join(save_path, ))
            except OSError:
                pass
        save_name = str(save_path) + '/' + str(var_name) + '.bin'
        np.array(var).tofile(save_name)
        print(str(var) + '已del')
        del var

    def step_one(self):
        # 整合数据
        self.P, self.U, self.V, self.W, self.THETA, self.GEOPT, self.QVAPOR, self.QCLOUD, self.QRAIN, self.QICE, self.QSANOW, self.QGRAUP = [], [], [], [], [], [], [], [], [], [], [], []
        self.ETA, self.MU, self.MUB, self.WW, self.QV_DIABATIC, self.QC_DIABATIC, self.H_DIABATIC, self.RU_TEND, self.RV_TEND = [], [], [], [], [], [], [], [], []

        for i in np.arange(len(self.wrfout_path_all)):
            fn = nc.Dataset(self.wrfout_path_all[i])
            p = getvar(fn, 'pressure') * 100
            self.P.append(p)

            u = getvar(fn, 'ua')
            self.U.append(u)

            v = getvar(fn, 'va')
            self.V.append(v)

            w = getvar(fn, 'wa')
            self.W.append(w)

            theta = getvar(fn, 'theta')
            self.THETA.append(theta)

            geopt = getvar(fn, 'geopt')
            self.GEOPT.append(geopt)

            qvapor = getvar(fn, 'QVAPOR')  # 水汽混合比
            self.QVAPOR.append(qvapor)

            qclod = getvar(fn, 'QCLOUD')  # 云水混合比
            self.QCLOUD.append(qclod)

            qrain = getvar(fn, 'QRAIN')  # 雨水混合比
            self.QRAIN.append(qrain)

            qice = getvar(fn, 'QICE')  # 云冰混合比
            self.QICE.append(qice)

            qsnow = getvar(fn, 'QSNOW')  # 雪混合比
            self.QSANOW.append(qsnow)

            qgraup = getvar(fn, 'QGRAUP')  # 霰混合比
            self.QGRAUP.append(qgraup)

            eta = getvar(fn, 'ZNU')  # wrfout中的是ZNU和ZNW，二者有什么区别？？周师兄fortran代码中读取的是名称为ZNU0的变量  (100,)
            self.ETA.append(eta)

            mu = getvar(fn, 'MU')
            self.MU.append(mu)

            mub = getvar(fn, 'MUB')
            self.MUB.append(mub)

            ww = getvar(fn, 'WW')[:-1]  # 这个是101层，怎么解决？按照周师兄代码的意思，直接取100层即可
            self.WW.append(ww)

            qv_diabatic = getvar(fn, 'QV_DIABATIC')
            self.QV_DIABATIC.append(qv_diabatic)

            qc_diabatic = getvar(fn, 'QC_DIABATIC')
            self.QC_DIABATIC.append(qc_diabatic)

            h_diabatic = getvar(fn, 'H_DIABATIC')
            self.H_DIABATIC.append(h_diabatic)

            ru_tend = getvar(fn, 'RU_TEND')
            self.RU_TEND.append(ru_tend)

            rv_tend = getvar(fn, 'RV_TEND')
            self.RV_TEND.append(rv_tend)

        self.P, self.U, self.V, self.W, self.THETA, self.GEOPT, self.QVAPOR, self.QCLOUD, self.QRAIN, self.QICE, self.QSANOW, self.QGRAUP = \
            np.array(self.P).astype(np.float32), np.array(self.U).astype(np.float32), np.array(self.V).astype(np.float32), np.array(self.W).astype(np.float32), np.array(self.THETA).astype(np.float32),\
                np.array(self.GEOPT).astype(np.float32), np.array(self.QVAPOR).astype(np.float32), np.array(self.QCLOUD).astype(np.float32), np.array(self.QRAIN).astype(np.float32), np.array(self.QICE).astype(np.float32),\
                np.array(self.QSANOW).astype(np.float32), np.array(self.QGRAUP).astype(np.float32)

        self.ETA, self.MU, self.MUB, self.WW, self.QV_DIABATIC, self.QC_DIABATIC, self.H_DIABATIC, self.RU_TEND, self.RV_TEND = \
            np.array(self.ETA).astype(np.float32), np.array(self.MU).astype(np.float32), np.array(self.MUB).astype(np.float32), np.array(self.WW).astype(np.float32), np.array(self.QV_DIABATIC).astype(np.float32),\
                np.array(self.QC_DIABATIC).astype(np.float32), np.array(self.H_DIABATIC).astype(np.float32), np.array(self.RU_TEND).astype(np.float32), np.array(self.RV_TEND).astype(np.float32)

        print('step_one 完成')

    def step_two(self):
        self.qm = np.array(self.QVAPOR) + np.array(self.QCLOUD)+ np.array(self.QRAIN) + np.array(self.QICE) + np.array(self.QSANOW) + np.array(self.QGRAUP)
        self.dqm_dt = np.gradient(self.qm,self.dt,axis=0)
        self.dv_dx = np.gradient(np.array(self.V),self.DX,axis=3)
        self.dv_dy = np.gradient(np.array(self.V),self.DY,axis=2)
        self.dv_dz = np.gradient(np.array(self.V),self.DZ,axis=1)
        self.du_dx = np.gradient(np.array(self.U),self.DX,axis=3)
        self.du_dy = np.gradient(np.array(self.U),self.DY,axis=2)
        self.du_dz = np.gradient(np.array(self.U),self.DZ,axis=1)
        self.ksi = self.dv_dx - self.du_dy

        self.thm = np.array(self.THETA) * (1 + 1.61 * np.array(self.QVAPOR))
        self.ad = self.__R * self.thm / self.__ps / (np.array(self.P) / self.__ps) ** (self.__cv / self.__cp)
        self.a = self.ad / (1 +self.qm)

        self.dgeopt_dz = np.gradient(np.array(self.GEOPT), self.DZ, axis=1)
        self.dgeopt_dx = np.gradient(np.array(self.GEOPT), self.DX, axis=3)
        self.dgeopt_dy = np.gradient(np.array(self.GEOPT), self.DY, axis=2)
        self.dp_dz = np.gradient(np.array(self.P), self.DZ, axis=1)
        self.dp_dx = np.gradient(np.array(self.P), self.DX, axis=3)
        self.dp_dy = np.gradient(np.array(self.P), self.DY, axis=2)


        self.Px = self.a * (self.dp_dx - 1 / self.dgeopt_dz * self.dp_dz * self.dgeopt_dx)  # PGFx
        self.Py = self.a * (self.dp_dy - 1 / self.dgeopt_dz * self.dp_dz * self.dgeopt_dy)  # PGFy

        self.dPx_dx = np.gradient(self.Px, self.DX, axis=3)
        self.dPx_dy = np.gradient(self.Px, self.DY, axis=2)
        self.dPx_dz = np.gradient(self.Px, self.DZ, axis=1)
        
        self.dPy_dx = np.gradient(self.Py, self.DX, axis=3)
        self.dPy_dy = np.gradient(self.Py, self.DY, axis=2)
        self.dPy_dz = np.gradient(self.Py, self.DZ, axis=1)
        
        self.divQ_tmp2 = self.ksi * self.__f0 - self.dPx_dx - self.dPy_dy

        self.ddivQ_tmp2_dt = np.gradient(self.divQ_tmp2, self.dt, axis=0)
        print('step_two 完成')

    def step_three(self):
        # 由于内存不足，需要删除变量（前两步是可以顺利运行的）
        del self.QVAPOR, self.QCLOUD, self.QRAIN, self.QICE, self.QSANOW, self.QGRAUP

        mud = (self.MU.reshape((-1,1,self.MU.shape[1],self.MU.shape[2])) + self.MUB.reshape((-1,1,self.MUB.shape[1],self.MUB.shape[2])) )*np.ones(self.U.shape)
        omega = self.WW / mud
        del self.MU,self.MUB,self.WW

        # 计算divQ1
        Fthe = np.array(self.H_DIABATIC * mud).astype(np.float32)  # 与周师兄代码相同
        Fqv = np.array(self.QV_DIABATIC * mud).astype(np.float32)  # 与周师兄代码相同
        Fp = np.array((self.__cp / self.__cv) * self.P  / mud * (Fthe / self.THETA + self.THETA / self.thm * self.__rv / self.__R * Fqv)).astype(np.float32)  # 与周师兄代码相同
        dFp_dz = np.gradient(Fp, self.DZ, axis=1).astype(np.float32)
        dFp_dy = np.gradient(Fp, self.DY, axis=2).astype(np.float32)
        dFp_dx = np.gradient(Fp, self.DX, axis=3).astype(np.float32)
        print(dFp_dx.dtype)
        # 计算Fqm
        # Qm = mud * qm
        dqm_dx = np.gradient(self.qm, self.DX, axis=3).astype(np.float32)
        dqm_dy = np.gradient(self.qm, self.DY, axis=2).astype(np.float32)
        dqm_dz = np.gradient(self.qm, self.DZ, axis=1).astype(np.float32)
        Fqm =np.array((self.dqm_dt + self.U * dqm_dx + self.V * dqm_dy + omega * dqm_dz) * mud + Fqv + self.QC_DIABATIC * mud).astype(np.float32)  # 与周师兄代码相同
        print(Fqm.dtype)
        # 计算Fpx，由于公式较长，分为多个部分计算，最后求和
        Fpx_1 = np.array(1 / (self.ad * mud) * self.dPx_dz * (-omega * self.dgeopt_dz + self.W * self.__g)).astype(np.float32)  # 与周师兄代码相同
        Fpx_2 = np.array(self.U * self.dPx_dx).astype(np.float32)  # 与周师兄代码相同
        Fpx_3 = np.array(self.V * self.dPx_dy).astype(np.float32)  # 与周师兄代码相同
        Fpx_4 = np.array(self.a * (self.du_dx + 1 / (mud * self.ad) * self.du_dz * self.dgeopt_dx) * self.dp_dx).astype(np.float32)  # 与周师兄代码相同
        Fpx_5 = np.array(self.a * (self.dv_dx + 1 / (mud * self.ad) * self.dv_dz * self.dgeopt_dx) *self. dp_dy).astype(np.float32)  # 与周师兄代码相同
        Fpx_6 = np.array(self.a / (mud * self.ad) * (self.dv_dx * self.dgeopt_dy - self.dv_dy * self.dgeopt_dx) * self.dp_dz).astype(np.float32)  # 与周师兄代码相同
        Fpx_7 = np.array(-self.a * (dFp_dx + 1 / (self.ad * mud) * dFp_dz * self.dgeopt_dx)).astype(np.float32).astype(np.float32)  # 与周师兄代码相同
        Fpx_8 = np.array(self.a ** 2 * Fqm / (self.ad * mud) * (self.dp_dx + 1 / (self.ad * mud) * self.dp_dz * self.dgeopt_dx)).astype(np.float32)  # 与周师兄代码相同
        Fpx = np.array(Fpx_1 + Fpx_2 + Fpx_3 + Fpx_4 + Fpx_5 + Fpx_6 + Fpx_7 + Fpx_8).astype(np.float32)
        print(Fpx.dtype)
        qx = np.gradient(Fpx, self.DZ, axis=1).astype(np.float32)
        divQ1 = np.gradient(qx, self.DX, axis=3).astype(np.float32)
        del qx

        # 计算divQ2
        dPy_dy = np.gradient(self.Py, self.DY, axis=2).astype(np.float32)
        dPy_dx = np.gradient(self.Py, self.DX, axis=3).astype(np.float32)
        dPy_dz = np.gradient(self.Py, self.DZ, axis=1).astype(np.float32)


        # 计算Fpy
        Fpy_1 = np.array(1 / (mud * self.ad) * dPy_dz * (-omega * self.dgeopt_dz + self.W * self.__g)).astype(np.float32)  # 与周师兄代码相同
        Fpy_2 = np.array(self.V * dPy_dy).astype(np.float32)  # 与周师兄代码相同
        Fpy_3 = np.array(self.U * dPy_dx).astype(np.float32)  # 与周师兄代码相同
        Fpy_4 = np.array(self.a * (self.du_dy + 1 / (mud * self.ad) * self.du_dz * self.dgeopt_dy) * self.dp_dx).astype(np.float32)  # 与周师兄代码相同
        Fpy_5 = np.array(self.a * (self.dv_dy + 1 / (mud * self.ad) * self.dv_dz * self.dgeopt_dy) * self.dp_dy).astype(np.float32)  # 与周师兄代码相同
        Fpy_6 = np.array(self.a / (mud * self.ad) * (self.du_dy * self.dgeopt_dx - self.du_dx * self.dgeopt_dy) * self.dp_dz).astype(np.float32)  # 与周师兄代码相同
        Fpy_7 = np.array(-self.a * (dFp_dy + 1 / (mud + self.ad) * dFp_dz * self.dgeopt_dy) ).astype(np.float32) # 与周师兄代码相同
        Fpy_8 =np.array( self.a ** 2 * Fqm / (self.ad * mud) * (self.dp_dy + 1 / (self.ad * mud) * self.dp_dz * self.dgeopt_dy)).astype(np.float32)  # 与周师兄代码相同
        Fpy = np.array(Fpy_1 + Fpy_2 + Fpy_3 + Fpy_4 + Fpy_5 + Fpy_6 + Fpy_7 + Fpy_8).astype(np.float32)
        qy = np.gradient(Fpy, self.DZ, axis=1, ).astype(np.float32)
        divQ2 = np.gradient(qy, self.DY, axis=2, ).astype(np.float32)
        del qy

        # 计算divQ3
        dksi_dx = np.gradient(self.ksi, self.DX, axis=3).astype(np.float32)
        dksi_dy = np.gradient(self.ksi, self.DY, axis=2).astype(np.float32)
        lamuta = np.array(self.du_dx + self.dv_dy).astype(np.float32)  # λ # 与周师兄代码相同

        # 计算dmud_dt
        domega_dz = np.gradient(omega, self.DZ, axis=1, ).astype(np.float32)
        dmud_dx = np.gradient(mud, self.DX, axis=3, ).astype(np.float32)
        dmud_dy = np.gradient(mud, self.DY, axis=2).astype(np.float32)

        dmud_dt = np.array(mud * (-domega_dz - self.du_dx - self.dv_dy) - self.U * dmud_dx - self.V * dmud_dy).astype(np.float32)  # 与周师兄代码相同

        du_dt = np.array((self.RU_TEND[:,:,:,:-1] - self.U * dmud_dt) / mud).astype(np.float32)  # 与周师兄代码相同
        dv_dt = np.array((self.RV_TEND[:,:,:-1,:] - self.V * dmud_dt) / mud ).astype(np.float32) # 与周师兄代码相同
        Fu = np.array((du_dt + self.U * self.du_dx + self.V * self.du_dy + omega * self.du_dz - self.__f0 * self.V + self.a * self.dp_dx + self.a / (
                    mud * self.ad) * self.dp_dz * self.dgeopt_dx) * mud ).astype(np.float32) # 与周师兄代码相同
        Fv = np.array((dv_dt + self.U * self.dv_dx + self.V * self.dv_dy + omega * self.dv_dz + self.__f0 * self.U + self.a * self.dp_dy + self.a / (
                    mud * self.ad) * self.dp_dz * self.dgeopt_dy) * mud).astype(np.float32)  # 与周师兄代码相同
        print(Fv.dtype)

        Fksi_tmp1 =np.array( 1 / (mud * self.ad) * self.dv_dz * (-omega * self.dgeopt_dz + self.W * self.__g) + self.Py).astype(np.float32)
        dFksi_tmp1_dx = np.gradient(Fksi_tmp1, self.DX, axis=3).astype(np.float32)
        Fksi_tmp2 = np.array(1 / (mud * self.ad) * self.du_dz * (-omega * self.dgeopt_dz + self.W * self.__g) + self.Px).astype(np.float32)
        dFksi_tmp2_dy = np.gradient(Fksi_tmp2, self.DY, axis=2).astype(np.float32)
        Fksi_tmp3 = np.array(Fv / mud).astype(np.float32)
        dFksi_tmp3_dx = np.gradient(Fksi_tmp3, self.DX, axis=3).astype(np.float32)
        Fksi_tmp4 = np.array(Fu / mud).astype(np.float32)
        dFksi_tmp4_dy = np.gradient(Fksi_tmp4, self.DY, axis=2).astype(np.float32)
        del Fu,Fv,Fksi_tmp1,Fksi_tmp2,Fksi_tmp3,Fksi_tmp4

        Fksi = np.array(-self.U * dksi_dx - self.V * dksi_dy - self.ksi * lamuta - dFksi_tmp1_dx + dFksi_tmp2_dy + dFksi_tmp3_dx - dFksi_tmp4_dy).astype(np.float32)
        dFksi_dz = np.gradient(Fksi, self.DZ, axis=1).astype(np.float32)
        # q_eta = f0 * Fksi + f0 ** 2 / (mud * ad) * (du_dz * dgeopt_dx + dv_dz * dgeopt_dy) - ddivQ3_var_dt
        # 周师兄的代码力，把divQ3分为了divQ3、divQ4、divQ5
        divQ3 = np.array(self.__f0 * dFksi_dz).astype(np.float32)
        del dFksi_dz,Fksi

        # 计算divQ4
        divQ_tmp1 = np.array(1 / (mud * self.ad) * (self.du_dz * self.dgeopt_dx + self.dv_dz * self.dgeopt_dy)).astype(np.float32)
        ddivQ_tmp1_dz = np.gradient(divQ_tmp1, self.DZ, axis=1).astype(np.float32)
        divQ4 = np.array(self.__f0 ** 2 * ddivQ_tmp1_dz).astype(np.float32)
        del divQ_tmp1,ddivQ_tmp1_dz

        # 计算divQ5
        dddivQ_tmp2_dt_dz = np.gradient(self.ddivQ_tmp2_dt, self.DZ, axis=1).astype(np.float32)
        divQ5 = np.array(-dddivQ_tmp2_dt_dz).astype(np.float32)
        del dddivQ_tmp2_dt_dz,self.ddivQ_tmp2_dt

        divQ = np.array(divQ1 + divQ2 + divQ3 + divQ4 + divQ5).astype(np.float32)

        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ,'divQ',save_path=self.save_path)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1,'divQ1',save_path=self.save_path)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2,'divQ2',save_path=self.save_path)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ3,'divQ3',save_path=self.save_path)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ4,'divQ4',save_path=self.save_path)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ5,'divQ5',save_path=self.save_path)

        # divQ1_1
        divQ1_1 = np.gradient(np.gradient(Fpx_1, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_1,'divQ1_1',save_path=self.save_path)

        # divQ1_2
        divQ1_2 = np.gradient(np.gradient(Fpx_2, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_2,'divQ1_2',save_path=self.save_path)

        # divQ1_3
        divQ1_3 = np.gradient(np.gradient(Fpx_3, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_3,'divQ1_3',save_path=self.save_path)

        # divQ1_4
        divQ1_4 = np.gradient(np.gradient(Fpx_4, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_4,'divQ1_4',save_path=self.save_path)

        # divQ1_5
        divQ1_5 = np.gradient(np.gradient(Fpx_5, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_5,'divQ1_5',save_path=self.save_path)

        # divQ1_6
        divQ1_6 = np.gradient(np.gradient(Fpx_6, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_6,'divQ1_6',save_path=self.save_path)

        # divQ1_7
        divQ1_7 = np.gradient(np.gradient(Fpx_7, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_7,'divQ1_7',save_path=self.save_path)

        # divQ1_8
        divQ1_8 = np.gradient(np.gradient(Fpx_8, self.DX, axis=3), self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ1_8,'divQ1_8',save_path=self.save_path)

        # divQ2_1
        d21 = np.array(1 / (mud * self.ad) * dPy_dz * (-omega * self.dgeopt_dz + self.W * self.__g)).astype(np.float32)
        dd21_dy = np.gradient(d21, self.DY, axis=2).astype(np.float32)
        divQ2_1 = np.gradient(dd21_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_1,'divQ2_1',save_path=self.save_path)
        del d21,dd21_dy

        # divQ2_2
        d22 = np.array(self.V * dPy_dy).astype(np.float32)
        dd22_dy = np.gradient(d22, self.DY, axis=2).astype(np.float32)
        divQ2_2 = np.gradient(dd22_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_2,'divQ2_2',save_path=self.save_path)
        del d22,dd22_dy

        # divQ2_3
        d23 = np.array(self.U * dPy_dx).astype(np.float32)
        dd23_dy = np.gradient(d23, self.DY, axis=2).astype(np.float32)
        divQ2_3 = np.gradient(dd23_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_3,'divQ2_3',save_path=self.save_path)
        del d23,dd23_dy

        # divQ2_4
        d24 = np.array(Fpy_4).astype(np.float32)
        dd24_dy = np.gradient(d24, self.DY, axis=2).astype(np.float32)
        divQ2_4 = np.gradient(dd24_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_4,'divQ2_1',save_path=self.save_path)
        del d24,dd24_dy

        # divQ2_5
        d25 = np.array(Fpy_5).astype(np.float32)
        dd25_dy = np.gradient(d25, self.DY, axis=2, ).astype(np.float32)
        divQ2_5 = np.gradient(dd25_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_5,'divQ2_5',save_path=self.save_path)
        del d25,dd25_dy

        # divQ2_6
        d26 = np.array(Fpy_6).astype(np.float32)
        dd26_dy = np.gradient(d26, self.DY, axis=2).astype(np.float32)
        divQ2_6 = np.gradient(dd26_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_6,'divQ2_6',save_path=self.save_path)
        del d26,dd26_dy

        # divQ2_7
        d27 = np.array(Fpy_7).astype(np.float32)
        dd27_dy = np.gradient(d27, self.DY, axis=2)
        divQ2_7 = np.array(np.gradient(dd27_dy, self.DZ, axis=1))
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_7,'divQ2_7',save_path=self.save_path)
        del d27,dd27_dy

        # divQ2_8
        d28 = np.array(Fpy_8).astype(np.float32)
        dd28_dy = np.gradient(d28, self.DY, axis=2).astype(np.float32)
        divQ2_8 = np.gradient(dd28_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_8,'divQ2_8',save_path=self.save_path)
        del d28,dd28_dy

        # divQ2_5_1
        d251 = np.array(self.a *self. dv_dy * self.dp_dy).astype(np.float32)
        dd251_dy = np.gradient(d251, self.DY, axis=2).astype(np.float32)
        divQ2_5_1 = np.gradient(dd251_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_5_1,'divQ2_5_1',save_path=self.save_path)
        del d251,divQ2_5_1

        # divQ2_5_2
        d252 = np.array(1 / (mud * self.ad) * self.dv_dz * self.dgeopt_dy * self.a * self.dp_dy).astype(np.float32)
        dd252_dy = np.gradient(d252, self.DY, axis=2).astype(np.float32)
        divQ2_5_2 = np.gradient(dd252_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_5_2,'divQ2_5_2',save_path=self.save_path)
        del d252,divQ2_5_2

        # divQ2_6_1
        d261 = np.array(self.du_dy * self.dgeopt_dx * self.a / (mud * self.ad) * self.dp_dz).astype(np.float32)
        dd261_dy = np.gradient(d261, self.DY, axis=2).astype(np.float32)
        divQ2_6_1 = np.gradient(dd261_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_6_1,'divQ2_6_1',save_path=self.save_path)
        del d261,divQ2_6_1

        # divQ2_6_2
        d262 = np.array(-self.du_dx * self.a / (mud * self.ad) * self.dgeopt_dy * self.dp_dz).astype(np.float32)
        dd262_dy = np.gradient(d262, self.DY, axis=2, ).astype(np.float32)
        divQ2_6_2 = np.gradient(dd262_dy, self.DZ, axis=1, ).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_6_2,'divQ2_6_2',save_path=self.save_path)
        del d262,divQ2_6_2

        # divQ2_7_1
        d271 = np.array(-self.a * dFp_dy).astype(np.float32)
        dd271_dy = np.gradient(d271, self.DY, axis=2, ).astype(np.float32)
        divQ2_7_1 = np.gradient(dd271_dy, self.DZ, axis=1, ).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_7_1,'divQ2_7_1',save_path=self.save_path)
        del d271,divQ2_7_1

        # divQ2_7_2
        d272 = np.array(-self.a / (mud * self.ad) * dFp_dz * self.dgeopt_dy).astype(np.float32)
        d272_dy = np.gradient(d272, self.DY, axis=2).astype(np.float32)
        divQ2_7_2 = np.gradient(d272_dy, self.DZ, axis=1).astype(np.float32)
        The_terrain_follows_the_equation_of_vertical_motion.data_save(self,divQ2_7_2,'divQ2_7_2',save_path=self.save_path)
        del d272,divQ2_7_2



        print('数据生成成功')

    def end(self):
        The_terrain_follows_the_equation_of_vertical_motion.step_one(self,)
        The_terrain_follows_the_equation_of_vertical_motion.step_two(self,)
        The_terrain_follows_the_equation_of_vertical_motion.step_three(self,)

# # 周括，三维散度气压倾向方程、垂直气压梯度力方程
# class Three_div_zk():
#     def __init__(self,z_level=np.arange(50,10000+250,250)):
#         self.z_level = z_level
#
#         pass
#
#     def step_one(self):
#         # 处理含有时间维度的数据
#         pass
#     def 气压倾向方程:
#         pass
#
#     def 垂直气压梯度力方程:
#         pass

# 层结稳定度倾向方程
class Strata_stability_tendency_equation():
    def __init__(self,wrf_path,save_path):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.DZ = 50 * 100  # 单位pa
        self.p_level = np.arange(1000,200-50,-50)
        self.wrf_path = wrf_path
        self.save_path = save_path
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass

    def step_one(self):
        # 读取数据
        self.fn = nc.Dataset(self.wrf_path)
        self.u_all = getvar(self.fn,'ua')
        self.v_all = getvar(self.fn,'va')
        self.w_all = getvar(self.fn,'wa')
        self.theta_all = getvar(self.fn,'theta')
        self.qv_all = getvar(self.fn,'QVAPOR')
        self.p_all = getvar(self.fn,'pressure')
        self.T_all = wrf.tk(self.p_all * 100, self.theta_all)
        self.theta_e_all = wrf.eth(self.qv_all,self.T_all,self.p_all*100)
        self.DX = self.fn.DX
        self.DY = self.fn.DY
        self.time_index = get_times(self.fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]

        # 插值
        self.theta_e = interplevel(self.theta_e_all,self.p_all,self.p_level)
        self.u = interplevel(self.u_all,self.p_all,self.p_level)
        self.v = interplevel(self.v_all,self.p_all,self.p_level)
        self.w = interplevel(self.w_all,self.p_all,self.p_level)

    def step_two(self):
        # 计算梯度
        self.dtheta_dx = np.gradient(self.theta_e,self.DX,axis=2)
        self.ddtheta_dxdp = np.gradient(self.dtheta_dx,self.DZ,axis=0)

        self.dtheta_dy = np.gradient(self.theta_e,self.DY,axis=1)
        self.ddtheta_dydp = np.gradient(self.dtheta_dy,self.DZ,axis=0)

        self.du_dx = np.gradient(self.u,self.DX,axis=2)
        self.dv_dy = np.gradient(self.v,self.DY,axis=1)
        self.du_dp = np.gradient(self.u,self.DZ,axis=0)
        self.dv_dp = np.gradient(self.v,self.DZ,axis=0)

        self.dtheta_dp = np.gradient(self.theta_e,self.DZ,axis=0)
        self.ddtheta_dpdp = np.gradient(self.dtheta_dp,self.DZ,axis=0)

    def step_three(self):

        # 计算
        part_one = np.array(-self.u*self.ddtheta_dxdp)
        part_two = np.array(-self.v*self.ddtheta_dydp)
        part_three =  np.array(-self.w*self.ddtheta_dpdp)
        M_one =  np.array(-self.du_dp*self.dtheta_dx)
        M_two =  np.array(-self.dv_dp*self.dtheta_dy)
        M_three =  np.array((self.du_dx + self.dv_dy)*self.dtheta_dp)

        # 保存
        data_all = np.concatenate((part_one,part_two,part_three,M_one,M_two,M_three))

        save_name = os.path.join(self.save_path, self.time_index + '.bin')
        np.array(data_all).tofile(save_name)

    def end(self):
        Strata_stability_tendency_equation.step_one(self)
        Strata_stability_tendency_equation.step_two(self)
        Strata_stability_tendency_equation.step_three(self)
        print('成功计算一次')

# 湿位涡
class MPV():
    def __init__(self,wrf_path,theta,save_path):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.DZ = np.array([975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])* 100  # 单位pa
        self.p_level= np.array([975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])  # 单位hPa
        self.wrf_path = wrf_path
        self.save_path = save_path
        self.theta_eta = theta  # 用来计算湿位涡的位温（根据输入，转换成广义位涡）

    def step_one(self):
        # 提取数据
        self.fn = nc.Dataset(self.wrf_path)
        self.u_all = getvar(self.fn,'ua')  # 单位m/s
        self.v_all = getvar(self.fn,'va')
        self.qv_all = getvar(self.fn,'QVAPOR')
        self.p_all = getvar(self.fn,'pressure')
        self.DX = self.fn.DX  # 单位m
        self.DY = self.fn.DY
        self.time_index = get_times(self.fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]

        # 插值
        self.u = interplevel(self.u_all,self.p_all,self.p_level)  # m/s
        self.v = interplevel(self.v_all,self.p_all,self.p_level)  # m/s
        self.theta_e = interplevel(self.theta_eta,self.p_all,self.p_level)  # K 等压面

    def step_two(self):
        # 计算梯度
        self.dtheta_dx = np.gradient(self.theta_e,self.DX,axis=2)
        # self.ddtheta_dxdp = np.gradient(self.dtheta_dx,self.DZ,axis=0)

        self.dtheta_dy = np.gradient(self.theta_e,self.DY,axis=1)
        #self.ddtheta_dydp = np.gradient(self.dtheta_dy,self.DZ,axis=0)

        self.du_dx = np.gradient(self.u,self.DX,axis=2)
        self.dv_dx = np.gradient(self.v,self.DX,axis=2)

        self.du_dy = np.gradient(self.u,self.DY,axis=1)
        self.dv_dy = np.gradient(self.v,self.DY,axis=1)
        self.du_dp = np.gradient(self.u,self.DZ,axis=0)
        self.dv_dp = np.gradient(self.v,self.DZ,axis=0)

        self.dtheta_dp = np.gradient(self.theta_e,self.DZ,axis=0)
        # self.ddtheta_dpdp = np.gradient(self.dtheta_dp,self.DZ,axis=0)

    def step_three(self):
        vor = self.dv_dx - self.du_dy  # m/s/m
        vora = vor + self.__f0
        self.MPV1 = np.array(-self.__g * vora * self.dtheta_dp)  # /s k/pa
        self.MPV2 = np.array(self.__g * self.dv_dp*self.dtheta_dx - self.__g*self.du_dp*self.dtheta_dy)

    def end(self):
        MPV.step_one(self)
        MPV.step_two(self)
        MPV.step_three(self)
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
        self.data = np.concatenate((self.MPV1,self.MPV2))
        save_name = os.path.join(self.save_path, self.time_index + '.bin')
        np.array(self.data).tofile(save_name)
        print('计算成功一次')

# 湿位涡2 《华东地区强对流降水过程湿斜压涡度的诊断分析》z坐标系下
class MPV_two():
    def __init__(self,wrf_path,theta,save_path,indk):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.z_level = np.insert(np.arange(1000, 10000 + 500, 500), 0, [250, 500, 750])
        self.wrf_path = wrf_path
        self.save_path = save_path
        self.indk = indk
    def cal_beta(self):
        # 提取数据
        self.fn = nc.Dataset(self.wrf_path)
        self.p = getvar(self.fn,'pressure')
        self.theta = getvar(self.fn,'theta')
        self.qv = getvar(self.fn,'QVAPOR')
        T = wrf.tk(self.p * 100,self.theta)
        rh = wrf.rh(self.qv, self.p * 100, T)
        # 计算湿系数
        L = 2.5e+6
        ba = np.where((T - 273.16) > -15, 17.2693882, 21.3745584)
        bb = np.where((T - 273.16) > -15, 35.86, 7.66)

        es = 6.11 * np.exp(ba * (T - 273.16) / (T - bb))
        e = rh / 100 * es  # ok
        e = np.where(e <= 0, 1e-20, e)

        qvs = 0.622 * es / (self.p - 0.378 * es)  # 饱和水汽比湿
        qv = 0.622 * e / (self.p - 0.378 * e)
        qv = np.where(qv > qvs, qvs, qv)

        td = (273.16 * ba - bb * np.log(e / 6.11)) / (ba - np.log(e / 6.11))
        tl1 = 338.52 - 0.24 * T + 1.24 * (td - 273.16)  # 抬升凝结温度

        self.beta = np.exp(L * qvs / (self.__cp * tl1) * (qv / qvs) ** self.indk)

    def step_one(self):


        pass


# 二阶湿位涡 《南疆西部干旱区两次极端暴雨过程对比分析》
class Second_MPV():
    def __init__(self,wrfpath,theta_g,save_path):
        self.save_path = save_path
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.theta_g = theta_g  # 广义位温  以下是对应的层数
        self.DZ = np.array([1000,975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])* 100  # 单位pa
        self.p_level= np.array([1000,975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])
        self.wrfpath = wrfpath
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
    def cal_one(self,):
        fn = nc.Dataset(self.wrfpath)
        u_all = getvar(fn,'ua')
        v_all = getvar(fn,'va')
        P = getvar(fn,'pressure')
        u = interplevel(u_all,P,self.p_level)
        v = interplevel(v_all,P,self.p_level)
        DX,DY = fn.DX,fn.DY
        avo = np.gradient(v,DX,axis=2) - np.gradient(u,DY,axis=1)
        self.time_index = get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        dtheta_dz = np.gradient(self.theta_g,self.DZ,axis=0)
        ddtheta_ddz = np.gradient(dtheta_dz,self.p_level,axis=0)
        self.S1 = self.__g**2 *(avo+self.__f0)*ddtheta_ddz
        self.S2 = 1/2*self.__g**2 * dtheta_dz*np.gradient((avo+self.__g)**2,self.p_level,axis=0)

    def end(self):
        self.data = np.concatenate((self.S1, self.S2))
        save_name = os.path.join(self.save_path, self.time_index + '.bin')
        np.array(self.data).astype(np.float32).tofile(save_name)
        print('计算成功一次')



# 位势散度  《一次飑线过程对流稳定度演变的诊断分析》
class Potential_divergence():
    def __init__(self, wrf_path, save_path):
        self.z_level = np.arange(250,10000+250,250)
        self.DZ = 250
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.wrf_path = wrf_path
        self.save_path = save_path
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
    def step_one(self):
        # 将数据插值到登高面
        fn = nc.Dataset(self.wrf_path)
        DX,DY = fn.DX,fn.DY
        u_all, v_all = getvar(fn, 'ua'), getvar(fn, 'va')
        p = getvar(fn,'pressure')*100
        theta = getvar(fn,'theta')
        T = wrf.tk(p, theta)
        QVAPOR = getvar(fn,'QVAPOR')
        theta_e_all = wrf.eth(QVAPOR,T,p)
        z = getvar(fn, 'z')
        u,v = interplevel(u_all,z,self.z_level),interplevel(v_all,z,self.z_level)
        theta_e = interplevel(theta_e_all,z,self.z_level)

        # 计算梯度
        du_dz,dv_dz = np.gradient(u,self.DZ,axis=0),np.gradient(v,self.DZ,axis=0)
        du_dx,dv_dy = np.gradient(u,DX,axis=2),np.gradient(v,DY,axis=1)
        dtheta_dx,dtheta_dy,dtheta_dz = np.gradient(theta_e,DX,axis=0),np.gradient(theta_e,DY,axis=1),np.gradient(theta_e,self.DZ,axis=0)

        # 计算
        part_i = -du_dz*dtheta_dx
        part_ii = -dv_dz*dtheta_dy
        part_iii = (du_dx + dv_dy)*dtheta_dz

        self.data_all = np.concatenate((part_i,part_ii,part_iii))
        self.time_index = str(get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0])

    def step_two(self):
        save_name = os.path.join(self.save_path, self.time_index + '.bin')
        np.array(self.data_all).astype(np.float32).tofile(save_name)
        print('计算成功一次')

    def end(self):
        Potential_divergence.step_one(self)
        Potential_divergence.step_two(self)


# 周括 三维散度的气压倾向方程和垂直气压梯度力方程
class zk_p():
    def __init__(self,wrf_path,save_path):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.DZ = np.array([1000,975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])* 100  # 单位pa
        self.p_level= np.array([1000,975,950,925,900,850,800,750,700,650,600,
                       550,500,450,400,350,300,250,200])
        self.wrf_path = wrf_path
        self.save_path = save_path
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass

    def step_one(self):
        # 提取数据
        pass

# 申冬冬 锋生函数
class F_f():
    # 热力和动力锋生
    class Dongli_f():
        pass
    class Reli_f():
        pass
    pass

# 马淑萍 《复杂地形强降雪过程中垂直运动诊断分析》WRF地形追随坐标系湿大气垂直运动方程
class Wrf_wf():
    def __init__(self,wrf_path,save_path):
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.fn = nc.Dataset(wrf_path)
        self.save_path = save_path
        self.time = get_times(self.fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
    def data_save(self,var,save_path):
        if not os.path.exists(os.path.join(save_path, )):
            try:
                os.makedirs(os.path.join(save_path, ))
            except OSError:
                pass
        save_name = str(save_path) + '/' + str(self.time) + '.bin'
        np.array(var).tofile(save_name)


    def step_one(self):
        # 准备变量，基本变量
        self.eta = getvar(self.fn,'ZNU')
        self.DX = self.fn.DX
        self.DY = self.fn.DY

        P = getvar(self.fn,'pressure')
        theta = getvar(self.fn,'theta')

        self.mu = np.array(getvar(self.fn,'MU')).reshape((1,theta.shape[1],theta.shape[2]))  # 扰动
        self.mub = np.array(getvar(self.fn,'MUB')).reshape((1,theta.shape[1],theta.shape[2]))  # 基
        self.mud = self.mub + self.mu
        QVAPOR = getvar(self.fn,'QVAPOR')
        QCLOD = getvar(self.fn, 'QCLOUD')
        QRAIN = getvar(self.fn, 'QRAIN')
        QICE = getvar(self.fn, 'QICE')
        QSNOW = getvar(self.fn, 'QSNOW')
        QGRAUP = getvar(self.fn, 'QGRAUP')
        thm = np.array(theta) * (1 + 1.61 * np.array(QVAPOR))
        self.qm = QVAPOR + QCLOD + QRAIN + QICE + QSNOW + QGRAUP
        self.ad = self.__R * thm / self.__ps / (np.array(P) / self.__ps) ** (self.__cv / self.__cp)
        self.a = self.ad / (1 +self.qm)

    def step_two(self):
        # 计算
        # 提取需要计算的变量
        self.ww = getvar(self.fn,'WW')[:-1]
        omega = self.ww /self.mud  # 那个omeag

        u = getvar(self.fn, 'ua')
        v = getvar(self.fn, 'va')
        w = getvar(self.fn,'wa')
        pe = getvar(self.fn,'P')  # 扰动气压

        dw_dx = np.gradient(w,self.DX,axis=2)
        dw_dy = np.gradient(w, self.DY, axis=1)
        dw_deta = np.gradient(u,self.eta,axis=0)
        dpe_deta = np.gradient(pe,self.eta,axis=0)

        self.right1 = -u*dw_dx
        self.right2 = -v*dw_dy
        self.right3 = -omega*dw_deta
        self.right4 = self.__g*self.a*dpe_deta/(self.mud*self.ad)
        self.right5 = -self.__g*self.qm*(self.mub*self.a)/(self.mud*self.ad)
        self.right6 = -self.__g*self.mu/self.mud

    def step_three(self):
        # 保存数据
        save_path1 = str(self.save_path) + "/纬向平流"
        Wrf_wf.data_save(self,self.right1, save_path=save_path1)

        save_path2 = str(self.save_path) + "/经向平流"
        Wrf_wf.data_save(self,self.right2, save_path=save_path2)

        save_path3 = str(self.save_path) + "/垂直平流"
        Wrf_wf.data_save(self,self.right3, save_path=save_path3)

        save_path4 = str(self.save_path) + "/垂直扰动气压梯度力"
        Wrf_wf.data_save(self,self.right4, save_path=save_path4)

        save_path5 = str(self.save_path) + "/水物质拖曳力"
        Wrf_wf.data_save(self,self.right5, save_path=save_path5)

        save_path6 = str(self.save_path) + "/扰动干空气浮力"
        Wrf_wf.data_save(self,self.right6, save_path=save_path6)

    def end(self):
        Wrf_wf.step_one(self)
        Wrf_wf.step_two(self)
        Wrf_wf.step_three(self)
        print('计算结束')

# 滞弹性近似垂直运动方程(参考 onenote 2023 11.13）
class get_var_base():
    # 求变量时间维度上的基态
    def __init__(self,wrfpath_all,var,save_path,style='time'):
        self.z_level = np.insert(np.arange(1000, 10000 + 500, 500), 0, [250, 500, 750])
        self.wrf_path = wrfpath_all
        self.var = var
        self.save_path = save_path
        self.style = style
    def cal_pr(self):
        data_all = []
        for i in np.arange(len(self.wrf_path)):
            fn = nc.Dataset(self.wrf_path[i])
            z = getvar(fn,'z')
            p = getvar(fn,self.var)
            p_z = interplevel(p,z,self.z_level)
            data_all.append(p_z)
        # 求时间平均
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
        # 时间空间平均
        if self.style == 'time_area':
            data_mean_time = np.squeeze(np.nanmean(np.array(data_all), axis=0))
            data_mean_time_area = np.squeeze(np.nanmean(np.array(data_mean_time), axis=(1,2))).reshape((-1,1,1))*np.ones(data_mean_time.shape)
            self.save_name = str(self.save_path) + "/" + str(self.var) + '_' + str(self.style) + '.bin'
            np.array(data_mean_time_area).astype(np.float32).tofile(self.save_name)
            self.data_shape = data_mean_time_area.shape
        else:
            data_mean = np.squeeze(np.nanmean(np.array(data_all), axis=0))
            self.save_name = str(self.save_path) + "/" + str(self.var) + '_' + str(self.style) + '.bin'

            np.array(data_mean).astype(np.float32).tofile(self.save_name)
            self.data_shape = data_mean.shape

class Ztx():
    # 集中方法，公式都是相同的，写一个公式的函数就好了
    def __init__(self, wrf_path, save_path,**kwargs):
        self.z_level = np.insert(np.arange(1000,10000+500,500), 0, [250, 500, 750])
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.wrf_path = wrf_path
        self.save_path = save_path
        self.kwargs_default = {'p_b':1e+20,'theta_b':1e+20}
        self.kwargs_default.update(kwargs)

    def cal_data(self,rho_b_z,pr,theta_r_z,theta_z_b,P_mean):
        part_1 = -1/rho_b_z * np.gradient(pr,self.z_level,axis=0)
        part_2 = self.__g* np.array(theta_r_z) / np.array(theta_z_b)
        part_3 = - (self.__cv / self.__cp) * (np.array(pr) / np.array(P_mean))*self.__g
        self.data_all = np.concatenate((part_1, part_2, part_3))

    def save_data(self):
        # 保存
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
        save_name = str(self.save_path) + "/" + str(self.time) + '.bin'

        np.array(self.data_all).astype(np.float32).tofile(save_name)

    def cal_style_one(self):
        # 使用pressure的空间平均作为基态 使用公式计算（冉）
        fn = nc.Dataset(self.wrf_path)
        self.time = get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        z = getvar(fn,'z')
        theta = getvar(fn,'theta')
        # 和使用平均的方法一样，也是将P的mean作为基态气压，然后得到扰动气压
        P = interplevel(getvar(fn,'pressure')*100, z, self.z_level)  # 单位Pa
        Pb = np.nanmean(P,axis=(1,2)).reshape((-1,1,1))*np.ones(P.shape)
        Pr = P - Pb
        theta_z = interplevel(theta, z, self.z_level)
        dpb_z_dz = np.gradient(Pb, self.z_level, axis=0)
        rho_b_z = - dpb_z_dz / self.__g
        T_z_b = Pb / (rho_b_z * self.__R)
        theta_z_b = T_z_b * (self.__ps / Pb) ** (self.__R / self.__cp)
        theta_r_z = theta_z - theta_z_b
        Ztx.cal_data(self,rho_b_z=rho_b_z,pr=Pr,theta_r_z=theta_r_z,theta_z_b=theta_z_b,P_mean=Pb)
        Ztx.save_data(self)



        pass

    def cal_style_two(self,p_time_mean):
        # 使用时间平均的气压数据计算基态位温
        # 输入单位应该是Pa
        fn = nc.Dataset(self.wrf_path)
        self.time = get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        z = getvar(fn,'z')
        theta = getvar(fn,'theta')
        P_z = interplevel(getvar(fn,'pressure'),z,self.z_level) * 100
        pr = P_z - p_time_mean
        # 和使用平均的方法一样，也是将P的mean作为基态气压，然后得到扰动气压
        theta_z = interplevel(theta, z, self.z_level)
        dpb_z_dz = np.gradient(p_time_mean, self.z_level, axis=0)
        rho_b_z = - dpb_z_dz / self.__g
        T_z_b = p_time_mean / (rho_b_z * self.__R)
        theta_z_b = T_z_b * (self.__ps / p_time_mean) ** (self.__R / self.__cp)
        theta_r_z = theta_z - theta_z_b
        Ztx.cal_data(self,rho_b_z=rho_b_z,pr=pr,theta_r_z=theta_r_z,theta_z_b=theta_z_b,P_mean=p_time_mean)
        Ztx.save_data(self)

    def cal_style_three(self,p_time_mean,theta_time_mean):
        # 使用时间平均的基态气压和基态位温，直接计算
        fn = nc.Dataset(self.wrf_path)
        self.time = get_times(fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        z = getvar(fn,'z')
        theta = getvar(fn,'theta')
        P_z = interplevel(getvar(fn,'pressure'),z,self.z_level) * 100
        pr = P_z - p_time_mean
        # 和使用平均的方法一样，也是将P的mean作为基态气压，然后得到扰动气压
        theta_z = interplevel(theta, z, self.z_level)
        dpb_z_dz = np.gradient(p_time_mean, self.z_level, axis=0)
        rho_b_z = - dpb_z_dz / self.__g
        theta_r_z = theta_z - theta_time_mean
        Ztx.cal_data(self,rho_b_z=rho_b_z,pr=pr,theta_r_z=theta_r_z,theta_z_b=theta_time_mean,P_mean=p_time_mean)
        Ztx.save_data(self)


# 适用于中尺度对流系统的动量方程《吉林一次极端降水发生发展动热力过程的数值模拟分析 》
# 《雷暴大风过程中对流层中低层动量通量和动能通量输送特征研究》
class Momentum_equation():
    def __init__(self,wrf_path, save_path):
        self.fn = nc.Dataset(wrf_path)
        self.__g = 9.8
        self.__R = 287.05
        self.__ps = 100000.0
        self.__cp = 1005.0
        self.__cv = 718.0
        self.__f0 = 4.97e-5
        self.z_level = np.arange(250,10000+250,250)
        self.DZ = 250
        self.z = getvar(self.fn,'z')
        self.time = get_times(self.fn).dt.strftime("%Y-%m-%d_%H:%M").values.flatten()[0]
        self.save_path = save_path
    def cal_one(self):
        u = getvar(self.fn,'ua')
        v = getvar(self.fn,'va')
        w = getvar(self.fn,'wa')
        theta = getvar(self.fn,'theta')
        P = getvar(self.fn,'pressure')*100
        T = np.squeeze(wrf.tk(P, theta).values)
        rho = P / (self.__R * T)

        # 差值
        self.rho = interplevel(rho,self.z,self.z_level)
        self.P = interplevel(P,self.z,self.z_level)
        self.u = interplevel(u,self.z,self.z_level)
        self.v = interplevel(v,self.z,self.z_level)
        self.w = interplevel(w,self.z,self.z_level)

        self.DX = self.fn.DX
        self.DY = self.fn.DY

    def cal_two(self):
        dru_dx,dru_dy,dru_dz = np.gradient(self.rho*self.u,self.DX,axis=2),np.gradient(self.rho*self.u,self.DY,axis=2),np.gradient(self.rho*self.u,self.DZ,axis=0)
        du_dx,dv_dy,dw_dz = np.gradient(self.u,self.DX,axis=2),np.gradient(self.v,self.DY,axis=2),np.gradient(self.w,self.DZ,axis=0)
        part1_I_I = self.u*dru_dx + self.v*dru_dy + self.w*dru_dz
        part1_I_II = self.rho*self.u*(du_dx + dv_dy + dw_dz)
        part1_I = -(part1_I_I + part1_I_II)
        part1_II = self.__f0*self.rho*self.v
        part1_III = -np.gradient(self.P,self.DX,axis=2)
        #part1 = part1_I + part1_II + part1_III

        drv_dx,drv_dy,drv_dz = np.gradient(self.rho*self.v,self.DX,axis=2),np.gradient(self.rho*self.v,self.DY,axis=2),np.gradient(self.rho*self.v,self.DZ,axis=0)
        part2_I_I = self.u*drv_dx + self.v*drv_dy + self.w*drv_dz
        part2_I_II = self.rho*self.v*(du_dx + dv_dy + dw_dz)
        part2_I = -(part2_I_I + part2_I_II)
        part2_II = -self.__f0*self.rho*self.v
        part2_III = -np.gradient(self.P,self.DY,axis=1)
        # part2 = part2_I + part2_II + part2_III

        data_all = np.concatenate((part1_I,part1_II,part1_III,part2_I,part2_II,part2_III)).astype(np.float32)
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
        save_name = str(self.save_path) + "/" + str(self.time) + '.bin'
        np.array(data_all).astype(np.float32).tofile(save_name)

    def end(self):
        Momentum_equation.cal_one(self)
        Momentum_equation.cal_two(self)
        print('计算一次')

class  Momentum_energy_equation(Momentum_equation):
    def __init__(self,wrf_path, save_path):
        super.__init__(wrf_path, save_path)
        Momentum_equation.cal_one(self)

    def step_one(self):
        k = 1/2*(self.u**2 + self.v**2)  # 水平风动能
        rk = self.rho * k
        drk_dx,drk_dy = np.gradient(rk,self.DX,axis=2), np.gradient(rk,self.DY,axis=1)
        du_dx,dv_dy = np.gradient(self.u,self.DX,axis=2), np.gradient(self.v,self.DY,axis=1)
        part_i = self.u*drk_dx + self.v*drk_dy
        part_ii = rk*(du_dx + dv_dy)
        self.data_all = part_i + part_ii

    def step_two(self):
        if not os.path.exists(os.path.join(self.save_path, )):
            try:
                os.makedirs(os.path.join(self.save_path, ))
            except OSError:
                pass
        save_name = str(self.save_path) + "/" + str(self.time) + '.bin'
        np.array(self.data_all).astype(np.float32).tofile(save_name)

    def end2(self):  # 换成end2，不然就是对该方法的重写了。。
        Momentum_energy_equation.step_one(self)
        Momentum_energy_equation.step_two(self)






