import pandas as pd
import numpy as np


class NRS:

    def __init__(self,file):

        # Đọc dữ liệu từ file Excel
        self.lines = pd.read_excel(file, sheet_name='lines').values
        self.codes = pd.read_excel(file, sheet_name='line_codes').values
        self.general = pd.read_excel(file, sheet_name='general').values
        self.xy = pd.read_excel(file, sheet_name='coordinates').iloc[:, 1:].values
        self.loads = pd.read_excel(file, sheet_name='loads').values
        print(self.loads)
        self.feeder=self.Ybus()
    def Ybus(self):

        '''get data'''
        lines, codes, general, xy = self.lines, self.codes, self.general, self.xy

        p_base = general[0, 1] / 3  # Công suất danh định theo pha
        v_base = general[0, 0] / np.sqrt(3)  # Điện áp dây đến trung bình
        z_base = v_base**2 / p_base

        num_n = int(np.max(np.max(lines[:, :2])))  # Số nút trong mạng
        num_l = len(lines)  # Số đoạn dây
        ybus = np.zeros((3 * num_n, 3 * num_n), dtype=complex)  # Ma trận dẫn suất YBus
        z_line = np.zeros((3, 3, num_l), dtype=complex)  # Ma trận ặc trưng của đoạn dây


        '''Y branch'''
        for k in range(num_l):

            n1, n2, len_, cde = lines[k]
            len_ /= 1000  # kilômét
            r1, x1, r0, x0 = codes[int(cde)-1, 1:] * len_

            zs = (2 * r1 + r0) / 3 + (2 * x1 + x0) / 3 * 1j ##
            zm = (r0 - r1) / 3 + (x0 - x1) / 3 * 1j ##

            z_line[:, :, k] = np.array([[zs, zm, zm], [zm, zs, zm], [zm, zm, zs]]) / z_base ## matrix
            yL = np.linalg.inv(z_line[:, :, k])

            nt1 = [int(n1), int(n1 + num_n), int(n1 + 2 * num_n)]       ## Node i
            nt2 = [int(n2), int(n2 + num_n), int(n2 + 2 * num_n)]       ## Node j
            for i in range(3):
                for j in range(3):

                    ybus[nt1[i]-1,nt1[j]-1] += yL[i][j]     ## Node ii
                    ybus[nt1[i]-1,nt2[j]-1] -= yL[i][j]     ## Node ij
                    ybus[nt2[i]-1,nt1[j]-1] -= yL[i][j]     ## Node ji
                    ybus[nt2[i]-1,nt2[j]-1] += yL[i][j]     ## Node jj



        vs = general[0, 2] * np.exp(np.array([0, -2 * np.pi / 3, 2 * np.pi / 3]) * 1j)  # Vector điện áp của nguồn

        n_slack = [1, num_n + 1, 2 * num_n + 1]  # Chỉ số của nút nguồn

        n_other = np.setdiff1d(np.arange(1,3 * num_n+1), n_slack)  # Chỉ số của các nút khác

        feeder = {}

        feeder['z_line'] = z_line  # Ma trận đặc trưng của đoạn dây
        feeder['ybus'] = ybus  # Ma trận dẫn suất YBus
        feeder['loads'] = self.loads
        feeder['vs_initial'] = vs  # Vector điện áp nguồn ban đầu

        feeder['vn_initial'] = np.ones((num_n - 1,1))  # Vector điện áp của các nút ban đầu

        feeder['p_base'] = p_base
        feeder['v_base'] = v_base
        feeder['n_slack'] = n_slack
        feeder['n_other'] = n_other
        feeder['num_n'] = num_n  # Số nút
        feeder['num_l'] = num_l  # Số đoạn dây

        feeder['num_d'] = len(feeder['loads'])  # Số lượng tải
##        feeder['num_e'] = len(feeder['profiles'])  # Số lượng hồ sơ tải

        return feeder
    def load(self):
        num_n=self.feeder['num_n']
        s_load=np.zeros((3*num_n,1),dtype=complex)
        for i in range(self.feeder['num_d']):
            n1 = int(self.feeder['loads'][i,0])
            ph = self.feeder['loads'][i,1]
            p = self.feeder['loads'][i,2]
            q= self.feeder['loads'][i,3]
            if ph==1 :
                s_load[n1 - 1] = p + 1j * q
            elif ph == 2:
                s_load[n1 + num_n - 1] = p + 1j * q
            elif ph == 3:
                s_load[n1 + 2 * num_n - 1] = p + 1j * q
        return s_load
nrs=NRS('FEEDER901.xlsx')
print(nrs.load())

