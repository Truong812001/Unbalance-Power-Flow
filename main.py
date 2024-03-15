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





        n_slack = [1, num_n + 1, 2 * num_n + 1]  # Chỉ số của nút nguồn

        n_other = np.setdiff1d(np.arange(1,3 * num_n+1), n_slack)  # Chỉ số của các nút khác

        feeder = {}

        feeder['z_line'] = z_line  # Ma trận đặc trưng của đoạn dây
        feeder['ybus'] = ybus  # Ma trận dẫn suất YBus
        feeder['loads'] = self.loads

        vs = general[0, 2] * np.exp(np.array([0, -2 * np.pi / 3, 2 * np.pi / 3]) * 1j)  # Vector điện áp của nguồn
        feeder['vs_initial'] = vs  # Vector điện áp nguồn ban đầu
        vn = np.zeros((3*(num_n-1),1),dtype=complex)
        for i in range(num_n-1):
            vn[i]=vs[0]
            vn[i+num_n-1]=vs[1]
            vn[i+2*(num_n-1)]=vs[2]
        feeder['vn_initial'] = vn  # Vector điện áp của các nút ban đầu

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
    def main(self):
        num_n=self.feeder['num_n']
        ybus=self.feeder['ybus']

        G = np.real(ybus)
        B = np.imag(ybus)

        ''''''
        n_slack=self.feeder['n_slack']
        n_other=self.feeder['n_other']

        '''get load'''
        s_load = self.load()
        sref=np.zeros((3*(num_n-1),1),dtype=complex)
        dem=0
        for i in n_other:
            sref[dem] = s_load[i-1][0]
            dem+=1
        print(sref)
        pref = np.real(sref)
        qref = np.imag(sref)
        print(pref)


        '''preset'''
        v = np.ones(3 * num_n)
        an = np.zeros(3 * num_n)
        vn_initial=self.feeder['vn_initial']
        vs_initial=self.feeder['vs_initial']


        dem=0
        for i in n_other:
            an[i-1] = np.angle(vn_initial[dem])
            dem += 1
        dem=0
        for i in n_slack:
            an[i-1] = np.angle(vs_initial[dem])
            v[i-1] = np.abs(vs_initial[dem])
            dem+=1

        print(an)
        print(v)
        err = 100
        conv = np.zeros(10)
        iter = 1
        num_t = 3 * num_n       # tổng nút
        num_r = len(n_other)    # khac slack

        '''Jacobi'''
        H = np.zeros((num_t, num_t))
        N = np.zeros((num_t, num_t))
        J = np.zeros((num_t, num_t))
        L = np.zeros((num_t, num_t))
        while err > 1E-9:

            vn = v * np.exp(an * 1j)
            sn = vn * np.conj(ybus.dot(vn))
            p = np.real(sn)
            q = np.imag(sn)

            for k in range(num_t):
                for m in range(num_t):
                    if k == m:
                        H[k, k] = -B[k, k] * v[k] * v[k] - q[k]     #J1 check
                        N[k, k] = G[k, k] * v[k] + p[k] / v[k]      #J2
                        J[k, k] = -G[k, k] * v[k] * v[k] + p[k]     #J3 check
                        L[k, k] = -B[k, k] * v[k] + q[k] / v[k]     #J4
                    else:
                        akm = an[k] - an[m]
                        N[k, m] = v[k] * (G[k, m] * np.cos(akm) + B[k, m] * np.sin(akm))    #J2
                        L[k, m] = v[k] * (G[k, m] * np.sin(akm) - B[k, m] * np.cos(akm))    #J4
                        H[k, m] = L[k, m] * v[m]    #J1   J1=J4*v
                        J[k, m] = -N[k, m] * v[m]   #J3   J3=-J2*v


            dp = pref - p[n_other]
            dq = qref - q[n_other]
            Jac = np.block([[H[n_other, n_other], N[n_other, n_other]],
                            [J[n_other, n_other], L[n_other, n_other]]])
            dx = np.linalg.solve(Jac, np.concatenate((dp, dq)))
            an[n_other] += dx[:num_r]
            v[n_other] += dx[num_r:]
            err = np.linalg.norm(dx)
            conv[iter - 1] = err
            iter += 1

            if iter > 10:
                print('Phương pháp Newton. Sau 10 lần lặp, sai số là:', err)
                break
        vn = v * np.exp(an * 1j)
        res = {}
        res['jacobian'] = Jac
        res['v_node'] = vn
        res['s_node'] = vn * np.conj(feeder.ybus.dot(vn))
        res['p_loss'] = np.real(np.sum(res['s_node']))
        res['error'] = conv
        res['iter'] = iter
nrs=NRS('FEEDER901.xlsx')
print(nrs.main())

