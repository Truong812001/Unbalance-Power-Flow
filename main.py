import pandas as pd
import numpy as np
import cmath
import openpyxl

class NRS:

    def __init__(self,file):

        # Đọc dữ liệu từ file Excel
        self.lines = pd.read_excel(file, sheet_name='lines').values
        self.codes = pd.read_excel(file, sheet_name='line_codes').values
        self.general = pd.read_excel(file, sheet_name='general').values
        self.xy = pd.read_excel(file, sheet_name='coordinates').iloc[:, 1:].values
        self.loads = pd.read_excel(file, sheet_name='loads').values
        self.shunt = pd.read_excel(file, sheet_name='shunt').values
        self.cap = pd.read_excel(file, sheet_name='capacitor_bank').values
        self.feeder=self.Ybus()
    def Ybus(self):

        '''get data'''
        lines, codes, general, xy ,shunt = self.lines, self.codes, self.general, self.xy, self.shunt

        p_base = general[0, 1] / 3  # Công suất danh định theo pha
##        print(p_base)
        v_base = general[0, 0] / np.sqrt(3)  # Điện áp dây đến trung bình
##        print(v_base)
        z_base = v_base**2 / p_base
##        print(z_base)
##        print(p_base)
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


        ## Shunt
        # for i in range(len(shunt)):
        #     n1 = int(shunt[i,0])
        #     ph = int(shunt[i,1])
        #     q = shunt[i,2]

        #     if ph == 1 :
        #         ybus[n1-1,n1-1] -= 1j* q * z_base / (v_base**2)     ##  z_base = v_base**2 / p_base
        #     elif ph==2:
        #         ybus[n1+num_n-1,n1+num_n-1] -= 1j* q * z_base /(v_base**2)
        #     elif ph==3:
        #         ybus[n1+2*num_n-1,n1+2*num_n-1] -= 1j* q * z_base /(v_base**2)




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

        np.savetxt('ybus.txt', ybus, fmt='%1.7e')
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

        pref = np.real(sref)
        qref = np.imag(sref)


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



        err = 100
        conv = np.zeros(100)
        iter = 1
        num_t = 3 * num_n       # tổng nút
        num_r = len(n_other)    # khac slack

        '''Jacobi'''
        H = np.zeros((num_t, num_t))
        N = np.zeros((num_t, num_t))
        J = np.zeros((num_t, num_t))
        L = np.zeros((num_t, num_t))
        ybus=pd.DataFrame(ybus)
##        np.savetxt('ybus.txt', ybus, fmt='%1.7e')
        while err > 1E-9:

            vn = v * np.exp(an * 1j)


            sn = vn * np.conj(np.dot(ybus ,vn))

            p = np.real(sn)
            q = np.imag(sn)

            for k in range(num_t):
                for m in range(num_t):
                    if k == m:
                        H[k, k] = -B[k, k] * v[k] * v[k] - q[k]
                        N[k, k] = G[k, k] * v[k] + p[k] / v[k]
                        J[k, k] = -G[k, k] * v[k] * v[k] + p[k]
                        L[k, k] = -B[k, k] * v[k] + q[k] / v[k]
                    else:
                        akm = an[k] - an[m]
                        N[k, m] = v[k] * (G[k, m] * np.cos(akm) + B[k, m] * np.sin(akm))
                        L[k, m] = v[k] * (G[k, m] * np.sin(akm) - B[k, m] * np.cos(akm))
                        H[k, m] = L[k, m] * v[m]    ## check
                        J[k, m] = -N[k, m] * v[m]

            dp=np.zeros((num_r,1))
            dq=np.zeros((num_r,1))

            n_slack1=np.zeros((3,1),dtype=int)
            for i in range(len(n_slack)):
                n_slack1[i] = n_slack[i]-1

##            n_slack =[0,3,6]
            H1=H.copy()
            N1=N.copy()
            J1=J.copy()
            L1=L.copy()


            H1 = np.delete(H1, n_slack1, axis=0)
            H1 = np.delete(H1, n_slack1, axis=1)
            N1 = np.delete(N1, n_slack1, axis=0)
            N1 = np.delete(N1, n_slack1, axis=1)
            J1 = np.delete(J1, n_slack1, axis=0)
            J1 = np.delete(J1, n_slack1, axis=1)
            L1 = np.delete(L1, n_slack1, axis=0)
            L1 = np.delete(L1, n_slack1, axis=1)



            dem=0
            for i in n_other:
                dp[dem] = pref[dem] - p[i-1]
                dq[dem] = qref[dem] - q[i-1]
                dem+=1
            np.savetxt('pref.txt', pref, fmt='%1.7e')
            np.savetxt('qref.txt', qref, fmt='%1.7e')
            dpq=np.concatenate((dp,dq))
            np.savetxt('dpq.txt', dpq, fmt='%1.7e')

            top_row = np.block([[H1, N1]])
            bottom_row = np.block([[J1, L1]])
            Jac = np.block([[top_row], [bottom_row]])
            Jac = pd.DataFrame(Jac)
            np.savetxt('Jac.txt', Jac, fmt='%1.7e')

            dx = np.linalg.solve(Jac, dpq)
            np.savetxt('dx.txt', dx, fmt='%1.7e')
            dem=0
            for i in n_other:
                an[i-1] += dx[dem]
                v[i-1] += dx[dem+num_r]
                dem+=1

            err = np.linalg.norm(dx)
            conv[iter - 1] = err
            iter += 1

            if iter > 100:
                print('Phương pháp Newton. Sau 10 lần lặp, sai số là:', err)
                break
        vn = v * np.exp(an * 1j)
        res = {}
        res['jacobian'] = Jac
        res['v_node'] = vn
        res['s_node'] = vn * np.conj(np.dot(ybus ,vn))
        res['p_loss'] = np.real(np.sum(res['s_node']))
        res['error'] = conv
        res['iter'] = iter
        return res,v
    def report(self):
        res,v=self.main()
        num_n = self.feeder['num_n']
        a=[]
        b=[]
        c=[]
        for i in range(num_n):
            a.append(v[i])
            b.append(v[i+num_n])
            c.append(v[i+2*num_n])
        df = pd.DataFrame({'Phase a': a, 'Phase b': b, 'Phase c': c})

        # Lưu dataframe ra file Excel
        df.to_excel("output.xlsx", index=False)
nrs=NRS('FEEDER901.xlsx')
res=nrs.report()


### chạy 3 pha nếu để các tải giống nhau thì đãng lẽ ra các bus phải có điện áp giống nhau

## lamf database
## chuan hoa lai du lieu
## ve luoi
## viet dan
##contrain toois uwu rang buoc,  dua vao ham muc tieu. f(x)= -ploss +min(u,105 neu u-105<0 thi cho bang 0 vi dap ung dieu kien )
##so sanh valid sincal, đồ thị cần thay đổi, viet do an, luoi co nhanh, luoi cua lo 22kV
