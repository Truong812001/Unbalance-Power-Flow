#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cmath
import openpyxl
import sqlite3

class Get_database():
	def __init__(self,db_file):
		self.conn=sqlite3.connect(db_file)
		self.cursor=self.conn.cursor()
		# print("YBUS\n",pd.DataFrame(self.Ybus).to_string())

	def __getDataSQL__(self):
		self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
		tables = self.cursor.fetchall()

		"""tạo dict:table chứa keys:tên cột tương ứng với các giá trị của cột"""
		"""res1={'NO': [1, 2, 3], 'NAME': ['BUS1', 'BUS2', 'BUS3'], 'kV': [1.05, 1.0, 1.0], 'FLAG': [1, 0, 0], 'PLOAD_kw': [50.0, 100.0, None]"""
		res={}
		for table in tables:


			## Tạo list chứa giá trị tên cột
			list1=[]

			k = f'PRAGMA table_info({table[0]})'
			self.cursor.execute(k)
			columns = self.cursor.fetchall()

			list1= [column[1] for column in columns]


			## Tạo câu lệnh lấy dữ liệu theo hàng
			sql=f'SELECT * FROM {table[0]}'
			self.cursor.execute(sql)
			rows=self.cursor.fetchall()

			## Tạo dict chứa thông tin của một bảng
			res1={}

			for i,key in enumerate(list1):

				## List chứa giá trị của 1 hàng
				list2=[]
				list2 = [row[i] for row in rows]

				res1[key]=list2
			res[table[0]]=res1
		"""res1={'BUS':{"NO":[.....],"Name":[.....]},"LINE":{"NO":[...,]}}"""

		return res

class NRS:

    def __init__(self,file,db_file):
        try:
            self.data = self.data_excel(file)
            print("get_excel_data",self.data)
        except:
            self.data = Get_database(db_file).__getDataSQL__()
            print("get_database",self.data)
        self.feeder = self.Ybus()
    def data_excel(self,file):

        res ={}

        res1 = {
            "No": [],
            "Frombus": [],
            "Tobus": [],
            "Length": [],
            "LineCode": []
        }
        lines = pd.read_excel(file, sheet_name='lines').values
        # Duyệt qua từng hàng trong dữ liệu và thêm vào danh sách tương ứng
        for row in lines:
            res1["No"].append(row[0])
            res1["Frombus"].append(row[1])
            res1["Tobus"].append(row[2])
            res1["Length"].append(row[3])
            res1["LineCode"].append(row[4])

        res['LINE'] = res1


        codes = pd.read_excel(file, sheet_name='line_codes').values

        # Khởi tạo từ điển res1 với các danh sách rỗng
        res1 = {
            "NAME": [],
            "R1(Ohm/km)": [],
            "X1(Ohm/km)": [],
            "R0(Ohm/km)": [],
            "X0(Ohm/km)": []
        }

        # Duyệt qua từng hàng trong dữ liệu và thêm vào danh sách tương ứng
        for row in codes:
            res1["NAME"].append(row[0])
            res1["R1(Ohm/km)"].append(row[1])
            res1["X1(Ohm/km)"].append(row[2])
            res1["R0(Ohm/km)"].append(row[3])
            res1["X0(Ohm/km)"].append(row[4])
        res['LINE_CODE'] = res1

        loads = pd.read_excel(file, sheet_name='loads').values

        # Khởi tạo từ điển res1 với các danh sách rỗng
        res1 = {
            "BUS": [],
            "Phase A (kW)": [],
            "Phase A (kVar)": [],
            "Phase B (kW)": [],
            "Phase B (kVar)": [],
            "Phase C (kW)": [],
            "Phase C (kVar)": []
        }

        # Duyệt qua từng hàng trong dữ liệu và thêm vào danh sách tương ứng
        for row in loads:
            res1["BUS"].append(row[0])
            res1["Phase A (kW)"].append(row[1])
            res1["Phase A (kVar)"].append(row[2])
            res1["Phase B (kW)"].append(row[3])
            res1["Phase B (kVar)"].append(row[4])
            res1["Phase C (kW)"].append(row[5])
            res1["Phase C (kVar)"].append(row[6])
        res['LOADS'] = res1

        general = pd.read_excel(file, sheet_name='general').values

        # Khởi tạo từ điển res1 với các danh sách rỗng
        res1 = {
            "Voltage (kV)": [],
            "Nominal Power (MW)": [],
            "Voltage at subestation (pu)": []
        }

        # Duyệt qua từng hàng trong dữ liệu và thêm vào danh sách tương ứng
        for row in general:
            res1["Voltage (kV)"].append(row[0])
            res1["Nominal Power (MW)"].append(row[1])
            res1["Voltage at subestation (pu)"].append(row[2])
        res['GEN'] = res1

        capacitor_bank = pd.read_excel(file, sheet_name='capacitor_bank').values
        res1 = {
             "No": [],
             "Q (kVar)" : []
        }
        for row in capacitor_bank:
             res1["No"].append(row[0])
             res1["Q (kVar)"].append(row[1])
        res['CAPACITOR_BANK'] = res1
        return res


    def Ybus(self):

        '''get data'''

        '''{'GEN': {'Voltage (kV)': [0.416], 'Nominal Power (MW)': [0.8],
                 'Voltage at subestation (pu)': [1.05]},
         'LINE': {'No': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  'Frombus': [1, 2, 3, 4, 5, 2, 7, 9, 9, 10, 5],
                  'Tobus': [2, 3, 4, 5, 6, 7, 4, 10, 12, 11],
                  'Length': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  'LineCode': ['AC-70', 'AC-80', 'AC-70', 'AC-80', 'AC-90', 'AC-70', 'AC-80', 'AC-90', 'AC-90', 'AC-80', 'AC-80']},
        'LINE_CODE': {'NAME': ['AC-70', 'AC-80', 'AC-90'],
                      'R1(Ohm/km)': [0.63, 0.735, 0.525],
                      'X1(Ohm/km)': [1.23, 1.435, 1.025],
                      'R0(Ohm/km)': [0.63, 0.735, 0.525],
                      'X0(Ohm/km)': [1.23, 1.435, 1.025]},
        'LOADS': {'BUS': [2], 'Phase A (kW)': [0.861], 'Phase A (kVar)': [0.122997],
                  'Phase B (kW)': [0.861], 'Phase B (kVar)': [0.122997],
                  'Phase C (kW)': [0.861], 'Phase C (kVar)': [0.122997]}}'''

        p_base = self.data['GEN']['Nominal Power (MW)'][0] ## / 3  # Công suất danh định theo pha

        v_base = self.data['GEN']['Voltage (kV)'][0] ##/ np.sqrt(3)  # Điện áp dây đến trung bình

        z_base = v_base**2 / p_base
##        print(z_base)
##        print(p_base)
        num_n = int(max(self.data['LOADS']['BUS']))  # Số nút trong mạng
        num_l = int(max(self.data['LINE']['No']))  # Số đoạn dây
        ybus = np.zeros((3 * num_n, 3 * num_n), dtype=complex)  # Ma trận dẫn suất YBus
        z_line = np.zeros((3, 3, num_l), dtype=complex)  # Ma trận ặc trưng của đoạn dây


        '''Y branch'''
        for k in range(num_l):
            line=self.data['LINE']
            n1, n2, len_, cde = line['Frombus'][k],line['Tobus'][k],line['Length'][k],line['LineCode'][k]
            # len_ /= 1000  # kilômét

            line_code_data = self.data['LINE_CODE']
            if cde in line_code_data['NAME']:
                index = line_code_data['NAME'].index(cde)
                r1 = line_code_data['R1(Ohm/km)'][index] * len_
                r0 = line_code_data['R0(Ohm/km)'][index] * len_
                x1 = line_code_data['X1(Ohm/km)'][index] * len_
                x0 = line_code_data['X0(Ohm/km)'][index] * len_


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


        vs = self.data['GEN']['Voltage at subestation (pu)'][0] * np.exp(np.array([0, -2 * np.pi / 3, 2 * np.pi / 3]) * 1j)  # Vector điện áp của nguồn
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

        return feeder
    def loads(self):
        num_n=self.feeder['num_n']
        s_load=np.zeros((3*num_n,1),dtype=complex)
        load = self.data['LOADS']
        for i in range(len(load['BUS'])):
            pa=load['Phase A (kW)'][i]
            qa=load['Phase A (kVar)'][i]
            pb=load['Phase B (kW)'][i]
            qb=load['Phase B (kVar)'][i]
            pc=load['Phase C (kW)'][i]
            qc=load['Phase C (kVar)'][i]
            # self.feeder['p_base']*1000
            n1 = int(load['BUS'][i])
            s_load[n1 - 1] = -(pa + 1j * qa)/(self.data['GEN']['Nominal Power (MW)'][0]*1000)
            s_load[n1 + num_n - 1] = -(pb + 1j * qb)/(self.data['GEN']['Nominal Power (MW)'][0]*1000)
            s_load[n1 + 2 * num_n - 1] = -(pc + 1j * qc)/(self.data['GEN']['Nominal Power (MW)'][0]*1000)
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
        s_load = self.loads()

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
            dpq=np.concatenate((dp,dq))
            top_row = np.block([[H1, N1]])
            bottom_row = np.block([[J1, L1]])
            Jac = np.block([[top_row], [bottom_row]])
            Jac = pd.DataFrame(Jac)

            dx = np.linalg.solve(Jac, dpq)
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
        res['v_node'] = v
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
        print(df)
        # Lưu dataframe ra file Excel
        df.to_excel("output.xlsx", index=False)

if __name__ == '__main__' :
    file =r'FEEDER901x.xlsx'
    db_file = r'data.db'
    nrs=NRS(file,db_file)
    res=nrs.report()


### chạy 3 pha nếu để các tải giống nhau thì đãng lẽ ra các bus phải có điện áp giống nhau

## lamf database
## chuan hoa lai du lieu
## ve luoi
## viet dan
##contrain toois uwu rang buoc,  dua vao ham muc tieu. f(x)= -ploss +min(u,105 neu u-105<0 thi cho bang 0 vi dap ung dieu kien )
##so sanh valid sincal, đồ thị cần thay đổi, viet do an, luoi co nhanh, luoi cua lo 22kV
