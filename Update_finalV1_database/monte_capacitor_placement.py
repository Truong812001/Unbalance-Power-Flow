import Newton
import numpy as np
import pandas as pd
import time
import pyswarms as ps


class RecursiveMatrix:
    def __init__(self):

        self.newton = Newton.NRS(file,db_file)
        self.feeder = self.newton.feeder
        self.data = self.newton.data
        self.matrix = self.feeder['ybus']
        self.Q =self.data['CAPACITOR_BANK']['Q (kVar)']
        self.k,self.v_max,self.v_min = self.data['SETTING']['k'],self.data['Setting']['v_max'],self.data['Setting']['v_min']
        self.loss = 100000000000
        self.dem = 0
    def recursive(self, matrix, depth=0 ,location={}):
        if depth == 3:
            # print(pd.DataFrame(matrix))
            # print(matrix)

            ## so truong hop
            self.dem +=1
            # print(self.dem)


            self.feeder['ybus'] = matrix
            # print(matrix)
            res,v = self.newton.main()
            # self.newton.report()
            # print(res['p_loss'])
##            fit_ness = self.fitness(v,res['p_loss'],self.k,self.v_max,self.v_min)

            fit_ness = res['p_loss']
            # print(fit_ness)
            # print(location)
            if self.loss >= fit_ness:
                self.loss = fit_ness
                print(location)
                print(self.loss)


##            self.loss = min(fit_ness,self.loss)
##            print(self.loss)
                ##get vi tri tu kieu gi
##                print(loss)
            # print(res)
            # self.newton.report()

            return

        ## chạy số nút ## nếu là 0 thì không đặt
        for i in range(self.feeder['num_n']):
            if i != 0:
                ## chạy phase
                for j in range(3):

                    new_matrix = matrix.copy()
                    # print('bus' ,i+1)
                    # print('phase ',j+1)
                    # print("gia trị tụ ", self.Q[depth])
                    ##phase A
                    if j == 0:
                        new_matrix[i][i] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                        location[self.Q[depth]] = [i+1,'Phase A']
                    ## phase B
                    elif j == 1:
                        new_matrix[i+self.feeder['num_n']][i+self.feeder['num_n']] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                        location[self.Q[depth]] = [i+1,'Phase B']
                    ## phase C
                    elif j == 2:
                        location[self.Q[depth]] = [i+1,'Phase C']
                        new_matrix[i+2*self.feeder['num_n']][i+2*self.feeder['num_n']] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                    self.recursive(new_matrix, depth + 1,location)
            else:
                location[self.Q[depth]] = 0
                new_matrix = matrix.copy()
                self.recursive(new_matrix, depth + 1,location)

    def fitness(self,v,p_loss,k,v_min,v_max):
        fit_ness = p_loss*k

        for i in range(len(v)):
            if v[i]>v_max:
                fit_ness += (v[i]-v_max)*k
            elif v[i]<v_min:
                fit_ness += (v_min-v[i])*k
        return fit_ness


class PSO:
    def __init__(self):
        self.newton = Newton.NRS(file,db_file)
        self.feeder = self.newton.feeder
        self.data = self.newton.data
        self.matrix = self.feeder['ybus']
        self.Q =self.data['CAPACITOR_BANK']['Q (kVar)']

    def fit_ness(self,res):
        obj = res['p_loss']
        obj += self 
    def ploss(self,x):
        n_particles, dimensions = x.shape

        new_matrix = self.matrix.copy()
        # Chuyển đổi các giá trị liên tục trong x thành chỉ số vị trí
        x_index = np.round(x).astype(int)
        # x_index = np.clip(x_index, 0, 8)  # Đảm bảo chỉ số nằm trong khoảng từ 0 đến 8
        # for i in range(dimensions):
        #     if x_index[0][i]==2:
        #         return 1000000000
        #     elif x_index[0][i] ==4:
        #         return 1000000000

        for i in range(dimensions):
            if x_index[0][i]!=0:
                if (x_index[0][i]+1) in self.feeder['n_slack']:
                    return 1000000000

        for i in range(dimensions):
            pos = x_index[0][i]
            if pos!= 0 :
                new_matrix[pos][pos] += (1j * self.Q[i]) / (self.feeder['p_base']* 1000)

        self.feeder['ybus'] = new_matrix
        res,v = self.newton.main()


            # Tính toán giá trị hàm PLoss (ví dụ đơn giản: tổng các phần tử)

        return res['p_loss']

    def run(self):
        # Giới hạn trên và dưới cho các biến trong mảng 3x3
        # Số chiều của bài toán là 1 (chỉ cần xác định 1 vị trí từ 0 đến 8)
        dimensions = len(self.Q)  ## số tụ

        ## giới hạn điểm nút pha cho tụ [2 bus aa aa bb bb cc cc  -> giới hạn bằng 5] -1 do out of range
        bounds = (0 * np.ones(dimensions), (self.feeder['num_n']*3-1) * np.ones(dimensions))

        # Khởi tạo đối tượng PSO với các tham số cần thiết
        options = {'c1': 2, 'c2': 2, 'w': 0.9}


        # Tạo đối tượng PSO
        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)
        # Tối ưu hóa hàm PLoss
        cost, pos = optimizer.optimize(self.ploss, iters=20000)
        pos = np.round(pos).astype(int)
        print(pos)


##        new_matrix = self.matrix.copy()
##        for i in range(dimensions):
##            if pos[i]!=0:
##                if (pos[i]+1) in self.feeder['n_slack']:
##                    return 1000000000
##
##        for i in range(dimensions):
##            pos1 = pos[i]
##            if pos1!= 0 :
##                new_matrix[pos1][pos1] += (1j * self.Q[i]) / (self.feeder['p_base']* 1000)
##        self.feeder['ybus'] = new_matrix
##        res,v = self.newton.main()
##        print(res['p_loss'])
        self.result(pos)
    def result(self,pos):
        res={}
        for i in range(len(pos)):
            if pos[i] ==0 :
                res[self.Q[i]] = 0
            elif pos[i] < self.feeder['num_n'] :
                res[self.Q[i]] = [pos[i]+1, 'phase A']
            elif pos[i] < 2*self.feeder['num_n'] :
                res[self.Q[i]] = [pos[i]+1-self.feeder['num_n'], 'phase B']
            elif pos[i] < 3*self.feeder['num_n'] :
                res[self.Q[i]] = [pos[i]+1-2*self.feeder['num_n'], 'phase C']
        print(res)
        # Chuyển đổi vị trí tối ưu thành mảng nhị phân 0 hoặc 1




if __name__ == '__main__':
    file =r'FEEDER901x.xlsx'
    db_file = r'data.db'

    ### MOnte
##    rm = RecursiveMatrix()
##    start_time = time.time()
##    rm.recursive(rm.matrix)
##    end_time = time.time()
##    execution_time = end_time - start_time
##    print(f"Thời gian thực thi: {execution_time} giây")

    ## PSO
    pso=PSO()
    start_time = time.time()
    pso.run()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian thực thi: {execution_time} giây")
### ve ham abs


### chuan hoa lai du lieu db co the dung excel va db
###
# # if __name__ == '__main__':

#     file =r'FEEDER901xx.xlsx'
#     db_file = r'data.db'
# #     matrix = np.zeros((6, 6))
# #     monte().recursive(matrix)
# #     # nrs=Newton.NRS(file,db_file)
# #     # res=nrs.report()