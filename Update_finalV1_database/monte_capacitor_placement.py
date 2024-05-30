import Newton
import numpy as np
import pandas as pd 




class RecursiveMatrix:
    def __init__(self):

        self.newton = Newton.NRS(file,db_file)
        self.feeder = self.newton.feeder
        self.data = self.newton.data
        print(self.data)
        self.matrix = self.feeder['ybus']
        self.Q = [300,400,500]
        print(pd.DataFrame(self.matrix))
    
    def recursive(self, matrix, depth=0):
        if depth == 3:
            print(pd.DataFrame(matrix))
            # print(matrix)
            self.feeder['ybus'] = matrix
            self.newton.report()

            return

        for i in range(2):
            if i != 0:
                for j in range(3):
                    new_matrix = matrix.copy()
                    if j == 0:
                        new_matrix[i][i] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                    elif j == 1:
                        new_matrix[i+2][i+2] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                    elif j == 2:
                        new_matrix[i+4][i+4] += (1j * self.Q[depth]) / (self.feeder['p_base']* 1000)
                    self.recursive(new_matrix, depth + 1)
            else:
                new_matrix = matrix.copy()
                self.recursive(new_matrix, depth + 1)

if __name__ == '__main__':
    file =r'FEEDER901x.xlsx'
    db_file = r'data.db'
    rm = RecursiveMatrix()
    rm.recursive(rm.matrix)


# # if __name__ == '__main__':

#     file =r'FEEDER901xx.xlsx'
#     db_file = r'data.db'
# #     matrix = np.zeros((6, 6))
# #     monte().recursive(matrix)
# #     # nrs=Newton.NRS(file,db_file)
# #     # res=nrs.report()