import pandas as pd
import numpy as np

def read_excel(file):
    # Đọc dữ liệu từ file Excel
    lines = pd.read_excel(file, sheet_name='lines').values

    codes = pd.read_excel(file, sheet_name='line_codes').values
    general = pd.read_excel(file, sheet_name='general').values
    xy = pd.read_excel(file, sheet_name='coordinates').iloc[:, 1:].values
    return lines, codes, general, xy

def build_feeder(file):
    lines, codes, general, xy = read_excel(file)

    p_base = general[0, 1] / 3  # Công suất danh định theo pha
    v_base = general[0, 0] / np.sqrt(3)  # Điện áp dây đến trung bình
    z_base = v_base**2 / p_base

    num_n = int(np.max(np.max(lines[:, :2])))  # Số nút trong mạng
    num_l = len(lines)  # Số đoạn dây
    ybus = np.zeros((3 * num_n, 3 * num_n), dtype=complex)  # Ma trận dẫn suất YBus
    z_line = np.zeros((3, 3, num_l), dtype=complex)  # Ma trận đặc trưng của đoạn dây

    for k in range(num_l):
        n1, n2, len_, cde = lines[k]

        len_ /= 1000  # kilômét

        r1, x1, r0, x0 = codes[int(cde)-1, 1:] * len_
        zs = (2 * r1 + r0) / 3 + (2 * x1 + x0) / 3 * 1j
        zm = (r0 - r1) / 3 + (x0 - x1) / 3 * 1j
        z_line[:, :, k] = np.array([[zs, zm, zm], [zm, zs, zm], [zm, zm, zs]]) / z_base
        yL = np.linalg.inv(z_line[:, :, k])

        nt1 = [int(n1), int(n1 + num_n), int(n1 + 2 * num_n)]

        nt2 = [int(n2), int(n2 + num_n), int(n2 + 2 * num_n)]
        for i in range(3):

            for j in range(3):

                ybus[nt1[i]-1,nt1[j]-1] += yL[i][j]
                ybus[nt1[i]-1,nt2[j]-1] -= yL[i][j]
                ybus[nt2[i]-1,nt1[j]-1] -= yL[i][j]
                ybus[nt2[i]-1,nt2[j]-1] += yL[i][j]



    vs = general[0, 2] * np.exp(np.array([0, -2 * np.pi / 3, 2 * np.pi / 3]) * 1j)  # Vector điện áp của nguồn

    n_slack = [1, num_n + 1, 2 * num_n + 1]  # Chỉ số của nút nguồn

    n_other = np.setdiff1d(np.arange(1,3 * num_n+1), n_slack)  # Chỉ số của các nút khác


    feeder = {}

    feeder['z_line'] = z_line  # Ma trận đặc trưng của đoạn dây
    feeder['ybus'] = ybus  # Ma trận dẫn suất YBus


    feeder['loads'] = pd.read_excel(file, sheet_name='loads').values  # Đọc dữ liệu tải từ sheet 'loads'
    feeder['profiles'] = pd.read_excel(file, sheet_name='profiles').iloc[:, 1:].values / p_base / 1000  # Đọc dữ liệu hồ sơ tải và chuẩn hóa

    feeder['vs_initial'] = vs  # Vector điện áp nguồn ban đầu

    feeder['vn_initial'] = np.ones((num_n - 1,1))  # Vector điện áp của các nút ban đầu

    feeder['p_base'] = p_base
    feeder['v_base'] = v_base
    feeder['n_slack'] = n_slack
    feeder['n_other'] = n_other
    feeder['num_n'] = num_n  # Số nút
    feeder['num_l'] = num_l  # Số đoạn dây
    feeder['num_d'] = len(feeder['loads'])  # Số lượng tải
    feeder['num_e'] = len(feeder['profiles'])  # Số lượng hồ sơ tải

    return feeder

file = "FEEDER901.xlsx"  # Tên file Excel chứa dữ liệu
feeder = build_feeder(file)