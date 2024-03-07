import Ybus
import numpy as np
ybus=Ybus.feeder['ybus']
print(ybus)

num_n = Ybus.feeder['num_n']

B = np.imag(ybus)
G = np.real(ybus)
n_slack = Ybus.feeder['n_slack']
n_other = Ybus.feeder['n_other']
s_load = select_scenario(feeder, tm)
sref = -s_load[n_other]
pref = np.real(sref)
qref = np.imag(sref)
v = np.ones((3 * num_n, 1))
an = np.zeros((3 * num_n, 1))
an[n_other] = np.angle(Ybus.feeder['vn_initial'])
an[n_slack] = np.angle(Ybus.feeder['vs_initial'])
v[n_slack] = np.abs(Ybus.feeder['vs_initial'])
err = 100
conv = zeros(10,1)
iter = 1
num_t = 3*num_n
num_r = length(n_other)
H = np.zeros(num_t,num_t)
N = np.zeros(num_t,num_t)
J = np.zeros(num_t,num_t)
L = np.zeros(num_t,num_t)