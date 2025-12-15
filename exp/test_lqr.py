import numpy as np
from include.lqr import discretize_sys_zoh, solve_dare
from include.params import create_default_lip_params

params = create_default_lip_params(m=1/0.8)
dt = 0.01
w = np.sqrt(params.g/params.z_c)


A = np.array([[0, 1, 0, 0],
                [w**2, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, w**2, 0]])

B = np.array([[0, 0],
                [1, 0],
                [0, 0],
                [0, -1]])

Q = np.diag([10, 1, 10, 1])
R = np.diag([0.1, 0.1])

A_d, B_d = discretize_sys_zoh(params)

print(f'A_d: \n {A_d}')
print(f'\nB_d: \n {B_d}')

P, K = solve_dare(A_d, B_d, Q, R)

print(f'\n\nP: \n {P}')
print(f'\nK: \n {K}')

sys_mat = A_d - B_d @ K

eig_vals, eig_vecs = np.linalg.eig(sys_mat)

print(f'\n\nEigenvalues: {eig_vals} \n\n')