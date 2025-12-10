import numpy as np
import scipy as sp
from include.params import LIPParams

def discretize_sys_zoh(params: LIPParams) -> tuple[np.ndarray, np.ndarray]:
    '''
    Performs ZOH discretization on a system defined in state-space form as:
    s_dot = A @ s + B @ u

    Returns a tuple (A_d, B_d) such that:
    s_{k+1} = A_d @ s_k + B_d @ u_k

    This assumes sampling time of dt and system dynamics of:
    x_ddot = w^2 * x + u_p/(m*z_c)
    y_ddot = w^2 * y - u_r/(m*z_c)

    The state, s, is assumed to be:
    s = [x, x_dot, y, y_dot]'

    And the input is assumed to be:
    u = [u_p, u_r]'
    '''
    m = params.m
    g = params.g
    z_c = params.z_c
    dt = params.dt

    w = np.sqrt(g/z_c)
    C = np.cosh(w*dt)
    S = np.sinh(w*dt)

    A_d = np.array([[C, S/w, 0, 0], 
                    [w*S, C, 0, 0], 
                    [0, 0, C, S/w], 
                    [0, 0, w*S, C]])
    
    B_d = (1/(m*z_c*w))*np.array([[(C-1)/w, 0],
                                  [S, 0],
                                  [0, -(C-1)/w],
                                  [0, -S]])
    
    return (A_d, B_d)


def solve_dare(A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns a tuple (P, K) that solves the DARE for the infinite horizon
    undiscounted setting. Here, P and K are defined such that:
    u_k = -K*e_k

    Where e_k is the error at step k. 
    '''
    P = sp.linalg.solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

    return (P, K)
    