import numpy as np
from .params import LIPParams



def calc_next_step(px_prev: float, py_prev: float, sx, sy, n: int) -> tuple[float, float]:
    '''
    Calculates next step location using step size.
    Assumes that the first support foot is the right foot.
    
    px_prev: x position of foot at step n-1
    :param py_prev: y position of foot at step n-1
    s_x: Step size in x-direction (step length)
    s_y: Step size in y-direction (step width)
    param n: What step we are on (starts at n=0)
    '''
    px = px_prev + sx
    py = py_prev - ((-1)**n)*sy

    return (px, py)


def calc_walk_primitive(sx_next: float, sy_next: float, params: LIPParams, n: int) -> tuple[tuple[float, float], tuple[float, float]]:
    '''
    Docstring for calc_walk_primitive
    '''
    xbar = sx_next/2
    ybar = ((-1)**n)*sy_next/2

    T_c = params.T_c
    T_ss = params.T_ss
    C = np.cosh(T_ss/T_c)
    S = np.sinh(T_ss/T_c)

    vbar_x = ((C+1)/(T_c*S))*xbar
    vbar_y = ((C-1)/(T_c*S))*ybar

    return ((xbar, ybar), (vbar_x, vbar_y))

def calc_target_state(p_q: float, q_bar: float, vbar_q: float) -> tuple[float, float]:
    q_d = p_q + q_bar
    qdot_d = vbar_q

    return (q_d, qdot_d)

def calc_modified_foot_placement(q_i: float, qdot_i: float, q_d: float, qdot_d: float, params:LIPParams) -> float:
    T_c = params.T_c
    T_ss = params.T_ss
    a = params.a_weight
    b = params.b_weight

    C = np.cosh(T_ss/T_c)
    S = np.sinh(T_ss/T_c)

    D = a*(C-1)**2 + b*(S/T_c)**2

    term_one = -a*((C-1)/D)*(q_d - C*q_i - T_c*S*qdot_i)
    term_two = -((b*S)/(T_c*D))*(qdot_d - S*q_i/T_c - C*qdot_i)

    return term_one + term_two


def finite_diff_first(x: np.ndarray, dt: float) -> np.ndarray:
    """Simple central-difference first derivative with one-sided at boundaries."""
    N = len(x)
    dx = np.zeros_like(x)

    if N == 1:
        return dx  # degenerate case

    # central differences for interior points
    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)

    # forward / backward for ends
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt

    return dx