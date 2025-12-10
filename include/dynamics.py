import numpy as np
from .params import LIPParams
from .gait_classes import Footstep

def _cosh_sinh(t: np.ndarray | float, T_c: float) -> tuple[np.ndarray, np.ndarray]:
    """Return C(t) = cosh(t/T_c), S(t) = sinh(t/T_c)."""
    tau = np.asarray(t) / T_c
    C = np.cosh(tau)
    S = np.sinh(tau)
    return C, S

def integrate_lip(q_i: float, qdot_i: float, p_star: float, params:LIPParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_c = params.T_c
    times = np.linspace(0, params.T_ss, num = int(params.T_ss/params.dt))
    C, S = _cosh_sinh(times, T_c)
    q_t = (q_i - p_star)*C + T_c*qdot_i*S + p_star
    qdot_t = ((q_i - p_star)/T_c)*S + qdot_i*C

    return (q_t, qdot_t, times)


def simulate_lip_virtual_inputs(s: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, 
                                dist_std: float = 0) -> np.ndarray:
    '''
    for state, s, x and y should be relative to the stance foot location
    '''

    s_next = Ad @ s + Bd @ u + np.random.normal(loc = 0, scale = dist_std)

    return s_next


def compute_virtual_inp_lim(x_rel: float, y_rel: float, params: LIPParams) -> tuple[float, float]:
    '''
    Returns tuple (u_p_max, u_r_max)
    '''
    Tp_max = 80
    Tr_max = 60

    r = np.sqrt(x_rel**2 + y_rel**2 + params.z_c**2)
    Sr = -y_rel/r
    Sp = x_rel/r

    Cr = np.sqrt(1 - Sr**2)
    Cp = np.sqrt(1 - Sp**2)
    D = np.sqrt(1 - Sr**2 - Sp**2)

    eps = 1e-3
    Cr = max(Cr, eps)
    Cp = max(Cp, eps)


    u_p_max = (D/Cp)*Tp_max
    u_r_max = (D/Cr)*Tr_max

    return (u_p_max, u_r_max)