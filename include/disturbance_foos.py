import numpy as np

def norm_dist(scale = 35):
    return scale*np.random.randn(2)

def const_x_dist(scale = 20):
    return np.array([scale, 0])

def const_y_dist(scale = 20):
    return np.array([0, scale])

def unif_x_dist(scale = 40):
    return scale*np.array([np.random.rand()-0.5, 0])

def unif_y_dist(scale = 40):
    return scale*np.array([0, np.random.rand()-0.5])

def unif_dist(scale = 40):
    return scale*(np.random.rand(2) - 0.5)

def rand_imp(scale = 200):
    if np.random.rand() < 0.05:
        dir = np.random.rand()
        dir /= np.linalg.norm(dir)
        return scale*dir