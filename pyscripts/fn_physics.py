import numpy as np

def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

def heating_law(time, Tenv, T0, R, q0, S):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time) + q0*np.exp(S*time)
    return T
