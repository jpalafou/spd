import numpy as np

def step_function(xy: np.ndarray,case: int, vx=1):
    x=xy[0]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,2,1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #Pressure
        return np.ones(x.shape)
    else:
        return np.ones(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, vx=1):
    x=xy[0]
    if case==0:
        #density
        return 1.0+0.125*(np.sin(2*np.pi*x))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #Pressure
        return np.ones(x.shape)
    else:
        return np.ones(x.shape)