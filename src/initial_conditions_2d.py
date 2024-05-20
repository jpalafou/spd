import numpy as np

def step_function(xy: np.ndarray,case: int, vx=1, vy=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,
                        np.where(np.fabs(y-0.5)<0.25,2,1),1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #Pressure
        return np.ones(x.shape)
    else:
        return np.ones(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, vx=1, vy=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return 1.0+0.125*(np.sin(2*np.pi*(x+y)))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #Pressure
        return np.ones(x.shape)
    else:
        return np.ones(x.shape)