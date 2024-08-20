import matplotlib.pyplot as plt
import numpy as np
from simulator import Simulator
from typing import Tuple

def plot_fields(s: Simulator,
                M: np.ndarray,
                dim: str = "z",
                transpose: bool = True,
                regular: bool = False,
                figsize: Tuple = (6,4),
                **kwargs):
    
    fig,axs = plt.subplots(1,s.nvar,figsize=(figsize[0]*s.nvar,figsize[1]))
    if regular:
        M = s.regular_mesh(M)
    if transpose:
        M=s.transpose_to_fv(M)
    for var in range(s.nvar):
        plt.sca(axs[var])
        if s.ndim==2:
            x,y = s.regular_faces()
            plt.pcolormesh(x,y,M[var],**kwargs)
            plt.colorbar()
        elif s.ndim==3:
            x,y,z = s.regular_faces()
            if dim=="z":
                plt.pcolormesh(x,y,M[var,M.shape[1]//2],**kwargs)
            elif dim=="y":
                plt.pcolormesh(x,z,M[var,:,M.shape[2]//2],**kwargs)
            plt.colorbar()
        else:
            x = s.regular_centers()[0]
            plt.plot(x,M[var])
        plt.title(s.variables[var],**kwargs)