import pytest
import sys
sys.path.append("../src")
import numpy as np
import polynomials as poly
from transforms import compute_A_from_B
from transforms import compute_A_from_B_full
from sd_simulator import SD_Simulator

tolerance=1E-14
N=6

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("ndim" , [1,2,3])
def test_preserve(p,ndim):
    s = SD_Simulator(p=p,Nx=N,Ny=N,Nz=N,ndim=ndim,use_cupy=False)
    s.perform_time_evolution(1)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance
