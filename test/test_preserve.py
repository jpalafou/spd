import pytest
import sys
import numpy as np
sys.path.append("../src")
from sdader_simulator import SDADER_Simulator

tolerance=1E-14
N=6

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
@pytest.mark.parametrize("update" , ["SD","FV"])
def test_update_sd(p,N,update):
    s = SDADER_Simulator(p=p,N=N,use_cupy=False,
                         update=update)
    s.perform_time_evolution(1)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance
    
#Now for the FB scheme
@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
def test_update_fb(p,N):
    s = SDADER_Simulator(p=p,N=N,use_cupy=False,
                         update="FV",FB=True,godunov=True)
    s.perform_time_evolution(1)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance

#Now for the FB scheme
@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
def test_update(p,N):
    s = SDADER_Simulator(p=p,N=N,use_cupy=False,
                         update="FV",FB=True)
    s.perform_time_evolution(1)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance