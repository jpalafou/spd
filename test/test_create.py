import pytest
import sys
import numpy as np
sys.path.append("../src")
from simulator import Simulator
from sd_simulator import SD_Simulator
from fv_simulator import FV_Simulator
from sdader_simulator import SDADER_Simulator

N=6

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N),(N,N,N)])
@pytest.mark.parametrize("simulator" , [Simulator,SD_Simulator,FV_Simulator,SDADER_Simulator])
def test_create(p,N,simulator):
    s = simulator(p=p,N=N,use_cupy=False)