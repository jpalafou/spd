import numpy as np
import initial_conditions_1d as ic1d
import initial_conditions_2d as ic2d
import initial_conditions_3d as ic3d

def step_function(**kwargs):
    return lambda xyz,case : ic3d.step_function(xyz,case,**kwargs)

def sine_wave(**kwargs):
    return lambda xyz,case : ic3d.sine_wave(xyz,case,**kwargs)
