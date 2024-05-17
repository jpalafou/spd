from enum import Enum

import numpy as np
import cupy as cp

class CupyLocation(Enum):
    device = 0
    host = 1

class GPUData:
    def __init__(self, instance, value):
        self.values = {}
        self._set(instance, value)
    def __get__(self, instance, owner):
        if instance is None:
            return self
        icl = instance.current_location
        if icl != self.current_location:
            self.values[icl] = instance.convert_value(self.values[self.current_location])
            self.current_location = icl
        return self.values[icl]
    def __set__(self, instance, value):
        self._set(instance, value)
    def _set(self, instance, value):
        self.values[instance.current_location] = instance.convert_value(value)
        self.current_location = instance.current_location

def GPUDataManager(use_gpu: bool):
    class GPUDM:
        def __init__(self, use_gpu: bool):
            super().__setattr__("use_gpu", use_gpu)
            super().__setattr__("current_location", CupyLocation.host)
        def convert_value(self, value):
            if isinstance(value, np.ndarray) and self.current_location == CupyLocation.device:
                return cp.asarray(value)
            elif isinstance(value, cp.ndarray) and self.current_location == CupyLocation.host:
                return cp.asnumpy(value)
            return value
        def switch_to(self, location: CupyLocation):
            if not self.use_gpu:
                return
            self.current_location = location
        def __setattr__(self, key, value):
            if hasattr(self, key):
                super().__setattr__(key, value)
            else:
                setattr(type(self), key, GPUData(self, value))
        @property
        def xp(self):
            return {CupyLocation.host: np, CupyLocation.device: cp}[self.current_location]
    return GPUDM(use_gpu)


