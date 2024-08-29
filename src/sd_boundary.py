import numpy as np
from sd_simulator import SD_Simulator
from slicing import cut
from slicing import indices
from slicing import indices2
   
def store_interfaces(self: SD_Simulator,
                     M: np.ndarray,
                     dim: str) -> None:
    """
    Stores the values of flux points at the extremes of elements(0,-1)
    These arrays are then used to solve the Riemann problem
    """
    shift=self.ndim+self.dims2[dim]-1
    axis = -(self.dims2[dim]+1)
    self.MR_fp[dim][cut(None,-1,shift)] = M[indices( 0,self.dims2[dim])]
    self.ML_fp[dim][cut(1 ,None,shift)] = M[indices(-1,self.dims2[dim])]

def apply_interfaces(self: SD_Simulator,
                     F: np.ndarray,
                     F_fp: np.ndarray,
                     dim: str):
    """
    Applies the values of flux points at the extremes of elements(0,-1)
    This is done after the Riemann problem at element interfaces has been
    solved. 
    """
    shift=self.ndim+self.dims2[dim]-1
    F_fp[indices( 0,self.dims2[dim])] = F[cut(None,-1,shift)]
    F_fp[indices(-1,self.dims2[dim])] = F[cut(1, None,shift)]

def store_BC(self: SD_Simulator,
             BC_array: np.ndarray,
             M: np.ndarray,
             dim: str) -> None:
    """
    Stores the solution at flux points for the extremes of the domain
    These boundary arrays can then be communicated between domains
    """    
    idim = self.dims2[dim]
    BC = self.BC[dim]
    for side in [0,1]:
        if  BC[side] == "periodic":
            BC_array[side] = M[indices2(side-1,self.ndim,idim)]
        elif BC[side] == "reflective":
            BC_array[side] = M[indices2(-side,self.ndim,idim)]
            BC_array[side,self.vels[idim]] *= -1
        elif BC[side] == "gradfree":
            BC_array[side] = M[indices2(-side,self.ndim,idim)]
        elif BC[side] == "ic":
            next
        elif BC[side] == "eq":
            next
        elif BC[side] == "pressure":
            #Overwrite solution with ICs
            M[indices2(-side,self.ndim,idim)] = BC_array[side]
        else:
            raise("Undetermined boundary type")
                         
def apply_BC(self: SD_Simulator,
             dim: str) -> None:
    """
    Fills up the missing first column of M_L
    and the missing last column of M_R
    """
    shift=self.ndim+self.dims2[dim]-1
    self.ML_fp[dim][indices( 0,shift)] = self.BC_fp[dim][0]
    self.MR_fp[dim][indices(-1,shift)] = self.BC_fp[dim][1]

def Boundaries_sd(self: SD_Simulator,
                  M: np.ndarray,
                  dim: str):
    store_BC(self,self.BC_fp[dim],M,dim)
    store_interfaces(self,M,dim)
    Comms_fp(self,M,dim)
    apply_BC(self,dim)

def Comms_fp(self: SD_Simulator,
             M: np.ndarray,
             dim: str):
    """
    Stores the solution at flux points for the extremes of the domain
    These boundary arrays can then be communicated between domains
    """
    comms = self.comms
    rank = comms.rank
    rank_dim = comms.__getattribute__(dim)    
    idim = self.dims2[dim]
    Buffers={}
    for side in [0,1]:
        Buffer = M[indices2(-side,self.ndim,idim)]
        Buffer = self.dm.asnumpy(Buffer).flatten()
        Buffers[side] = Buffer

    neighbour = comms.left[idim] if rank%2 else comms.right[idim]
    side = rank_dim%2
    send_recv(self,neighbour,Buffers[side],dim,side)

    neighbour = comms.right[idim] if rank%2 else comms.left[idim]
    side = 1-rank_dim%2
    send_recv(self,neighbour,Buffers[side],dim,side)

def send_recv(self: SD_Simulator, neighbour, Buffer, dim: str, side):
    comms = self.comms
    rank = comms.rank
    if neighbour != rank:
        comms.send_recv_replace(Buffer,neighbour,side)
        BC = self.BC_fp[dim][side]
        self.BC_fp[dim][side][...] = self.dm.xp.asarray(Buffer).reshape(BC.shape)