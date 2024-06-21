import numpy as np

def cut(start,end,shift):
    return (Ellipsis,)+(slice(start,end),)+(slice(None),)*(shift)

def indices(i,dim):
    return (Ellipsis, i, slice(None), slice(None))[:(2+dim)]
   
def store_interfaces(self,M,dim) -> None:
    shift=self.ndim+self.dims2[dim]-1
    axis = -(self.dims2[dim]+1)
    self.MR_fp[dim][cut(None,-1,shift)] = M[indices( 0,self.dims2[dim])]
    self.ML_fp[dim][cut(1 ,None,shift)] = M[indices(-1,self.dims2[dim])]

def apply_interfaces(self,M,dim):
    shift=self.ndim+self.dims2[dim]-1
    M[indices( 0,self.dims2[dim])] = self.MR_fp[dim][cut(None,-1,shift)]
    M[indices(-1,self.dims2[dim])] = self.ML_fp[dim][cut(1, None,shift)]


def store_BC(self,BC_array,M,dim) -> None:
    #We store the intercell face/corner values to be used when performing
    #the Riemann Solver
    na=np.newaxis
    if self.BC[dim] == "periodic":
        BC_array[0] = slice_array(M,dim,-1,self.ndim)
        BC_array[1] = slice_array(M,dim, 0,self.ndim)
    elif self.BC[dim] == "reflective":
        BC_array[0] = slice_array(M,dim, 0,self.ndim)
        BC_array[1] = slice_array(M,dim,-1,self.ndim)
        BC_array[:,self.dims2[dim]] = -BC_array[:,self.dims2[dim]]
    elif self.BC[dim] == "gradfree":
        BC_array[0] = slice_array(M,dim, 0,self.ndim)
        BC_array[1] = slice_array(M,dim,-1,self.ndim)
    else:
        raise("Undetermined boundary type")
                         
def apply_BC(self,dim) -> None:
    #Here we fill up the missing first column of U_L
    #and the missing last column of U_R
    if dim=="x":
        #nvar,nader,Nz,Ny,Nx+1,p+1,p+1
        if self.Z:
            self.dm.ML_fp_x[..., 0,:,:] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1,:,:] = self.dm.BC_fp_x[1]
        elif self.Y:
            self.dm.ML_fp_x[..., 0,:] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1,:] = self.dm.BC_fp_x[1]
        else:
            self.dm.ML_fp_x[..., 0] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1] = self.dm.BC_fp_x[1]
    elif dim=="y":
        #nvar,nader,Nz,Ny+1,Nx,p+1,p+1
        if self.Z:
            self.dm.ML_fp_y[..., 0,:,:,:] = self.dm.BC_fp_y[0]
            self.dm.MR_fp_y[...,-1,:,:,:] = self.dm.BC_fp_y[1]
        else:
            self.dm.ML_fp_y[..., 0,:,:] = self.dm.BC_fp_y[0]
            self.dm.MR_fp_y[...,-1,:,:] = self.dm.BC_fp_y[1]
    elif dim=="z":
        #nvar,nader,Nz+1,Ny,Nx,p+1,p+1
        self.dm.ML_fp_z[..., 0,:,:,:,:] = self.dm.BC_fp_z[0]
        self.dm.MR_fp_z[...,-1,:,:,:,:] = self.dm.BC_fp_z[1]

def slice_array(M,dim,idx,ndim):
    if ndim==1:
        if dim=="x":
            return M[...,idx,idx]
        else:
            raise("Incorrect dimension")
    elif ndim==2:
        if dim=="x":
            return M[...,idx,:,idx]
        elif dim=="y":
            return M[...,idx,:,idx,:]
        else:
            raise("Incorrect dimension")
    elif ndim==3:
        if dim=="x":
            return M[...,idx,:,:,idx]
        elif dim=="y":
            return M[...,idx,:,:,idx,:]
        elif dim=="z":
            return M[...,idx,:,:,idx,:,:]
        else:
            raise("Incorrect dimension")
