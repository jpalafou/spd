try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
import numpy as np

class CommHelper():
    def __init__(self,comm,ndim):
        self.comm = comm
        if MPI_AVAILABLE:
            self.size = comm.Get_size()
            self.rank = comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0
        
        self.nx = int(self.size**(1./ndim))
        while (self.size%self.nx != 0):
            self.nx+=1
            
        if ndim>1:
            self.ny = int((self.size/self.nx)**(1./(ndim-1)))
            while (int((self.size/self.nx)%self.ny) != 0):
                self.ny+=1
        else:
            self.ny=1

        self.nz = int(self.size/self.nx/self.ny)
    
        self.x = int(self.rank%self.nx)
        self.y = int((self.rank/self.nx)%self.ny)
        self.z = int((self.rank/self.nx/self.ny))

        mesh = np.arange(self.size).reshape([self.nz,self.ny,self.nx])
        self.left={}
        self.right={}
        for dim in range(ndim):
            self.left[dim]  = np.roll(mesh,+1,axis=2-dim)[self.z,self.y,self.x]
            self.right[dim] = np.roll(mesh,-1,axis=2-dim)[self.z,self.y,self.x]

    def send_recv(self, sender, receiver, send_buffer, recv_buffer):
        if self.rank%2:
            self.comm.Send(send_buffer, dest=receiver)
            self.comm.Recv(recv_buffer, source=sender)
        else:
            self.comm.Recv(recv_buffer, source=sender)
            self.comm.Send(send_buffer, dest=receiver)