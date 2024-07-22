try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
import numpy as np

class CommHelper():
    def __init__(self,ndim):
        
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
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

    def send_recv(self, neighbour, send_buffer, recv_buffer):
        if self.rank%2:
            self.comm.Send(send_buffer, dest=neighbour)
            self.comm.Recv(recv_buffer, source=neighbour)
        else:
            self.comm.Recv(recv_buffer, source=neighbour)
            self.comm.Send(send_buffer, dest=neighbour)
    
    def send_recv_replace(self,buffer,neighbour,side):
        self.comm.Sendrecv_replace(buffer,neighbour,sendtag=side,source=neighbour,recvtag=1-side)
        
    def reduce_min(self, M):
        if self.size>1:
            return self.comm.allreduce(M,op=MPI.MIN)
        else:
            return M