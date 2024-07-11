import numpy as np
from simulator import Simulator
from slicing import cut
from slicing import indices

def detect_troubles(self: Simulator):
    # Reset to check troubled control volumes
    ngh=self.Nghc
    self.dm.troubles[...] = 0
    self.fill_active_region(self.dm.W_cv)
    W_new = self.compute_primitives(self.dm.U_new)
    ##############################################
    # NAD Check for numerically adimissible values
    ##############################################
    # First check if DMP criteria is met, if it is we can avoid computing alpha
    # W_old -> s.dm.M_fv
    for dim in self.dims2:
        self.fv_Boundaries(self.dm.M_fv,dim)
    W_max = self.dm.M_fv.copy()
    W_min = self.dm.M_fv.copy()
    for dim in self.dims:
        W_max = np.maximum(compute_W_max(self.dm.M_fv,dim),W_max)
        W_min = np.minimum(compute_W_min(self.dm.M_fv,dim),W_min)
            
    W_max = self.crop(W_max,ngh=ngh)
    W_min = self.crop(W_min,ngh=ngh)
    
    if self.p > 0:
        W_min -= np.abs(W_min) * self.tolerance
        W_max += np.abs(W_max) * self.tolerance

    possible_trouble = np.where(W_new >= W_min, 0, 1)
    possible_trouble = np.where(W_new <= W_max, possible_trouble, 1)
       
    # Now check for smooth extrema and relax the criteria for such cases
    if np.any(possible_trouble) and self.p > 1:
        self.fill_active_region(W_new)
        alpha = W_new.copy()*0 + 1
        for dim in self.dims2:
            shift = self.dims2[dim]
            self.fv_Boundaries(self.dm.M_fv,dim)
            alpha_new = compute_smooth_extrema(self, self.dm.M_fv, dim)[self.crop_fv(None,None,shift,ngh)]
            alpha = np.where(alpha_new < alpha, alpha_new, alpha)

        possible_trouble *= np.where(alpha<1, 1, 0)

    self.dm.troubles[...] = np.amax(possible_trouble,axis=0)

    ###########################
    # PAD Check for physically admissible values
    ###########################
    if self.PAD:
        # For the density
        self.dm.troubles = np.where(
            W_new[self._d_, ...] >= self.min_rho, self.dm.troubles, 1
        )
        self.dm.troubles = np.where(
            W_new[self._d_, ...] <= self.max_rho, self.dm.troubles, 1
        )
        # For the pressure
        self.dm.troubles = np.where(
            W_new[self._p_, ...] >= self.min_P, self.dm.troubles, 1)

    #self.n_troubles += self.dm.troubles.sum()
    self.fill_active_region(self.dm.troubles)
    for dim in self.dims2:
        self.fv_Boundaries(self.dm.M_fv,dim)
    trouble = self.dm.M_fv[0]
    self.dm.theta[0][...] = trouble
    theta = self.dm.theta[0]

    if self.blending:
        apply_blending(self,trouble,theta)

    for dim in self.dims2:
        idim = self.dims2[dim]
        affected_faces = self.dm.__getattribute__(f"affected_faces_{dim}")
        affected_faces[...] = 0
        affected_faces[...] = np.maximum(theta[self.crop_fv(ngh-1,-ngh,idim,ngh)],theta[self.crop_fv(ngh,-(ngh-1),idim,ngh)])
        
        #if self.BC[dim] == "periodic":
        #    affected = np.maximum(affected_faces[indices(0,idim)],affected_faces[indices(-1,idim)])
        #    affected_faces[indices(0,idim)] = affected_faces[indices(-1,idim)] = affected


def compute_W_ex(W, dim, f):
    W_f = W.copy()
    # W_f(i) = f(W(i-1),W(i),W(i+1))
    # First comparing W(i) and W(i+1)
    W_f[cut(None,-1,dim)] = f(  W[cut(None,-1,dim)],W[cut(1, None,dim)])
    # Now comparing W_f(i) and W_(i-1)
    W_f[cut( 1,None,dim)] = f(W_f[cut( 1,None,dim)],W[cut(None,-1,dim)])
    return W_f

def compute_W_max(W, dim):
    return compute_W_ex(W, dim, np.maximum)

def compute_W_min(W, dim):
    return compute_W_ex(W, dim, np.minimum)

def first_order_derivative(U, h, dim):
    dU = (U[cut(2,None,dim)] - U[cut(None,-2,dim)])/(h[cut(2,None,dim)] - h[cut(None,-2,dim)])
    return dU

def compute_min(A, Amin, dim):
    Amin[cut(None,-1,dim)] = np.minimum(A[cut(None,-1,dim)],   A[cut(1,None,dim)])
    Amin[cut( 1,None,dim)] = np.minimum(A[cut(None,-1,dim)],Amin[cut(1,None,dim)])

def compute_smooth_extrema(self, U, dim):
    eps = 0
    idim = self.dims2[dim]
    centers = self.centers[dim][self.shape(idim)]
    # First derivative dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    dU  = first_order_derivative( U, centers, idim)
    # Second derivative d2Udx2(i) = [dU(i+1)-dU(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    d2U = first_order_derivative(dU, centers[cut(1,-1,idim)], idim)
    dv = 0.5 * self.h_fp[dim][cut(2,-2,idim)] * d2U
    # vL = dU(i-1)-dU(i)
    vL = dU[cut(None,-2,idim)] - dU[cut(1,-1,idim)]
    # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
    alphaL = (
        -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0)) / dv
    )
    alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
    alphaL = np.where(alphaL < 1, alphaL, 1)
    # vR = dU(i+1)-dU(i)
    vR = dU[cut( 2,None,idim)] - dU[cut(1,-1,idim)]
    # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
    alphaR = np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0)) / dv
    alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
    alphaR = np.where(alphaR < 1, alphaR, 1)
    alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
    compute_min(alphaR, alphaL, idim)
    return alphaL

def apply_blending(self,trouble,theta):
    #First neighbors
    for idim in self.dims:
        theta[cut(None,-1,idim)] = np.maximum(.75*trouble[cut( 1,None,idim)],theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.75*trouble[cut(None,-1,idim)],theta[cut( 1,None,idim)])
          
    if self.ndim==2:
        #Second neighbors
        theta[:-1,:-1] = np.maximum(.5*trouble[1: ,1: ],theta[:-1,:-1])
        theta[:-1,1: ] = np.maximum(.5*trouble[1: ,:-1],theta[:-1,1: ])
        theta[1: ,:-1] = np.maximum(.5*trouble[:-1,1: ],theta[1: ,:-1])
        theta[1: ,1: ] = np.maximum(.5*trouble[:-1,:-1],theta[1: ,1: ])

    elif self.ndim==3:
        #Second neighbors
        a = slice(None,-1)
        b = slice( 1,None)
        cuts = [(a,a),(a,b),(b,a),(b,b)]
        for i in range(len(cuts)):
            for idim in self.dims:
                shape1 = tuple(np.roll(np.array((slice(None),)+cuts[ i]),-idim))
                shape2 = tuple(np.roll(np.array((slice(None),)+cuts[::-1][-i]),-idim))
                theta[shape1] = np.maximum(.5*trouble[shape2],theta[shape1])
        #Third neighbors
        cuts = [(x,y,z) for x in (a,b) for y in (a,b) for z in (a,b)]
        for i in range(len(cuts)):
            self.dm.theta[cuts[i]] = np.maximum(.375*trouble[cuts[::-1][i]],self.dm.theta[cuts[i]])
        
    for idim in self.dims:
        theta[cut(None,-1,idim)] = np.maximum(.25*(theta[cut( 1,None,idim)]>0),theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.25*(theta[cut(None,-1,idim)]>0),theta[cut( 1,None,idim)])
     
        
