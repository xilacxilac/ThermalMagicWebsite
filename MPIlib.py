#==============================================================================
#
#   Python Library for MPI Simulation App
#
#==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression
#==============================================================================
#
#   Define constants
#
#==============================================================================
mu0 = 1.25663706212 * 10**(-6)      # in H/m, Permeability Constant
kB = 1.380649 * 10 **(-23)          # in J/K, Boltzmann Constant
#==============================================================================
#
#   The Langevin Function or its Derivative
#
#   - y is the argument
#
#   - der denotes whether we want to calculate the derivative (der=1) or not (der=0)
#==============================================================================
def MvH(y,der):
    if type(y)==float:
        ny = np.abs(y)
    else:
       ny = np.linalg.norm(y)
    if der == 0:
         L = np.cosh(ny)/np.sinh(ny) - 1/ny
         if np.isnan(L):
             if ny>1:
                 L = 1
             else:
                 L = 0
    else:
         L = -1/(np.sinh(ny)*np.sinh(ny)) + 1/ny**2
         if np.isnan(L) or ny<0.000001:
             L = 1/3-(ny**2)*3/45+(ny**4)*5/945-(ny**6)*7/4725 
    return L    
#==============================================================================
#
#   The FFP trajectory and its velocity
#
#   - Hfun [T] is the applied excitation field, with two components, 
#     
#     the field itself and its time-derivative
#
#   - Ginv [m/T] is the inverse of the gradient field matrix
#
#==============================================================================  
def FFPtraj(t,Ginv,Hfun,center=0):
    r = np.matmul(Ginv,Hfun(t)[0])
    rdot = np.matmul(Ginv,Hfun(t)[1])
    if center != 0:
        r=r+center
    return r, rdot
#==============================================================================
#
#   The signal
#
#   - Rho consists of concentration, Temperature, and particle diameter
#    
#==============================================================================
def Signal(t,Rho,Rhodim,FFPtrajfun,MsTfun,MvHfun,MvHfunDer,Bsfun,G):
    rtmp = FFPtrajfun(t)
    r = np.transpose(rtmp[0])
    rdot = np.transpose(rtmp[1])
    S = 0
    dim = Rho.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                Rhotmp = Rho[i,j,k,0]
                if Rhotmp > 0:
                    Temptmp = Rho[i,j,k,1] 
                    dtmp = Rho[i,j,k,2]
                    xtmp = [Rhodim[0][i],Rhodim[1][j],Rhodim[2][k]]
                    #
                    #   Determine the coil sensitivity, saturation magnetization,
                    #
                    #   and magnetic moment
                    #
                    Bstmp = Bsfun(xtmp)
                    Mstmp = MsTfun(Temptmp)
                    magtmp = (Mstmp*np.pi*dtmp**3)/6
                    xi = mu0*magtmp/(Temptmp*kB)
                    Grdot = G@rdot
                    GrdotT = np.transpose(Grdot)
                                        
                    Grx = G @ (r-xtmp)
                    nGrx = np.linalg.norm(Grx)
                    Ltmp = MvHfun(xi*Grx)/(xi*nGrx)
                    Ldot = MvHfunDer(xi*Grx)
                    tmp = GrdotT@Grx
                    tmp = xi*tmp*Grx
                    tmp = tmp/(nGrx**2)
                    A1 = Ldot*tmp
                    A2 = Ltmp*(xi*Grdot-tmp)
                    PSF = A1 + A2
                    S = S + Bstmp*Rhotmp*PSF*-magtmp
    return S
#==============================================================================
#
#   Routine to Collect the Measurements for a Specific Cell,
#
#   not Weighted for now
#
#==============================================================================
def CellCollect(x,y,z,dx,dy,dz,r,v,s,direction):
    #
    # s Signal Data, v Tangent Data
    #
    # Parameter Direction Controls from "Which Side we are Coming" in 1D Setting
    #
    S = []; V = []; indx = []
    if dy == 0 and dz == 0: # 1D Case
        m = len(r)
        for i in range(m):
            if direction == 0:
                if r[i]>x+dx or r[i]<x-dx:
                    i
                else:
                    indx.append(i)
                    V.append(v[i])
                    S.append(s[i])
            else:
                if r[i]>x+dx or r[i]<x-dx or np.sign(v[i])!=direction:
                    i
                else:
                    indx.append(i)
                    V.append(v[i])
                    S.append(s[i])
        V = np.reshape(V,(len(V),1))
        S = np.reshape(S,(len(S),1))
        V = np.transpose(V); S = np.transpose(S)    
    elif dz == 0: # 2D Case
        m,n = r.shape
        for i in range(m):
            if r[i,0]>x+dx/2 or r[i,0]<x-dx/2 or r[i,1]>y+dy/2 or r[i,1]<y-dy/2:
                i
            else:
                indx.append(i)
                V.append(v[i,:])
                S.append(s[i,:])
        V = np.reshape(V,(len(V),2))
        S = np.reshape(S,(len(S),2))
        V = np.transpose(V); S = np.transpose(S)    
    else:
        m,n = r.shape
        for i in range(m):
            if r[i,0]>x+dx or r[i,0]<x-dx or r[i,1]>y+dy or r[i,1]<y-dy or r[i,2]>z+dz or r[i,2]<z-dz:
                i
            else:
                indx.append(i)
                V.append(v[i,:])
                S.append(s[i,:])
        V = np.reshape(V,(len(V),3))
        S = np.reshape(S,(len(S),3))
        V = np.transpose(V); S = np.transpose(S)  
    return V, S, indx
#==============================================================================
#
#   Routine to Estimate the Trace Data used for Illustration Purposes
#
#==============================================================================
def TraceEstimate(x,y,z,r,rdot,st,BsensInv,mag,direction):
    model = LinearRegression()
    if np.all(y == 0) and np.all(z == 0): # 1D Case
        dx = (x[2]-x[1]); yall = []
        nx = len(x)
        uall = np.zeros(nx); xall = np.zeros(nx)
        k = 0; rdotsort = np.zeros(nx)
        for i in range(nx):
            V,S,indx = CellCollect(x[i],0,0,dx,0,0,r,rdot,st,direction)
            Snorm = BsensInv*S/(mu0*mag)
            if V.shape[1]>2: # Dimension of Space
                Q,R = np.linalg.qr(np.transpose(V))
                RinvT = np.linalg.inv(np.transpose(R))
                Atmp = np.matmul(Snorm,np.matmul(Q,RinvT))
                uall[k] = np.trace(Atmp)
                xall[k] = x[i]
                rdotsort[k] = direction*np.max(direction*(V[0,:]))
            k = k + 1
        #
        # Relate uall to Original Grid, i.e., x
        #
        usort=uall
    elif np.all(z == 0): # 2D case
        dx = np.abs(x[2]-x[1]); dy = np.abs(y[2]-y[1])
        nx = len(x); ny = len(y)
        uall = np.zeros(nx*ny); xall = np.zeros((nx*ny)); yall = np.zeros((nx*ny)); sall = np.zeros((nx*ny,2))
        k = 0
        rdotsort = 0
        for i in range(nx):
                for j in range(ny):
                    V,S,indx = CellCollect(x[i],y[j],0,dx,dy,0,r,rdot,st,direction)
                    Snorm = -np.matmul(BsensInv,S)/(mu0*mag)
                    if V.shape[1]>2: # dimension of space
                        fit = model.fit(np.transpose(V),np.transpose(Snorm))
                        Atmp = fit.coef_
                        uall[k] = np.trace(Atmp)
                        xall[k] = x[i]
                        yall[k] = y[j]
                        sall[k,0] = np.mean(((S[0,:])))
                        sall[k,1] = np.mean(((S[1,:])))
                    else:
                        print('Empty Cell')
                    k = k + 1                   
        #
        # Relate uall to original grid, i.e., x and y
        #
        usort=np.zeros((nx,ny))
        k = 0; f = 0
        for i in range(nx):
            for j in range(ny):
                a = xall==x[i]
                b = yall==y[j] 
                c = a & b
                d = np.where(c==True)
                try:
                    usort[i,j] = uall[d]
                    f = f + 1
                except:
                    usort[i,j] = 0
                    print("Error Finding Cell")
                k = k + 1
        print('{0} of {1} Cells filled'.format(f,nx*ny))
    else:
        print('nothing')
    return xall, yall, uall, usort, rdotsort, sall