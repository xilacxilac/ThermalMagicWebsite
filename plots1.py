import MPIlib as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from functools import partial
from io import BytesIO
import base64
import math

def signal_generation(g00, f0, f1, dia, Ms, Bs, lam_exp):
    lam = 2*1*10**(lam_exp) # the regularization parameter

    #==============================================================================
    #
    #   Define the gradient field, G,  and excitation field H, and the particle diameter
    #
    #   Determine the FOV
    #
    G = np.eye(3)
    G[0,0] = g00; # gradient in x direction [T/m]
    G[1,1] = -G[0,0]/2 # gradient in y direction [T/m]
    G[2,2] = -G[0,0]/2 # gradient in z direction [T/m] (not relevant)
    G = G/mp.mu0
    Ginv = np.linalg.inv(G)
    #
    #
    f = np.ones((3,1));f[0] = f0;f[1] = f1;f[2] = 0
    xmin = -0.005;xmax = -xmin
    ymin = -0.005;ymax = -ymin

    B = np.ones((3,1))
    B[0] = xmin * G[0,0] * mp.mu0
    B[1] = ymin * G[1,1] * mp.mu0
    B[2] = 0; # [T]
    H = B/mp.mu0

    zmin = 0;  # this is a 2D example, hence no extension in z direction
    FOV = [xmin,xmax,ymin,ymax]
    #==============================================================================
    #==============================================================================
    #
    #   The 3 following functions are now also available in the MPIlib
    #
    #
    #   Excitation field as a function of time
    #
    def Hfun(t,H,f):
        n = len(H)
        r = np.zeros((n))
        rdot = np.zeros((n))
        for i in range(n):
            r[i] = H[i]*np.sin(2*np.pi*f[i]*t)
            rdot[i] = H[i]*2*np.pi*f[i]*np.cos(2*np.pi*f[i]*t)
        return r, rdot
    #==============================================================================
    #
    #   Saturation magnetization as a function of temperature, so far a constant
    #
    def MsTfun(T):
        return Ms
    #==============================================================================
    #
    #   Coil sensitivity as a function of space, so far a constant
    #
    def Bsfun(x):
        return Bs
    #
    #
    #==============================================================================
    #==============================================================================
    #
    #   Define excitation field and FFP trajectory functions based on the parameters
    #   
    #   defined above, we also need to define the MvH function and its derivative
    #
    Happ = partial(Hfun,H=H,f=f)
    FFPtrajfun = partial(mp.FFPtraj,Ginv=Ginv,Hfun=Happ)
    MvHfun = partial(mp.MvH,der=0)
    MvHfunDer = partial(mp.MvH,der=1)
    #==============================================================================
    #
    #   Define time window, sampling frequency, and number of pixels in x and y direction    
    #
    fsample = 4*10**5
    nx = 9
    ny = 9
    nz = 1
    tmax=0.004
    nt = int(tmax*fsample) #10000
    t = np.linspace(0,tmax,nt+1)
    #
    #   Important: we delete the first time step, since t0 = 0 leads to problems
    #
    t = t[1:]
    #==============================================================================
    #
    #   Create the 3D (2D) phantom and plot it
    #
    Rho = np.zeros((nx,ny,1,4))
    Rhodim = [np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),[zmin]]
    dx = Rhodim[0][1]-Rhodim[0][0]; dy = Rhodim[1][1]-Rhodim[1][0]; dz = dx
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                Rho[i,j,k,1] = 295
                Rho[i,j,k,2] = dia*10**(-9)
                xtmp = np.asarray([Rhodim[0][i],Rhodim[1][j]])
                dist = np.linalg.norm(xtmp)
                if dist<0.003 and dist>0.001:
                    Rho[i,j,k,0] = 3.168*10**20*dx*dy*dz
    
    plt.imshow(Rho[:,:,0,0],aspect=1,extent=FOV)
    plt.xlabel("x [m]") 
    plt.ylabel("y [m]")
    plt.colorbar()
    buf1 = BytesIO()
    plt.savefig(buf1, format="png")

    plt.clf()

    #==============================================================================
    #
    #   Calculate the signal based on the above defined parameters
    #
    S = np.zeros((nt,3))
    FFP = np.zeros((nt,3))
    FFPdot = np.zeros((nt,3))
    for i in range(nt):
        FFP[i,:] = FFPtrajfun(t[i])[0]
        FFPdot[i,:] = FFPtrajfun(t[i])[1]
        S[i,:] = mp.Signal(t[i],Rho,Rhodim,FFPtrajfun,MsTfun,MvHfun,MvHfunDer,Bsfun,G)
    #==============================================================================
    #
    #   Plot the picked-up signals at the two different receive coils
    #
    plt.suptitle('Signal picked-up by receive coils')
    plt.subplot(2,1,1)
    plt.plot(t,S[:,0]);plt.xlabel('t [s]');plt.ylabel('$S_x$ [V]')
    plt.subplot(2,1,2)
    plt.plot(t,S[:,1]);plt.xlabel('t [s]');plt.ylabel('$S_y$ [V]')
    buf2 = BytesIO()
    plt.savefig(buf2, format="png")

    plt.clf()

    #==============================================================================
    #
    #   Plot the FFP trajectory
    #
    plt.title('FFP Trajectory')
    plt.scatter(FFP[:,0],FFP[:,1],s=2)
    plt.xlim([xmin,xmax]);plt.ylim([ymin,ymax])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    buf3 = BytesIO()
    plt.savefig(buf3, format="png")

    plt.clf()

    #==============================================================================
    #
    #   Plot the picked-up signals in space domain using the FFP trajectory to get raw image
    #
    #
    plt.title('Sum of signals picked-up by receive coil')
    plt.scatter(FFP[:,0],FFP[:,1],c=np.abs(S[:,0]) + np.abs(S[:,1]), s=30)
    plt.ylim([xmin,xmax]);plt.xlim([ymin,ymax]);plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    buf4 = BytesIO()
    plt.savefig(buf4, format="png")
    plt.clf()

    # room temperature
    T = 293
    y = mp.mu0 * Ms * np.pi * ((dia * (10 ** -9)) ** 3) / (6 * T * mp.kB)
    rbound = max(np.abs(B[0]), np.abs(B[1])) / mp.mu0
    lbound = -rbound
    Hs = np.linspace(lbound, rbound, 100)
    
    x = np.array([mp.MvH(y * H, 0) * H / np.abs(H) for H in Hs])

    plt.title('MvH')
    plt.plot(Hs, x)
    plt.xlabel('H')
    plt.ylabel('Ms')

    buf5 = BytesIO()
    plt.savefig(buf5, format="png")
    plt.clf()

    #==============================================================================================
    #
    #  The gridding algorithm requires some experimental parameters, hence we need to explicitly 
    #
    #  define them again
    #
    Bs0 = Bsfun(0)
    Ms = MsTfun(T)
    dia_nm = dia * (10 ** -9) # the diameter in m, note that dia is the diameter in nm
    mag = (Ms * np.pi * (dia_nm ** 3)) / 6
    Hsat = T * mp.kB / (mp.mu0 * mag)
    x = Rhodim[0]
    y = Rhodim[1] 
    extent=(Rhodim[0].min(),Rhodim[0].max(),Rhodim[1].max(),Rhodim[1].min()) # this is defined to make plotting easier later

    #==========================================================================================
    #
    #  Do the gridding and plot the result
    #
    xall2D, yall2D, uall2D, usort2D, rdotsort, sall = mp.TraceEstimate(x,y,0,FFP[:,0:2],FFPdot[:,0:2],S[:,0:2]@G[0:2,0:2],np.eye(2)/Bs0,mag,0)
    plt.imshow(usort2D,extent=extent,cmap='jet');plt.xlabel('y [m]');plt.ylabel('x [m]');plt.title('Gridded Raw Signal')

    buf6 = BytesIO()
    plt.savefig(buf6, format="png")
    plt.clf()

    #===============================================================================================================================
    #
    #  Populate the Toeplitz matrix, little thing to note: the Toeplitz matrix now "lives" entirely in space domain
    #
    T = np.zeros((nx*ny,nx*ny)); xdiff = np.zeros(3); tmp = np.zeros(nt); kindx1 = 0
    for i in range(nx):
        for j in range(ny):
            xtmp = np.zeros(3)
            xtmp[0] = x[i]; xtmp[1] = y[j]
            kindx2 = 0
            for i1 in range(nx):
                for j1 in range(ny):
                    xdiff[0] = x[i1]; xdiff[1] = y[j1]
                    Htmp = np.matmul(G,xtmp-xdiff)
                    T[kindx1,kindx2] = MvHfunDer(Htmp/Hsat)
                    kindx2 = kindx2 + 1
            kindx1 = kindx1 + 1

    #==========================================================================================
    #
    #  Here we plot a specific portion of the Toeplitz matrix to visualize the point-spread approach of it
    #
    #  This plot basically shows how a delta sample in the center of the FOV would be blurred if imaged with
    #
    #  this specific experimental setup.
    #
    M = np.reshape(T[int(nx*ny/2),:],(nx,ny))
    plt.imshow(M,extent=extent)
    plt.title('Point Spread Function')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    buf7 = BytesIO()
    plt.savefig(buf7, format="png")
    plt.clf()

    #==========================================================================================
    #
    #  Do the reconstruction using the Python implementation of ridge regression
    #
    model = linear_model.Ridge(alpha=lam)
    fit = model.fit(T,uall2D)
    Rho_hat = fit.coef_

    #==========================================================================================
    #
    #  The reconstruction result is a vector that we need to reshape to get a 2D representation
    #
    Rho_hat = np.reshape(Rho_hat,(nx,ny))
    plt.imshow(Rho_hat,extent=extent,cmap='jet');plt.title('Reconstruction')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    buf8 = BytesIO()
    plt.savefig(buf8, format="png")
    plt.clf()

    mid = math.floor(Rho_hat.shape[0] / 2)
    plt.title('Reconstruction along x-axis')
    plt.plot(x, Rho[:,mid,0,0])
    plt.plot(x, Rho_hat[:,mid])
    plt.xlabel('x [m]')
    plt.ylabel('Number of Particles')

    buf9 = BytesIO()
    plt.savefig(buf9, format="png")
    plt.clf()

    plt.title('Reconstruction along y-axis')
    plt.plot(y, Rho[mid,:,0,0])
    plt.plot(y, Rho_hat[mid,:])
    plt.xlabel('y [m]')
    plt.ylabel('Number of Particles')

    buf10 = BytesIO()
    plt.savefig(buf10, format="png")
    plt.clf()

    data1 = base64.b64encode(buf1.getbuffer()).decode("ascii")
    data2 = base64.b64encode(buf2.getbuffer()).decode("ascii")
    data3 = base64.b64encode(buf3.getbuffer()).decode("ascii")
    data4 = base64.b64encode(buf4.getbuffer()).decode("ascii")
    data5 = base64.b64encode(buf5.getbuffer()).decode("ascii")
    data6 = base64.b64encode(buf6.getbuffer()).decode("ascii")
    data7 = base64.b64encode(buf7.getbuffer()).decode("ascii")
    data8 = base64.b64encode(buf8.getbuffer()).decode("ascii")
    data9 = base64.b64encode(buf9.getbuffer()).decode("ascii")
    data10 = base64.b64encode(buf10.getbuffer()).decode("ascii")
    return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10

def lambda_dependent_graphs(g00, f0, f1, dia, Ms, Bs, lam_exp):
    lam = 2*1*10**(lam_exp) # the regularization parameter

    #==============================================================================
    #
    #   Define the gradient field, G,  and excitation field H, and the particle diameter
    #
    #   Determine the FOV
    #
    G = np.eye(3)
    G[0,0] = g00; # gradient in x direction [T/m]
    G[1,1] = -G[0,0]/2 # gradient in y direction [T/m]
    G[2,2] = -G[0,0]/2 # gradient in z direction [T/m] (not relevant)
    G = G/mp.mu0
    Ginv = np.linalg.inv(G)
    #
    #
    f = np.ones((3,1));f[0] = f0;f[1] = f1;f[2] = 0
    xmin = -0.005;xmax = -xmin
    ymin = -0.005;ymax = -ymin

    B = np.ones((3,1))
    B[0] = xmin * G[0,0] * mp.mu0
    B[1] = ymin * G[1,1] * mp.mu0
    B[2] = 0; # [T]
    H = B/mp.mu0

    zmin = 0;  # this is a 2D example, hence no extension in z direction
    FOV = [xmin,xmax,ymin,ymax]
    #==============================================================================
    #==============================================================================
    #
    #   The 3 following functions are now also available in the MPIlib
    #
    #
    #   Excitation field as a function of time
    #
    def Hfun(t,H,f):
        n = len(H)
        r = np.zeros((n))
        rdot = np.zeros((n))
        for i in range(n):
            r[i] = H[i]*np.sin(2*np.pi*f[i]*t)
            rdot[i] = H[i]*2*np.pi*f[i]*np.cos(2*np.pi*f[i]*t)
        return r, rdot
    #==============================================================================
    #
    #   Saturation magnetization as a function of temperature, so far a constant
    #
    def MsTfun(T):
        return Ms
    #==============================================================================
    #
    #   Coil sensitivity as a function of space, so far a constant
    #
    def Bsfun(x):
        return Bs
    #
    #
    #==============================================================================
    #==============================================================================
    #
    #   Define excitation field and FFP trajectory functions based on the parameters
    #   
    #   defined above, we also need to define the MvH function and its derivative
    #
    Happ = partial(Hfun,H=H,f=f)
    FFPtrajfun = partial(mp.FFPtraj,Ginv=Ginv,Hfun=Happ)
    MvHfun = partial(mp.MvH,der=0)
    MvHfunDer = partial(mp.MvH,der=1)
    #==============================================================================
    #
    #   Define time window, sampling frequency, and number of pixels in x and y direction    
    #
    fsample = 4*10**5
    nx = 9
    ny = 9
    nz = 1
    tmax=0.004
    nt = int(tmax*fsample) #10000
    t = np.linspace(0,tmax,nt+1)
    #
    #   Important: we delete the first time step, since t0 = 0 leads to problems
    #
    t = t[1:]
    #==============================================================================
    #
    #   Create the 3D (2D) phantom and plot it
    #
    Rho = np.zeros((nx,ny,1,4))
    Rhodim = [np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),[zmin]]
    dx = Rhodim[0][1]-Rhodim[0][0]; dy = Rhodim[1][1]-Rhodim[1][0]; dz = dx
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                Rho[i,j,k,1] = 295
                Rho[i,j,k,2] = dia*10**(-9)
                xtmp = np.asarray([Rhodim[0][i],Rhodim[1][j]])
                dist = np.linalg.norm(xtmp)
                if dist<0.003 and dist>0.001:
                    Rho[i,j,k,0] = 3.168*10**20*dx*dy*dz
    #==============================================================================
    #
    #   Calculate the signal based on the above defined parameters
    #
    S = np.zeros((nt,3))
    FFP = np.zeros((nt,3))
    FFPdot = np.zeros((nt,3))
    for i in range(nt):
        FFP[i,:] = FFPtrajfun(t[i])[0]
        FFPdot[i,:] = FFPtrajfun(t[i])[1]
        S[i,:] = mp.Signal(t[i],Rho,Rhodim,FFPtrajfun,MsTfun,MvHfun,MvHfunDer,Bsfun,G)
    #==============================================================================================
    #
    #  The gridding algorithm requires some experimental parameters, hence we need to explicitly 
    #
    #  define them again
    #
    T = 293
    Bs0 = Bsfun(0)
    Ms = MsTfun(T)
    dia_nm = dia * (10 ** -9) # the diameter in m, note that dia is the diameter in nm
    mag = (Ms * np.pi * (dia_nm ** 3)) / 6
    Hsat = T * mp.kB / (mp.mu0 * mag)
    x = Rhodim[0]
    y = Rhodim[1] 
    extent=(Rhodim[0].min(),Rhodim[0].max(),Rhodim[1].max(),Rhodim[1].min()) # this is defined to make plotting easier later

    #==========================================================================================
    #
    #  Do the gridding and plot the result
    #
    xall2D, yall2D, uall2D, usort2D, rdotsort, sall = mp.TraceEstimate(x,y,0,FFP[:,0:2],FFPdot[:,0:2],S[:,0:2]@G[0:2,0:2],np.eye(2)/Bs0,mag,0)


    T = np.zeros((nx*ny,nx*ny)); xdiff = np.zeros(3); tmp = np.zeros(nt); kindx1 = 0
    for i in range(nx):
        for j in range(ny):
            xtmp = np.zeros(3)
            xtmp[0] = x[i]; xtmp[1] = y[j]
            kindx2 = 0
            for i1 in range(nx):
                for j1 in range(ny):
                    xdiff[0] = x[i1]; xdiff[1] = y[j1]
                    Htmp = np.matmul(G,xtmp-xdiff)
                    T[kindx1,kindx2] = MvHfunDer(Htmp/Hsat)
                    kindx2 = kindx2 + 1
            kindx1 = kindx1 + 1
    #==========================================================================================
    #
    #  Do the reconstruction using the Python implementation of ridge regression
    #
    model = linear_model.Ridge(alpha=lam)
    fit = model.fit(T,uall2D)
    Rho_hat = fit.coef_

    #==========================================================================================
    #
    #  The reconstruction result is a vector that we need to reshape to get a 2D representation
    #
    Rho_hat = np.reshape(Rho_hat,(nx,ny))
    plt.imshow(Rho_hat,extent=extent,cmap='jet');plt.title('Reconstruction')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    buf8 = BytesIO()
    plt.savefig(buf8, format="png")
    plt.clf()

    mid = math.floor(Rho_hat.shape[0] / 2)
    plt.title('Reconstruction along x-axis')
    plt.plot(x, Rho[:,mid,0,0])
    plt.plot(x, Rho_hat[:,mid])
    plt.xlabel('x [m]')
    plt.ylabel('Number of Particles')

    buf9 = BytesIO()
    plt.savefig(buf9, format="png")
    plt.clf()

    plt.title('Reconstruction along y-axis')
    plt.plot(y, Rho[mid,:,0,0])
    plt.plot(y, Rho_hat[mid,:])
    plt.xlabel('y [m]')
    plt.ylabel('Number of Particles')

    buf10 = BytesIO()
    plt.savefig(buf10, format="png")
    plt.clf()

    data8 = base64.b64encode(buf8.getbuffer()).decode("ascii")
    data9 = base64.b64encode(buf9.getbuffer()).decode("ascii")
    data10 = base64.b64encode(buf10.getbuffer()).decode("ascii")
    return data8, data9, data10