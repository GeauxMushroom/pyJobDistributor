import numpy as np
import subprocess
from numpy import sum,arange,sin,cos,exp,pi,zeros,sign
from numba import jit
from scipy.special import sph_jn
from scipy.integrate import quad
import scipy
import os

par = np.genfromtxt("parameters.txt")
iters = par[5] * par[6] * par[7] * par[8]
beta = par[0]

N_TAU = 400
N_OMEGA = 400
N_k = 32
omega = pi / beta * ( 2 * arange(-N_OMEGA, N_OMEGA) + 1)
tau = beta / N_TAU * (arange(-N_TAU, N_TAU) + 0.5)
tau2 = beta / (N_TAU * 10) * (arange(-N_TAU * 10, N_TAU *10) + 0.5)
t = par[3]
Loop_lim = 2
Epsilon = 0.01
mu = par[1]
U = par[2]
D = par[4]
Mix = 0.5


@jit
def DIFT(gio):
    # transform from iw to tau
    g = zeros(N_TAU * 2, np.complex_)
    for i in range(0, 2 * N_TAU):
        g[i] = 1.0 / beta * sum(gio * exp(-1j * omega * tau[i]))
    return g


@jit
def DFT(gt):
    # transform from tau to iw
    g = zeros(N_OMEGA * 2, np.complex_) 
    dt = beta /( 2.0 * N_TAU)
    for i in range(0, 2 * N_OMEGA):
        g[i] = dt * sum(gt * exp(1j * omega[i] * tau))
    return g


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


def SemiCircularDOS(w,D):
    #semicircular DOS
    if abs(w)<D:
        return(2 * np.sqrt(1-(w/D)**2)/pi/D)
    else:
        return 0


def SemiCircular(z):
    # integrate( A(w) / (z-w) dw), A(w) is Theta(D-|w|)*2*sqrt(1-(w/D)**2)/pi/D
    return complex_quadrature(lambda x:SemiCircularDOS(x,D)/(z-x),-D,D)[0]


def init_g_SC():
    #Hilbert transform of semicircular DOS
    g = zeros(N_OMEGA * 2, np.complex_)
    for i in range(0, 2 * N_OMEGA):
        g[i] = SemiCircular(1j * omega[i])
    return g 


def init_g_Wilson():
    #Hilbert transform of flat DOS with a cutoff
    #g(z)=integrate( A(w) / (z-w) dw), A(w) is Theta(D-|w|)
    g = 1.0/(2*D) * (np.log(1j * omega + D) - np.log(1j * omega -D))
    return g


def Get_g_legendre():
    ll = np.genfromtxt('Job_output_0/green_legendre_0.txt').transpose()[0]
    NL = len(ll)
    tnl = np.zeros([N_OMEGA,NL], np.complex_)
    l= np.arange(NL)
    # recover g(iw) from the legendre polynomial measurements, see 
    for i in range(N_OMEGA):
        tnl[i]=(-1)**i*(1j)**(l+1)*np.sqrt(2*l+1)*sph_jn(NL-1,(2*i+1)*pi/2)[0]
    g = tnl.dot(ll)
    return(np.hstack([-g[::-1],g]))


def Save_gs(gs_tau):
    # write hybridization function to a txt file so the impurity solver can read it
    np.savetxt('GS.txt', np.array([gs_tau[N_TAU:].real,gs_tau[N_TAU:].imag]).transpose(), fmt='%1.20f\t%1.20f')


def Load_gt():
    # load g(tau) from the txt output of the solver
    gg = np.genfromtxt("Job_output_0/green_tau_0.txt").transpose()[0] / (beta * beta /N_TAU * iters)
    g = zeros(N_TAU *2) * (1 + 1j)
    g[:N_TAU] = -1.0 * gg
    g[N_TAU:] = gg
    return g



def init_d_iw():
    delta_iw = zeros(N_OMEGA * 2, np.complex_)
    sum_ek = 0
    # gc = 1.0 / (1j * omega + mu)
    for kx in range(-N_k, N_k):
        for ky in range(-N_k, N_k):
            for kz in range(-N_k, N_k):
                epsilon_k = -2.0 * t * (cos(kx * pi / N_k) + cos(ky * pi / N_k) + cos(kz * pi / N_k))
                # delta_iw = delta_iw + (epsilon_k ** 2) * gc
                delta_iw += 1 / (1j* omega - epsilon_k) 
                sum_ek += epsilon_k **2
    sum_ek = sum_ek / (2*N_k) **3
    delta_iw = sum_ek * delta_iw / (2 * N_k) ** 3
    return delta_iw, sum_ek


def DIFT_D(delta_iw, C):
    dt = zeros(N_TAU * 2, np.complex_)
    for i in range(0, 2 * N_TAU):
        dt[i] = 1.0 / beta * sum((delta_iw - C / (1j*omega)) * exp(-1j * omega * tau[i])) - C*0.5*sign(tau[i])
    return dt



def Get_G_iw(F):
    G_iw = zeros(2 * N_OMEGA, np.complex_)
    for kx in range(-N_k, N_k ):
        for ky in range(-N_k, N_k ):
            for kz in range(-N_k, N_k):
                epsilon_k = -2.0 * t * (cos(kx * pi / N_k) + cos(ky * pi / N_k) + cos(kz * pi / N_k))
                G_iw = G_iw + 1 / ( 1 / F - epsilon_k)
    G_iw = G_iw / (2 * N_k) ** 3
    return G_iw

def Get_gtt():
    # calculate the correction from tail
    g = zeros(N_TAU * 2, np.complex_)
    for i in range(0, 2* N_TAU):
        g[i] = -0.5 * sign(tau[i]) + 2.0 / beta * sum( sin(omega[N_OMEGA:] * tau[i]) / omega[N_OMEGA:])
    return g


def DMFT_INIT():
#    delta_iw, ek = init_d_iw()
    delta_iw = V2 * init_g_Wilson()
    np.save('d_iw_0.npy',delta_iw)
#    delta_t = DIFT_D(delta_iw)  # fourier transform to tau
    delta_t = DIFT_D(delta_iw, V2)  # fourier transform to tau
#    delta_t_tail = V2 * Get_gtt()
#    delta_t += delta_t_tail
    Save_gs(delta_t) 
    subprocess.call('./ctqmc_dmft') # call the impurity solver
    g_iw = Get_g_legendre()
    np.save('g_iw_0.npy',g_iw)
    return


    
def callSolver(hostName):
    currPath = os.path.abspath(os.getcwd())
    #print currPath
    CPU_COMMAND = 'cd '+currPath+' && ./ctqmc_dmft'
    #print CPU_COMMAND
    #print hostName
    if hostName == 'LOCAL':
        subprocess.call(CPU_COMMAND,shell='True')
    else:
        ssh = subprocess.Popen(["ssh", "%s" % HOST, CPU_COMMAND],
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE
                               )
        ssh.wait()
        result = ssh.stdout.readlines()
        error = ssh.stderr.readlines()
    return 


def callSolverMic(hostName,micId):
    currPath = os.path.abspath(os.getcwd())
    #print currPath
    MIC_COMMAND = 'ssh mic%d '%int(micId)+'"cd '+currPath+' && ./ctqmc_dmft_mic"'
    #print CPU_COMMAND
    print hostName
    if hostName == 'LOCAL':
        subprocess.call(MIC_COMMAND,shell='True')
    else:
        ssh = subprocess.Popen(["ssh", "%s" % HOST, MIC_COMMAND],
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE
                               )
        ssh.wait()
        result = ssh.stdout.readlines()
        error = ssh.stderr.readlines()
    return


def DMFT_LOOP():
#    delta_iw = V2 * init_g_Wilson()
#    delta_iw, ek = init_d_iw()
    delta_iw = init_g_SC()
#    delta_t = DIFT_D(delta_iw, ek)  # fourier transform to tau
    delta_t= DIFT_D(delta_iw, 1.0)
    Save_gs(delta_t) 
    hostNameFile=open('hostName.txt','r')
    hostName=hostNameFile.read()
    hostNameFile.close()
#    subprocess.call('./ctqmc_dmft') # call the impurity solver
    callSolver(hostName)
    g_iw = Get_g_legendre()
    loop = 0
    while(loop < Loop_lim): # main loop
        loop += 1
        print "loop %d" % loop
        #F = 1.0 / (1.0 / g_iw + delta_iw)
        #G_iw = Get_G_iw(F) 
        #delta_iw_new = 1 / F - 1/ G_iw 
        delta_iw_new = t**2 * g_iw;
        delta_iw = Mix * delta_iw_new + (1.0 - Mix) * delta_iw
        #ek = Mix + (1.0- Mix) *ek
        c1 = Mix * t**2 + (1.0-Mix)*1
        np.save('d_iw_%d.npy' % loop, delta_iw)
        delta_t = DIFT_D(delta_iw, c1)  # fourier transform to tau
        #delta_t += delta_t_tail
        Save_gs(delta_t) 
#        subprocess.call('./ctqmc_dmft') # call the impurity solver
        callSolver(hostName)
        g_iw_new = Get_g_legendre()
        np.save('g_iw_%d.npy' % loop, g_iw)
        g_iw_diff = g_iw_new - g_iw # calculate the difference        
        err = np.max(abs(g_iw_diff / g_iw)[N_OMEGA-20:N_OMEGA+20]) 
        print err
        if err < Epsilon: # determine converge or not
            print "Converged!"
            return [1,g_iw_new]
        g_iw =  g_iw_new
        
    print("Not converging!")
    return [0,g_iw]





def DMFT_LOOP_MIC():
#    delta_iw = V2 * init_g_Wilson()
#    delta_iw, ek = init_d_iw()
    delta_iw = init_g_SC()
#    delta_t = DIFT_D(delta_iw, ek)  # fourier transform to tau
    delta_t= DIFT_D(delta_iw, 1.0)
    Save_gs(delta_t) 
    hostNameFile=open('hostName.txt','r')
    hostName, micId = hostNameFile.readlines()
    hostNameFile.close()
    hostName = hostname.strip()
    micId = int(micId)
#    subprocess.call('./ctqmc_dmft') # call the impurity solver
    callSolverMic(hostName,micId)
    g_iw = Get_g_legendre()
    loop = 0
    while(loop < Loop_lim): # main loop
        loop += 1
        print "loop %d" % loop
        #F = 1.0 / (1.0 / g_iw + delta_iw)
        #G_iw = Get_G_iw(F) 
        #delta_iw_new = 1 / F - 1/ G_iw 
        delta_iw_new = t**2 * g_iw;
        delta_iw = Mix * delta_iw_new + (1.0 - Mix) * delta_iw
        #ek = Mix + (1.0- Mix) *ek
        c1 = Mix * t**2 + (1.0-Mix)*1
        np.save('d_iw_%d.npy' % loop, delta_iw)
        delta_t = DIFT_D(delta_iw, c1)  # fourier transform to tau
        #delta_t += delta_t_tail
        Save_gs(delta_t) 
#        subprocess.call('./ctqmc_dmft') # call the impurity solver
        callSolverMic(hostName,micId)
        g_iw_new = Get_g_legendre()
        np.save('g_iw_%d.npy' % loop, g_iw)
        g_iw_diff = g_iw_new - g_iw # calculate the difference        
        err = np.max(abs(g_iw_diff / g_iw)[N_OMEGA-20:N_OMEGA+20]) 
        print err
        if err < Epsilon: # determine converge or not
            print "Converged!"
            return [1,g_iw_new]
        g_iw =  g_iw_new
        
    print("Not converging!")
    return [0,g_iw]

