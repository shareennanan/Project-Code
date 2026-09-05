''' standalone implementation of convection-diffusion on a line. Gaussian pulse and sinusoidal initial conditions.'''


#============================= MAIN SOLVER ============================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
'''
N=400                     #no. of nodes
x=np.linspace(0,1,N+1)[:-1]  #gives us the nodes, excluding the last point because periodicity

h=1/N                      #distance between nodes. Note for N nodes there are N shape functions/elements

A = (1/h) * (2 * np.eye(N) + np.diag([-1]*(N-1), k=1) + np.diag([-1]*(N-1), k=-1))
A[0,-1]=-1/h
A[-1,0]=-1/h

M = (h/6) * (4 * np.eye(N) + np.diag([1]*(N-1),k=1) + np.diag([1]*(N-1),k=-1))
M[0,-1]=h/6
M[-1,0]=h/6

C = np.diag([1/2]*(N-1),k=1) + np.diag([-1/2]*(N-1),k=-1)
C[0,-1]=-1/2
C[-1,0]=1/2

T=1.5
Nt=150      #no. of time intervals
dt=T/Nt    #length of time steps
a = 0.005   #diffusion rate 

u=np.zeros((N,Nt+1))

gaussian_pulse = False     #Method small pulse width and small diffusion constant
gaussian_periodic = True  #Method summation of pulses
# select both as False for sin initial cond


if gaussian_pulse:
    sigma = 0.05 #small pulse width 0.05
    x0=0.25
    u[:,0]=np.exp(-(x-x0)**2/(2*sigma**2)) #gaussian pulse
    def u_exact(x,t):
        return (sigma / np.sqrt(sigma**2 + 2*a*t)) * np.exp(-(x - x0 - t)**2 / (2*(sigma**2 + 2*a*t)))
    filename= 'gaussian_normal.mp4'
elif gaussian_periodic:
    sigma = 0.02 #large pulse width 0.2
    x0=0.25
    u[:,0]=np.exp(-(x-x0)**2/(2*sigma**2)) #gaussian pulse
    def u_exact(x, t):
        result = np.zeros_like(x)
        for n in range(-5, 6):
            result += (sigma / np.sqrt(sigma**2 + 2*a*t)) * np.exp(-(x - x0 - t + n)**2 / (2*(sigma**2 + 2*a*t)))
        return result
    filename='gaussian_periodic.mp4'
else:
    u[:,0]=np.sin(2*x*np.pi) #sin wave
    def u_exact(x,t):
     return np.exp(-a * (2 * np.pi) ** 2 * t) * np.sin(2 * np.pi * (x - t)) #for the animation
    filename='sinwave.mp4'

theta = 0.5 #1 for implicit euler and 0.5 for crank-nicolson

for i in range(0,Nt):
        u[:,i+1]=np.linalg.solve(M+theta*dt*(a*A+C),(M-(1-theta)*dt*(a*A+C)) @ u[:,i]) 


plt.figure()
plt.ylim(0,1) 
for t in [0,0.25,0.65,0.9]:
    i=int(round(t/dt))
    plt.plot(x,u[:,i],label=f't={t}') 
    plt.legend()

plt.xlabel('x')
plt.ylabel('$u_h(x,t)$')
plt.title("Gaussian Pulse under Convection-Diffusion")
plt.savefig('cd_1D_pulse.pdf', bbox_inches='tight')
plt.show()


#============================ ANIMATION ==================================
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
writer = animation.FFMpegWriter(fps=30)

times=np.linspace(0,T,Nt+1)

uh_snaps=u.T
ue_snaps=np.zeros((Nt+1,N))
for i in range(Nt+1):
     ue_snaps[i,:]=u_exact(x,times[i])

fig, ax = plt.subplots(figsize=(7, 4))
ymin = float(min(uh_snaps.min(), ue_snaps.min()))
ymax = float(max(uh_snaps.max(), ue_snaps.max()))
line_h, = ax.plot([], [], label="u_h")
line_e, = ax.plot([], [], "--", label="exact")
ax.set_xlim(0, 1); ax.set_ylim(ymin - 0.05, ymax + 0.05)
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(loc="upper right")
title = ax.set_title("")
 

def update(i):
    line_h.set_data(x, uh_snaps[i])
    line_e.set_data(x, ue_snaps[i])
    title.set_text(f"t = {times[i]:.3f}")
    return line_h, line_e, title
 
anim = animation.FuncAnimation(fig, update, frames=len(times),interval=50, blit=False)

anim.save(filename, writer=writer, dpi=120)
print(f"saved {filename}")
    
'''

# ================================== ERROR ==============================

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

s={}
Ns=[200,400]
T=1
plt.figure()

for N in Ns:
    
    x=np.linspace(0,1,N+1)[:-1]  #gives us the nodes, excluding the last point because periodicity
    
    h=1/(N)                      #distance between nodes. Note for N nodes there are N shape functions/elements
    
    A = (1/h) * (2 * np.eye(N) + np.diag([-1]*(N-1), k=1) + np.diag([-1]*(N-1), k=-1))
    A[0,-1]=-1/h
    A[-1,0]=-1/h
    
    M = (h/6) * (4 * np.eye(N) + np.diag([1]*(N-1),k=1) + np.diag([1]*(N-1),k=-1))
    M[0,-1]=h/6
    M[-1,0]=h/6
    
    C = np.diag([1/2]*(N-1),k=1) + np.diag([-1/2]*(N-1),k=-1)
    C[0,-1]=-1/2
    C[-1,0]=1/2
    
    dts=[]
    errors=[]
    
    for n in range(1,13): #halves the time step 
        Nt=2**n
        t=1 #time at which u_exact is calculated
        dt=T/Nt     #time step
        u=np.zeros((N, Nt+1))
        u[:,0]=np.sin(2*x*np.pi)
        a =0.02             #diffusion rate  
        theta = 0.5 #1 for implicit euler and 0.5 for crank-nicolson  
        for i in range(0,Nt):
            u[:,i+1]=np.linalg.solve(M+theta*dt*(a*A+C),(M-(1-theta)*dt*(a*A+C)) @ u[:,i]) 
            
        u_exact = np.exp(-a * (2 * np.pi) ** 2 * t) * np.sin(2 * np.pi * (x - t))
    
        x_mid = (x + h/2) % 1 #mod1 because of periodicity
        u_exact_mid = np.exp(-a * (2 * np.pi) ** 2 * t) * np.sin(2 * np.pi * (x_mid - t))
        u_mid = (u[:,-1]+np.roll(u[:,-1],-1))/2 #averaging u at neighbouring x values to get midpoint
    
        a1 = np.sum((h/6)*((u_exact-u[:,-1])**2)[:-1])
        a2 = np.sum((h/6)*((u_exact-u[:,-1])**2)[1:])  
        a3 = np.sum((4*h/6)*(u_exact_mid-u_mid)**2) 
    
        error = np.sqrt(a1+a2+a3)
    
        dts.append(dt)
        errors.append(error)
    
    
    
    dts=np.array(dts)
    errors=np.array(errors)
    c=0.7*errors[0]/(dts[0]**2)
    plt.loglog(dts,errors, '-o', label=f"$h={h}$")
    if N== 400:
       s[N]=errors[-1]
    if N==200:
       s[N]=errors[-1]
        
    
print(s[400]/s[200])    

plt.loglog(dts,c*dts**2,'--',label='$O(\Delta t^2)$')
plt.xlabel('$\Delta t$')
plt.ylabel('$||u(T)-u_h(T)||_{L^2}$')
plt.legend()
plt.title('$L^2$ Error vs $\Delta t$')
plt.savefig('cd_1D_sin_error.pdf', bbox_inches='tight')
plt.show()






