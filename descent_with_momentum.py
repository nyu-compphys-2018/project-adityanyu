# Program to minimize the vector sum of forces on each droplet in the sample (1)
# The state visually closest to the original, where (1) is satisfied is the state we want

#################
### Procedure ###
#################

# 1. Calculate the total energy of the sample.
# 2. For each particle, calculate the energy (potential) of the particle
#    shifted back and forth in X,Y,Z directions, and use this to determine
#    direction (of force) in which it will be moved. (Steepest Descent)
# 3. If the TOTAL ENERGY of the shifted system is less than that the previous, continue. 
# 4. Repeat till minima is found.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as time
import matplotlib.animation as animation
from testcases import * #different test cases are stored here


#################
### Functions ###
#################

# Function for visualizing the droplets in 3D 
def plotsphere(x,y,z,r,ax,k):
    #Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = x + r * np.outer(np.cos(u), np.sin(v))
    ys = y + r * np.outer(np.sin(u), np.sin(v))
    zs = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
    # Plot the surface
    ax.plot_surface(xs, ys, zs, color=k)
    
# Function to visualize the cross section at height 'z' of the droplets
def plotcs(xs,ys,zs,rs,ax,z):
    N = len(xs)
    #Identify droplets that cut plane at height z 
    for i in range(N):
        r_in_cs = rs[i] - np.fabs(zs[i] - z) #Radius of the circle in the cross section at height 'z'
        if r_in_cs > 0: #Droplets that cut plane at height z 
            circle = plt.Circle((xs[i],ys[i]),r_in_cs, facecolor='None', edgecolor='k')
            ax.add_artist(circle)

def localenergy(xs,ys,zs,rs,i):
    U = 0
    k = 1 
    for j in range(N):
        if j != i:
            r = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) - (rs[i] + rs[j]))
            if r<0:
                U += 0.5 * k * r**2
    return U

def totalenergy(xs,ys,zs,rs):
    U = 0
    k=1
    for i in range(N):
        for j in range(0,i):
            r = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) - (rs[i] + rs[j]))
            if r < 0 :
                U += 0.5 * k * r**2
    return U

def force_residual(xs,ys,zs,rs,i): #returns the force components on i'th droplet
    k=1
    N = len(xs)
    fx,fy,fz=0,0,0
    for j in range(N):
        if j!=i:
            d = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)) #Distance bw centers 
            r = d - (rs[i] + rs[j])
            if r<0:
                fx += -k * r * (xs[j]-xs[i])/d
                fy += -k * r * (ys[j]-ys[i])/d
                fz += -k * r * (zs[j]-zs[i])/d
    
    return fx ,fy, fz
#####################
### Load the test ###
#####################
#rv,rc = 0.5, 0.5 
#xs,ys,zs,rs, edge_drops = gettest_bcc(rv, rc)

n=5
r=0.5    
xs,ys,zs,rs,edge_drops = get_n_cube_lattice(n,r)

N = len(xs) #Number of droplets
#####################

#To get an estimate of the dimensions of the box -> to get fixed droplets on boundaries
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
zmin, zmax = min(zs), max(zs)
rmin, rmax = min(rs), max(rs)

########################
### Gradient Descent ###
########################

dx, dy, dz = 1e-5, 1e-5, 1e-5

grad_U_x, grad_U_y, grad_U_z = np.zeros(N), np.zeros(N), np.zeros(N)
v_x,v_y,v_z =  np.zeros(N), np.zeros(N), np.zeros(N)

rf = np.zeros(N - len(edge_drops)) #Magnitude of residual force on the droplets
xs_new, ys_new, zs_new = xs+0.0, ys+0.0, zs+0.0

U_total = totalenergy(xs,ys,zs,rs)

U_total_old = U_total     #Energy of the system currently
U_total_new = U_total - 1 #Energy of the system after one step of descent 

U=[]
RF=[]


def calc_grad(xs,ys,zs,rs):
    dx, dy, dz = 1e-5, 1e-5, 1e-5
    xs_f ,xs_b = xs + 0.0, xs + 0.0
    ys_f ,ys_b = ys + 0.0, ys + 0.0
    zs_f ,zs_b = zs + 0.0, zs + 0.0
    
    xs_f[i] , xs_b[i] = xs[i] + dx  , xs[i] - dx
    ys_f[i] , ys_b[i] = ys[i] + dy  , ys[i] - dy
    zs_f[i] , zs_b[i] = zs[i] + dz  , zs[i] - dz


t1=time.time()

alpha = 0.1 # Start on the descent with this alpha
beta  = 0.9 #Other parameter for descent with momentum
while U_total_old > U_total_new:
    
    for i in range(N):
        xs_f ,xs_b = xs + 0.0, xs + 0.0
        ys_f ,ys_b = ys + 0.0, ys + 0.0
        zs_f ,zs_b = zs + 0.0, zs + 0.0
        
        xs_f[i] , xs_b[i] = xs[i] + dx  , xs[i] - dx
        ys_f[i] , ys_b[i] = ys[i] + dy  , ys[i] - dy
        zs_f[i] , zs_b[i] = zs[i] + dz  , zs[i] - dz
        
        grad_U_x[i] = (localenergy(xs_f,ys,zs,rs,i) - localenergy(xs_b,ys,zs,rs,i) ) /(2*dx)
        grad_U_y[i] = (localenergy(xs,ys_f,zs,rs,i) - localenergy(xs,ys_b,zs,rs,i) ) /(2*dy)
        grad_U_z[i] = (localenergy(xs,ys,zs_f,rs,i) - localenergy(xs,ys,zs_b,rs,i) ) /(2*dx)
        
#        v_x[i] = beta*v_x[i] + (1-beta)*grad_U_x[i]
#        v_y[i] = beta*v_y[i] + (1-beta)*grad_U_y[i]
#        v_z[i] = beta*v_z[i] + (1-beta)*grad_U_z[i]
        
        v_x[i] = beta*v_x[i] + (1-beta)*grad_U_x[i]
        v_y[i] = beta*v_y[i] + (1-beta)*grad_U_y[i]
        v_z[i] = beta*v_z[i] + (1-beta)*grad_U_z[i]
        
        #Update positions for all except edge_drops
        if i not in edge_drops:
            xs_new[i] = xs[i] - alpha * v_x[i]
            ys_new[i] = ys[i] - alpha * v_y[i]
            zs_new[i] = zs[i] - alpha * v_z[i]
        
    j = 0 
    for i in range(N):
        if i not in edge_drops:
            fx, fy, fz = force_residual(xs,ys,zs,rs,i)
            rf[j] = np.sqrt(fx**2 + fy**2 + fz**2)
            j+=1
    
    U.append(U_total_old)
    RF.append(rf.copy())
    U_total_old = totalenergy(xs,ys,zs,rs)
    U_total_new = totalenergy(xs_new, ys_new, zs_new, rs)
    
    if U_total_old > U_total_new:
        xs,ys,zs = xs_new+0.0,ys_new+0.0,zs_new+0.0

t2=time.time()

#fig1=plt.figure()
#plt.plot(U)
#plt.xlabel('iterations')
#plt.ylabel('Total potential energy')
#plt.title('Gradient descent 5x5x5')
##plt.savefig('U5x5x5.jpg', format='jpg', dpi=1000)
#
#
#number_of_frames = int(len(RF))
#
#def update_hist(i, data):
#    plt.cla()
#    plt.hist(RF[i], bins=100, range=[0,0.2])
#    plt.xlim(0,0.1)
#    plt.ylim(0,9)
#
#
#fig4 = plt.figure()
#plt.xlim(0,0.1)
#plt.ylim(0,9)
#hist = plt.hist(RF[0],bins=100, range=[0,0.2])
#anim = animation.FuncAnimation(fig4, update_hist, number_of_frames, fargs=(RF, ))
##writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'),25)
#
##anim.save('5by5by5.gif', writer='imagemagick',fps=10)

        
        
        