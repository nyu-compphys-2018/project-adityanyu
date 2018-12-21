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
    
# Function to visualize the cross section at height 'z' of the droplets
def plotcs(xi,yi,zi,xs,ys,zs,rs,ax,z):
    N = len(xs)
    #Identify droplets that cut plane at height z 
    for i in range(N):
        r_in_cs = rs[i] - np.fabs(zs[i] - z) #Radius of the circle in the cross section at height 'z'
        if r_in_cs > 0: #Droplets that cut plane at height z 
            circle = plt.Circle((xs[i],ys[i]),r_in_cs, facecolor='None', edgecolor='k')
            ax.add_artist(circle)
    
    for i in range(N):
        r_in_cs = rs[i] - np.fabs(zi[i] - z) #Radius of the circle in the cross section at height 'z'
        if r_in_cs > 0: #Droplets that cut plane at height z 
            circle = plt.Circle((xi[i],yi[i]),r_in_cs, facecolor='g', edgecolor='None', alpha=0.5)
            ax.add_artist(circle)

#### Potential energy and residual forces for the harmonic potential ###
#def localenergy(xs,ys,zs,rs,i,nlist):
#    U = 0
#    k = 1
#    for j in nlist[i]:
#        if j != i:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
#            r = (rs[i] + rs[j]) - d 
#            if r>0:
#                U += 0.5 * k * r**2
#    return U
#
#def totalenergy(xs,ys,zs,rs,nlist):
#    U = 0
#    k = 1
#    N = len(xs)
#    for i in range(N):
#        for j in nlist[i]:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
#            r = (rs[i] + rs[j]) - d 
#            if r > 0 :
#                U += 0.5 * k * r**2
#    return U
#
#def force_residual(xs,ys,zs,rs,i,nlist): #returns the force components on i'th droplet
#    k=1
#    N = len(xs)
#    fx,fy,fz=0,0,0
#    for j in nlist[i]:
#        if j!=i:
#            d = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)) #Distance bw centers 
#            r = (rs[i] + rs[j]) - d  #Overlap 
#            if r>0:
#                fx += -k * r * (xs[j]-xs[i])/d
#                fy += -k * r * (ys[j]-ys[i])/d
#                fz += -k * r * (zs[j]-zs[i])/d
#    
#    return fx ,fy, fz

### Potential energy and residual forces for the area dependent anisotropic potential ###            
def localenergy(xs,ys,zs,rs,i,nlist):
    U = 0
    N = len(xs)
    for j in nlist[i]:
        if j != i:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
            x = (rs[i] + rs[j]) - d 
            if x>0:
                U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
    return U

def totalenergy(xs,ys,zs,rs,nlist):
    U = 0
    N = len(xs)
    for i in range(N):
        for j in nlist[i]:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
            x = (rs[i] + rs[j]) - d 
            if x>0:
                U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
    return U

def force_residual(xs,ys,zs,rs,i,nlist): #returns the force components on i'th droplet
    N = len(xs)
    fx,fy,fz=0,0,0
    for j in nlist[i]:
        if j!=i:
            d = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)) #Distance bw centers
            x = (rs[i] + rs[j]) - d
            if x>0:
                fx += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(xs[j]-xs[i])/d
                fy += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(ys[j]-ys[i])/d
                fz += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(zs[j]-zs[i])/d
    
    return fx ,fy, fz

###############################################################################
### Load real data ############################################################
###############################################################################  
xs,ys,zs,rs = get_real_data()
xi,yi,zi = xs,ys,zs
N = len(xs) #Number of droplets

#To get an estimate of the dimensions of the box -> to get fixed droplets on boundaries
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
zmin, zmax = min(zs), max(zs)
rmin, rmax = min(rs), max(rs)

# Identifying the droplets that will remain fixed throughout the descent
edge_drops=[]
for i in range(N):
    if xs[i]<2*rmax or xs[i]>(xmax-2*rmax) or ys[i]<2*rmax or ys[i]>(ymax-2*rmax) or zs[i]<2*rmax or zs[i]>(zmax-2*rmax):
        edge_drops.append(i)
edge_drops = np.array(edge_drops)

#Creating the neighbourlist of all droplets
n_list = []
for i in range(N):
    n_of_i = []
    for j in range(N):
        if j!=i:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
            r = d - (rs[i] + rs[j])
            if r<1:
                n_of_i.append(j)
    n_list.append(np.array(n_of_i))
                

###############################################################################

########################
### Gradient Descent ###
########################

rf = np.zeros(N - len(edge_drops)) #Magnitude of residual force on the droplets
xs_new, ys_new, zs_new = xs+0.0, ys+0.0, zs+0.0

U_total = totalenergy(xs,ys,zs,rs,n_list)

U_total_old = U_total     #Energy of the system currently
U_total_new = U_total - 1 #Energy of the system after one step of descent 

U=[]
RF=[]

t1=time.time()   

alpha = 1e-3    
while U_total_old > U_total_new and alpha>1e-6:
    
    j = 0
    for i in range(N):
        if i not in edge_drops:
            
            fx, fy, fz = force_residual(xs,ys,zs,rs,i,n_list)
            rf[j] = np.sqrt(fx**2 + fy**2 + fz**2)
            #Update positions for all except edge_drops
            
            if rf[j] == 0:
                xs_new[i] = xs[i]
                ys_new[i] = ys[i]
                zs_new[i] = zs[i] 
            else:
                xs_new[i] = xs[i] + alpha * fx
                ys_new[i] = ys[i] + alpha * fy
                zs_new[i] = zs[i] + alpha * fz
            
            j+=1
            
    U_total_old = totalenergy(xs,ys,zs,rs,n_list)
    U_total_new = totalenergy(xs_new, ys_new, zs_new, rs,n_list)
    
    if U_total_old > U_total_new:
        xs,ys,zs = xs_new+0.0,ys_new+0.0,zs_new+0.0
        U.append(U_total_new)
        RF.append(rf.copy())
    else:
        alpha = alpha/10
        print(alpha)
        U_total_new = U_total_old - 1

t2=time.time()

sumrf = []
for i in range(len(RF)):
    sumrf.append(sum(RF[i]))

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot( U[1:] )
ax1.set_ylabel('Total Energy (1/$\sigma$)')
#ax1.hlines(5.999, 1, 500, linestyles='dashed')
ax2 = fig.add_subplot(2,1,2)
ax2.plot(sumrf[1:])
ax2.set_xlabel('iterations')
ax2.set_ylabel('$\sum$ |residual forces|  (1/$\sigma$)')
#plt.savefig('emulsion_cubic_anisotropic_UandF.jpeg', format='jpeg')
#plt.show()

fig3 = plt.figure()
plt.hist(RF[0], bins=1000, label='Initial distribution',alpha=0.7)
#plt.hist(RF[10000], bins=1000, label='Final distribution')
plt.xscale('log')
plt.xlabel('|residual force|')
plt.ylabel('Number of droplets')
plt.legend()
#plt.savefig('emulsion_cubic_anisotropic_residual_histogram.jpg',format='jpeg')
#
fig,ax = plt.subplots()
plotcs(xi,yi,zi,xs,ys,zs,rs,ax,30)
ax.set_xlim(0, 144)
ax.set_ylim(0, 144)
ax.set_xlabel('microns')
ax.set_ylabel('microns')
#plt.savefig('emulsion_cubic_anisotropic_cross_section.jpg',format='jpeg')

    
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
#hist = plt.hist(RF[0],bins=100000, range=[0,0.2])
#anim = animation.FuncAnimation(fig4, update_hist, number_of_frames, fargs=(RF, ))
##writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'),25)
##
##anim.save('5by5by5.gif', writer='imagemagick',fps=10)

        
        
        


