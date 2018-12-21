import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as time
import matplotlib.animation as animation
from testcases import * #different test cases are stored here


##### Potential energy and residual forces for the harmonic potential ###
#
#def localenergy(xs,ys,zs,rs,i):
#    U = 0
#    k = 1 
#    N = len(xs)
#    for j in range(N):
#        if j != i:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
#            r = (rs[i] + rs[j]) - d 
#            if r>0:
#                U += 0.5 * k * r**2
#    return U
#
#def totalenergy(xs,ys,zs,rs):
#    U = 0
#    k = 1
#    N = len(xs)
#    for i in range(N):
#        for j in range(i):
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
#            r = (rs[i] + rs[j]) - d 
#            if r > 0 :
#                U += 0.5 * k * r**2
#    return U
#
#def force_residual(xs,ys,zs,rs,i): #returns the force components on i'th droplet
#    k=1
#    N = len(xs)
#    fx,fy,fz=0,0,0
#    for j in range(N):
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
def localenergy(xs,ys,zs,rs,i):
    U = 0
    for j in range(N):
        if j!=i:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
            x = (rs[i] + rs[j]) - d
            if x>0:
                U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
    return U

def totalenergy(xs,ys,zs,rs):
    U = 0
    for i in range(N):
        for j in range(i):
            if j!=i:
                d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
                x = (rs[i] + rs[j]) - d
                if x>0:
                    U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
                if x>0.5:
                    print(x)
    return U


def force_residual(xs,ys,zs,rs,i): #returns the force components on i'th droplet
    N = len(xs)
    fx,fy,fz=0,0,0
    for j in range(N):
        if j!=i:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) #Distance bw centers 
            x = (rs[i] + rs[j]) - d
            if x>0:
                fx += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(xs[j]-xs[i])/d
                fy += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(ys[j]-ys[i])/d
                fz += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(zs[j]-zs[i])/d
    
    return fx ,fy, fz

######## MAIN CODE #####

xs, ys, zs,rs, edge_drops = xs,ys,zs,rs,edge_drops = get_n_cube_lattice(5,0.6)
xi,yi,zi = xs+0.0,ys+0.0,zs+0.0
N = len(xs)

rf = np.zeros(N - len(edge_drops)) #Magnitude of residual force on the droplets
xs_new, ys_new, zs_new = xs+0.0, ys+0.0, zs+0.0

U_total = totalenergy(xs,ys,zs,rs)

U_total_old = U_total     #Energy of the system currently
U_total_new = U_total - 1 #Energy of the system after one step of descent 

U=[]
RF=[]

t1=time.time()

alpha = 1e-3
while U_total_old > U_total_new and alpha>1e-7:
    
    j = 0
    for i in range(N):
        if i not in edge_drops:
            
            fx, fy, fz = force_residual(xs_new,ys_new,zs_new,rs,i)
            rf[j] = np.sqrt(fx**2 + fy**2 + fz**2)
            #Update positions for all except edge_drops
            
            if rf[j]==0:
                xs_new[i] = xs[i]
                ys_new[i] = ys[i]
                zs_new[i] = zs[i]
            else:
                xs_new[i] = xs[i] + alpha * fx
                ys_new[i] = ys[i] + alpha * fy
                zs_new[i] = zs[i] + alpha * fz
            
            j= j+1
    
    U_total_old = totalenergy(xs,ys,zs,rs)
    U_total_new = totalenergy(xs_new, ys_new, zs_new, rs)
    
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

#fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#ax1.plot( U[1:] )
#ax1.set_ylabel('Total Energy (k=1)')
##ax1.hlines(5.999, 1, 500, linestyles='dashed')
#ax2 = fig.add_subplot(2,1,2)
#ax2.plot(sumrf)
#ax2.set_xlabel('iterations')
#ax2.set_ylabel('$\sum$ |residual forces|')
##plt.savefig('5by5by5_polydisperse.jpeg', format='jpeg')
##plt.show()
#
#fig4 = plt.figure()
#plt.hist(RF[0], bins=100, label='Initial distribution')
#plt.hist(RF[92],bins=100, label='Final distribution',alpha=0.7)
#plt.xlabel('|residual force|')
#plt.ylabel('Number of droplets')
#plt.legend()
#plt.savefig('5by5by5_polydisperse_histogram_adaptive.jpg',format='jpeg')



#def plotsphere(x,y,z,r,ax,k):
#    
#    #Make data
#    u = np.linspace(0, 2 * np.pi, 100)
#    v = np.linspace(0, np.pi, 100)
#    xs = x + r * np.outer(np.cos(u), np.sin(v))
#    ys = y + r * np.outer(np.sin(u), np.sin(v))
#    zs = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
#    
#    # Plot the surface
#    ax.plot_surface(xs, ys, zs, color=k)
#    plt.show()
#
#fig2 = plt.figure()
#ax = plt.axes(projection='3d')
#for i in range(N):
#    if i not in edge_drops:
#        plotsphere(xs[i],ys[i],zs[i],rs[i],ax,'y')

#number_of_frames = int(len(RF)/10)
#
#def update_hist(i, data):
#    plt.cla()
#    plt.hist(RF[i*10], bins=100, range=[0,0.2])
#    plt.xlim(0,0.2)
#    plt.ylim(0,27)
#    plt.xlabel('|residual force|')
#    plt.ylabel('number of droplets')
#
#
#fig4 = plt.figure()
#plt.xlabel('|residual force|')
#plt.ylabel('number of droplets')
#plt.xlim(0,0.2)
#plt.ylim(0,27)
#hist = plt.hist(RF[0],bins=100, range=[0,0.2])
#anim = animation.FuncAnimation(fig4, update_hist, number_of_frames, fargs=(RF, ))
##writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'),25)
##anim.save('5by5by5_r=0.6.gif', writer='imagemagick',fps=10)
