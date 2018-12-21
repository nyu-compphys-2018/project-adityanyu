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
    plt.show()

#### Potential energy and residual forces for the harmonic potential ###

def localenergy(xs,ys,zs,rs,i):
    U = 0
    k = 1 
    N = len(xs)
    for j in range(N):
        if j != i:
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
            r = (rs[i] + rs[j]) - d 
            if r>0:
                U += 0.5 * k * r**2
    return U

def totalenergy(xs,ys,zs,rs):
    U = 0
    k = 1
    N = len(xs)
    for i in range(N):
        for j in range(i):
            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) 
            r = (rs[i] + rs[j]) - d 
            if r > 0 :
                U += 0.5 * k * r**2
    return U

def force_residual(xs,ys,zs,rs,i): #returns the force components on i'th droplet
    k=1
    N = len(xs)
    fx,fy,fz=0,0,0
    for j in range(N):
        if j!=i:
            d = ((((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)) #Distance bw centers 
            r = (rs[i] + rs[j]) - d  #Overlap 
            if r>0:
                fx += -k * r * (xs[j]-xs[i])/d
                fy += -k * r * (ys[j]-ys[i])/d
                fz += -k * r * (zs[j]-zs[i])/d
    
    return fx ,fy, fz



#### Potential energy and residual forces for the area dependent anisotropic potential ###
#def localenergy(xs,ys,zs,rs,i):
#    u = 0
#    N = len(xs)
#    for j in range(N):
#        if j!=i:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
#            x = (rs[i] + rs[j]) - d
#            if x>0:
#                u += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
#    return u
#
#def totalenergy(xs,ys,zs,rs):
#    U = 0
#    N = len(xs)
#    for i in range(N):
#        for j in range(i):
#            if j!=i:
#                d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
#                x = (rs[i] + rs[j]) - d
#                if x>0:
#                    U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
#    return U
#
#
#def force_residual(xs,ys,zs,rs,i): #returns the force components on i'th droplet
#    N = len(xs)
#    fx,fy,fz=0,0,0
#    for j in range(N):
#        if j!=i:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) #Distance bw centers 
#            x = (rs[i] + rs[j]) - d
#            if x>0:
#                s = (rs[i] + rs[j] + d)/2  
#                fx += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(xs[j]-xs[i])/d
#                fy += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(ys[j]-ys[i])/d
#                fz += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(zs[j]-zs[i])/d
#    
#    return fx ,fy, fz

#####################
### Load the test ###
#####################
rv,rc = 0.4, 0.6 
xs,ys,zs,rs, edge_drops = gettest_bcc(rv, rc)
edge_drops=np.arange(8)
print(xs[8],ys[8],zs[8])
N = len(xs) #Number of droplets
for i in range(8):
    rs[i] += np.random.uniform(-rc/10,rc/10)
#####################

#To get an estimate of the dimensions of the box -> to get fixed droplets on boundaries
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
zmin, zmax = min(zs), max(zs)
rmin, rmax = min(rs), max(rs)

########################
### Gradient Descent ###
########################


xi,yi,zi = xs+0.0,ys+0.0,zs+0.0
N = len(xs)
dx, dy, dz = 1e-5, 1e-5, 1e-5
grad_U_x, grad_U_y, grad_U_z = np.zeros(N), np.zeros(N), np.zeros(N)
rf = np.zeros(N - len(edge_drops)) #Magnitude of residual force on the droplets
xs_new, ys_new, zs_new = xs+0.0, ys+0.0, zs+0.0

U_total = totalenergy(xs,ys,zs,rs)

U_total_old = U_total     #Energy of the system currently
U_total_new = U_total - 1 #Energy of the system after one step of descent 

U=[]
RF=[]
x=[]
y=[]
z=[]
t1=time.time()


alpha = 0.0001
while U_total_old > U_total_new and alpha>1e-10:
    j = 0
    for i in range(N):
        if i not in edge_drops:
            xs_f ,xs_b = xs + 0.0, xs + 0.0
            ys_f ,ys_b = ys + 0.0, ys + 0.0
            zs_f ,zs_b = zs + 0.0, zs + 0.0
            
            xs_f[i] , xs_b[i] = xs[i] + dx  , xs[i] - dx
            ys_f[i] , ys_b[i] = ys[i] + dy  , ys[i] - dy
            zs_f[i] , zs_b[i] = zs[i] + dz  , zs[i] - dz
            
#            grad_U_x[i] = (localenergy(xs_f,ys,zs,rs,i) - localenergy(xs_b,ys,zs,rs,i) ) /(2*dx)
#            grad_U_y[i] = (localenergy(xs,ys_f,zs,rs,i) - localenergy(xs,ys_b,zs,rs,i) ) /(2*dy)
#            grad_U_z[i] = (localenergy(xs,ys,zs_f,rs,i) - localenergy(xs,ys,zs_b,rs,i) ) /(2*dx)
            
            fx, fy, fz = force_residual(xs,ys,zs,rs,i)
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
        x.append(xs[8])
        y.append(ys[8])
        z.append(zs[8])
        
t2=time.time()

### To generate the plots 
fig1=plt.figure()
plt.plot(U)
plt.xlabel('iterations')
plt.ylabel('Potential energy (k=1)')
#plt.savefig('U_bcc.jpg', format='jpg')

fig2=plt.figure()
plt.plot(RF)
plt.xlabel('iterations')
plt.ylabel('|Residual force|')
#plt.savefig('RF_bcc.jpg', format='jpg')

fig3=plt.figure()
plt.plot(x,label='x position')
plt.plot(y,label='y position')
plt.plot(z,label='z position')
plt.xlabel('iterations')
plt.ylabel('position')
plt.title('Evolution of Position')
plt.legend()
#plt.savefig('position_bcc.jpg', format='jpg')

fig1 = plt.figure()
ax = plt.axes(projection='3d')
for i in range(8):
    plotsphere(xs[i],ys[i],zs[i],rs[i],ax,'g')
for i in range(8,9):
    plotsphere(xs[i],ys[i],zs[i],rs[i],ax,'k')


###To run the animation ###

#plt.savefig('picture_bcc.jpg', format='jpg')

#number_of_frames = int(len(RF)/100)
#
#def update_hist(i, data):
#    plt.cla()
#    plt.hist(RF[i*100], bins=100, range=[0,0.2])
#    plt.xlim(0,0.2)
#    plt.ylim(0,2)
#
#
#fig3 = plt.figure()
#plt.xlim(0,0.2)
#plt.ylim(0,2)
#hist = plt.hist(RF[0],bins=100, range=[0,0.2])
#anim = animation.FuncAnimation(fig3, update_hist, number_of_frames, fargs=(RF, ))
#writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'),25)

#anim.save('test.gif', writer='imagemagick',fps=1)
    



        
        
        