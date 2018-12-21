import numpy as np

# Test case 1: Unit bcc cell 

#This test case is the bcc unit cell where droplets(spheres) on the vertices are fixed 
#and the body center droplet is allowed to shift in position.
def gettest_bcc(rv, rc):
    #rc is the radius of the central droplet
    #rv is the radius of the bounding (vertex) droplets
    xs, ys, zs, rs = np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)
    count = 0 
    for i in range (2):
        for j in range (2):
            for k in range(2):
                xs[count], ys[count], zs[count], rs[count] = i, j, k, rv
                count += 1
                
    xs[8],ys[8],zs[8],rs[8] = 0.5,0.5,0.5,rc
    
    #Disturbing position of the central droplet
    xs[8] += np.random.uniform(-0.05,0.05)
    ys[8] += np.random.uniform(-0.05,0.05)
    zs[8] += np.random.uniform(-0.05,0.05)
    
    edge_drops=[0,1,2,3,4,5,6,7] #Droplets that stay fixed
    edge_drops = np.array(edge_drops) #Converting to np array 

    #For polydispersity, with introduce a change in radius of the central and edgedrops 
    for i in range(9): 
        rs[i] += np.random.uniform(-rc/10,rc/10)
        
    return xs, ys, zs, rs, edge_drops

#Test Case 2: nxnxn cubic lattice

def get_n_cube_lattice(n,r):
    xs, ys, zs, rs = np.zeros(int(n)**3), np.zeros(int(n)**3), np.zeros(int(n)**3), np.zeros(int(n)**3)
    count = 0 
    for i in range (n):
        for j in range (n):
            for k in range(n):
                xs[count], ys[count], zs[count], rs[count] = i, j, k, r
                count += 1
                
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    rmin, rmax = min(rs), max(rs)
    
    N = len(xs)
    edge_drops=[]
    for i in range(N):
        if xs[i]<rmax or xs[i]>(xmax-rmax) or ys[i]<rmax or ys[i]>(ymax-rmax) or zs[i]<rmax or zs[i]>(zmax-rmax):
            edge_drops.append(i)
    edge_drops = np.array(edge_drops) #Converting to np array 

    #Disturbing position of the central droplets
    for i in range (N):
        if i not in edge_drops:
            xs[i] += np.random.uniform(-r/10,r/10)
            ys[i] += np.random.uniform(-r/10,r/10)
            zs[i] += np.random.uniform(-r/10,r/10)
    
    #For polydispersity 
    for i in range(N): 
        rs[i] += np.random.uniform(-r/10,r/10)
        
    return xs,ys,zs,rs, edge_drops

#Test case 3: Experimental data
def get_real_data():
    #Loading data from the text file into arrays
    data = np.loadtxt(fname = "test_data.txt") #loads into np array "data"
    xs = data[:,0] * 144.7245/512
    ys = data[:,1] * 144.7245/512
    zs = data[:,2] * 48.0373/228
    rs = data[:,3]
    return xs,ys,zs,rs

#######################################
########### FORCE MODELS ##############
#######################################
    
#def localenergy(xs,ys,zs,rs,i,nlist):
#    U = 0
#    k = 1 
#    for j in nlist[i]:
#        d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
#        x = (rs[i] + rs[j]) - d
#        if x>0:
#            U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
#    return U
#
#def totalenergy(xs,ys,zs,rs,nlist):
#    U = 0
#    k=1
#    for i in range(N):
#        for j in nlist[i]:
#            d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5)
#            x = (rs[i] + rs[j]) - d
#            if x>0:
#                U += - ( ( (rs[i]+rs[j]) * (x**4 - 4*(rs[i]+rs[j])*x**3 + 12*(rs[i]*rs[j])*x**2 + 3*((rs[i]-rs[j])**2)*(rs[i]+rs[j])*x - 3*rs[i]**4 - 3*rs[j]**4 + 6*(rs[i]**2)*(rs[j]**2)   ) ) / ( (12*rs[i]*rs[j]) * ( x - rs[i] - rs[j]) ) )
#    return U
#
#
#def force_residual(xs,ys,zs,rs,i,nlist): #returns the force components on i'th droplet
#    k=1
#    N = len(xs)
#    fx,fy,fz=0,0,0
#    for j in nlist[i]:
#        d = (((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)**0.5) #Distance bw centers 
#        x = (rs[i] + rs[j]) - d
#        if x>0:
#            s = (rs[i] + rs[j] + d)/2  
#            fx += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(xs[j]-xs[i])/d
#            fy += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(ys[j]-ys[i])/d
#            fz += - (( (rs[i]+rs[j])*(2*rs[i] + 2*rs[j] - x)*(x)*(2*rs[i]-x)*(2*rs[j]-x) ) / ( (rs[i]*rs[j])*(4)*((rs[i]+rs[j]-x)**2)))*(zs[j]-zs[i])/d
#    
#    return fx ,fy, fz