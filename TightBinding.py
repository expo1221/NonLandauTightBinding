import random
import time
import scipy
import csv
import queue
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigs
from sympy.core.intfunc import igcdex #get (x,y,g) s.t. ax+yb=(a,b)
from math import *
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

def erase(a, b): #erase_b a
    if a%abs(b) == 0:
        return 1
    else:
        p=2
        t=abs(a)
        tp = abs(b)
        while t>1 and tp>1:
            if tp%p == 0:
                while tp%p == 0:
                    tp = tp//p
                while t%p == 0:
                    t = t//p
                if t<p or tp < p:
                    break
            p = p+1
        return t

def getBasis(A): #get Basis(with scalar field Z) from the set of an integer vectors in Z^2
    xallzero=False
    yallzero=False
    A=[(np.array(elem) if elem[0]>=0 else -np.array(elem)) for elem in A]
    while(True):
        arr0 = np.array([i[0] for i in A])
        if any(arr0):
            xpivot = np.argmin(np.ma.masked_array(arr0, mask=arr0==0))
        else:
            xallzero=True
            break
        breakornot=True
        for i in range(len(A)):
            if i !=xpivot and A[i][0]!=0:
                breakornot=False
                A[i]-=(A[i][0]//A[xpivot][0])*A[xpivot]
        if breakornot:
            break
        
    if xallzero is True:
        xpivot = -1
    A=[(A[i] if (i==xpivot or A[i][1]>=0) else -A[i]) for i in range(len(A))]
    while(True):
        yallzero=True
        for i in range(len(A)):
            if i!=xpivot and A[i][1]!=0:
                yallzero=False
                break
        if yallzero:
            break
        ypivot=i
        for j in range(i+1,len(A)):
            if j!=xpivot and A[j][1]<A[ypivot][1] and A[j][1]!=0:
                ypivot=j
        breakornot=True
        for i in range(len(A)):
            if i!=xpivot and i !=ypivot and A[i][1]!=0:
                breakornot=False
                A[i]-=(A[i][1]//A[ypivot][1])*A[ypivot]
        if breakornot:
            break
    if yallzero is False:
        A[xpivot]-=(A[xpivot][1]//A[ypivot][1])*A[ypivot]
    if xallzero:
        if yallzero:
            return [np.array([0,0]),np.array([0,0])]
        else:
            return [np.array([0,0]),A[ypivot]]
    else:
        if yallzero:
            return [A[xpivot],np.array([0,0])]
        else:
            return [A[xpivot],A[ypivot]]

class TB:
    def __init__(self, lattice, atoms, hops, symbolic = True):
        self.lattice = sp.Matrix(lattice).T
        #check |lattice|>0
        t = self.lattice.det()
        if (symbolic == False or t.is_real) and t<=0:
            raise Exception('Exception in TB __init__:Wrong lattice')
        self.atoms = {(0,atom):sp.sympify(atoms[atom]) for atom in atoms}
        if symbolic is False:
            self.lattice = np.matrix(self.lattice)
            self.atoms = {(0,atom):[float(t) for t in self.atoms[atom]] for atom in self.atoms}

        self.hops=set()
        for hop in hops:
            if (len(hop) !=4) or (hop[0] not in atoms) or (hop[1] not in atoms) or (len(hop[2])!=2):
                raise Exception('Exception in TB __init__:Wrong hopping')
            tmp = sp.sympify(hop[2])
            if all([(t.is_integer) for t in tmp]) == False:
                raise Exception('Exception in TB __init__:Wrong hopping')
            self.hops.add(((0,hop[0]),(0,hop[1]),tmp,hop[3]))
        
        self.D = None
        self.layers = [{atom for atom in atoms}]
        self.symbolic = True
        if symbolic == False:
            self.symoff()
    
    def getD(self):
        if self.symbolic is False:
            raise Exception('Exception in getD:Only symbolic TB can be used to get D')
        if self.D is not None:
            return self.D
        latticeinv=self.lattice.inv()
        X=[]
        Y=[]
        for hop in self.hops:
            hop_vector = latticeinv*(sp.Matrix(self.atoms[hop[1]][0:2])-sp.Matrix(self.atoms[hop[0]][0:2])) + sp.Matrix(hop[2])
            if hop_vector[0].is_rational and hop_vector[1].is_rational:
                X.append(hop_vector[0])
                Y.append(hop_vector[1])
            else:
                raise Exception('Exception in getD:Rank>2')
        lcmX = lcm(*[x.q for x in X])
        lcmY = lcm(*[y.q for y in Y])
        A=[[int(X[i]*lcmX), int(Y[i]*lcmY)] for i in range(len(X))]
        B=getBasis(A)
        self.D = sp.Matrix([[sp.Rational(B[0][0],lcmX),sp.Rational(B[1][0],lcmX)],[sp.Rational(B[0][1],lcmY),sp.Rational(B[1][1],lcmY)]])
        return self.D
    
    def extend(self, Transform_matrix): #duplicated
        matrix = sp.Matrix if self.symbolic else np.matrix
        T=matrix(Transform_matrix)
        detT = T.det() if self.symbolic else np.linalg.det(T)
        if T.shape != (2,2) or (all([(t.is_integer) for t in T]) if self.symbolic else np.equal(np.mod(T,1),0).all()) == False or detT<=0:
            raise Exception('Exception in extend:Wrong Transform_matrix.')
        lattice = self.lattice * T
        J=set()
        atoms={}
        T_2=gcd(T[0,1],T[1,1])
        Tinv = T.inv() if self.symbolic else np.linalg.inv(T)
        layers=[]
        for i in range(len(self.layers)):
            layers.append(set())
        for i in range(detT//T_2):
            for j in range(T_2):
                z=Tinv * matrix([[i],[j]])
                t=matrix([[i],[j]])-T*matrix([[floor(z[0,0])],[floor(z[1,0])]])
                J.add((t[0,0],t[1,0]))
                for (n,atom) in self.atoms:
                    v = matrix(self.atoms[(n,atom)][0:2]) + self.lattice*t
                    atoms[(n,(atom,(int(t[0,0]),int(t[1,0]))))] = [v[0],v[1]]+self.atoms[(n,atom)][2:]
                    layers[n].add((atom,(int(t[0,0]),int(t[1,0]))))
        hops=set()
        for j in J:
            for hop in self.hops:
                z=Tinv * matrix([j[0]+hop[2][0],j[1]+hop[2][1]])
                z=matrix([[floor(z[0,0])],[floor(z[1,0])]])
                t=matrix([j[0]+hop[2][0],j[1]+hop[2][1]])-T*z
                hops.add(((hop[0][0],(hop[0][1],(int(j[0]),int(j[1])))), (hop[1][0],(hop[1][1],(int(t[0,0]),int(t[1,0])))), (z[0,0],z[1,0]), hop[3]))
        model = TB([[1,0],[0,1]],{},{},self.symbolic)
        model.lattice = lattice
        model.atoms = atoms
        model.hops = hops
        model.layers = layers
        if self.D != None:
            model.D = Tinv * self.D
        return model

    def merge(self, TBmodel): #not duplicated
        if (self.lattice != TBmodel.lattice) if (self.symbolic and TBmodel.symbolic) else (np.allclose(np.array(self.lattice, dtype=float), np.array(TBmodel.lattice,dtype=float)) == False):
            raise Exception('Exception in merge:Two lattices do not match')
        model = self
        if model.symbolic and (TBmodel.symbolic == False):
            model.symoff()
        if (model.symbolic == False) and TBmodel.symbolic:
            TBmodel = TBmodel.duplicated().symoff()
        for (n,atom) in TBmodel.atoms:
            model.atoms[(len(self.layers)+n, atom)] = TBmodel.atoms[(n,atom)]

        for hop in TBmodel.hops:
            model.hops.add(((len(self.layers)+hop[0][0],hop[0][1]), (len(self.layers)+hop[1][0],hop[1][1]), hop[2], hop[3]))
        model.layers = self.layers + TBmodel.layers
        
        return model

    def symoff(self):
        self.symbolic = False
        self.lattice = np.matrix(self.lattice,dtype=float)
        self.atoms = {atom:[float(t) for t in self.atoms[atom]] for atom in self.atoms}
        self.hops = {(hop[0],hop[1],tuple(int(t) for t in hop[2]),hop[3]) for hop in self.hops}
        return self
    
    def duplicated(self):
        model = TB([[1,0],[0,1]],{},{},self.symbolic)
        model.lattice = deepcopy(self.lattice)
        model.atoms = deepcopy(self.atoms)
        model.hops = deepcopy(self.hops)
        model.layers = deepcopy(self.layers)
        model.D = deepcopy(self.D)
        
        return model

    def translate(self,v): #not duplicated
        v = [sp.sympify(x) for x in v]
        if self.symbolic == False:
            v = [float(x) for x in v]
        atoms = dict()
        for atom in self.atoms:
            atoms[atom]=[]
            for i in range(max(len(self.atoms[atom]), len(v))):
                tmp = 0
                if i<len(self.atoms[atom]):
                    tmp += self.atoms[atom][i]
                if i<len(v):
                    tmp += v[i]
                atoms[atom].append(tmp)
        self.atoms = atoms
        return self

    def rotate(self,angle = None, sine = None): #not duplicated
        if angle !=None:
            c = sp.cos(sp.sympify(angle))
            s = sp.sin(sp.sympify(angle))
        elif sine != None:
            s = sp.sympify(sine)
            c = sp.sqrt(1-s*s)
        else:
            raise Exception('Exception in rotate: Angle or sine should be given')
        if self.symbolic == False:
            c = float(c)
            s = float(s)
        for atom in self.atoms:
            x = self.atoms[atom][0]
            y = self.atoms[atom][1]
            self.atoms[atom][0] = c*x-s*y
            self.atoms[atom][1] = s*x+c*y
        lat = deepcopy(self.lattice)
        for i in range(2):
            self.lattice[0,i] = c*lat[0,i]-s*lat[1,i]
            self.lattice[1,i] = s*lat[0,i]+c*lat[1,i]
        return self

    def Hamiltonian(self, k):
        k = sp.sympify(k)
        if self.symbolic == False:
            k = [float(x) for x in k]
        idx={}
        cnt = 0
        for atom in self.atoms:
            idx[atom] = cnt
            cnt = cnt+1
        H = [[0 for _ in range(cnt)] for _ in range(cnt)]
        for hop in self.hops:
            hop_term = - hop[3]*np.exp(-1j*2*pi*float(k[0]*hop[2][0]+k[1]*hop[2][1])) 
            H[idx[hop[0]]][idx[hop[1]]] = H[idx[hop[0]]][idx[hop[1]]] + hop_term
            H[idx[hop[1]]][idx[hop[0]]] = H[idx[hop[1]]][idx[hop[0]]] + np.conj(hop_term)
        return [H,idx]

class MagneticTB:
    def __init__(self, TBmodel, Phi = None, A = None, T=None, potential_type = "nonlinear", symbolic = None):
        if symbolic == None:
            symbolic = TBmodel.symbolic
        if TBmodel.symbolic == False and symbolic:
            raise Exception('Exception in __init__:Can not change non-symbolic TB model to symbolic MagneticTB model')
        if TBmodel.symbolic == False and potential_type != "nonlinear":
            raise Exception('Exception in __init__:Nonsymbolic TB model can only be used with nonlinear vector potential')
        matrix = sp.Matrix if symbolic else np.matrix
        if Phi != None:
            Phi = sp.sympify(Phi)
        if (potential_type in ["linear","landau","nonlinear"]) is False:
            raise Exception('Exception in __init__:Unvalid potential_type')
        if A != None and T!=None:
            A = sp.Matrix(A)
            T = sp.Matrix(T)
            if Phi != None:
                if A[1,0]-A[0,1] != Phi:
                    raise Exception('Exception in __init__:Phi does not match with A')
            else:
                Phi = A[1,0]-A[0,1]
        else:
            if Phi == None:
                raise Exception('Exception in __init__:You should enter Phi or (A and T)')
            if potential_type == "nonlinear":
                A=sp.Matrix([[0,0],[Phi,0]])
                T=sp.Matrix([[Phi.q, 0],[0,1]])
            else:
                D=TBmodel.getD()
                lcm0 = lcm(D[0,0].q,D[0,1].q,D[1,0].q,D[1,1].q)
                D_=[[(lcm0*D[0,0]).p, (lcm0*D[0,1]).p], [(lcm0*D[1,0]).p, (lcm0*D[1,1]).p]]
                if gcd(D_[0][0],D_[0][1],D_[1][0],D_[1][1]) != 1:
                    raise Exception('Exception in __init__:Wrong D. Connected component may be >=2')
                alpha = lcm0 // (D_[0][0]*D_[1][1])
                beta1 = D_[0][0]
                beta2 = gcd(D_[1][0],D_[1][1])
                gamma = D_[1][1]//beta2
                gamma21 = D_[1][0]//beta2
                if potential_type == "linear":
                    erase_gamma_beta1 = erase(beta1, gamma)
                    t,_,_ = igcdex(gamma, erase_gamma_beta1)
                    delta = beta2 * (gamma*(1+gamma21*t) - gamma21)
                    i1, i2, _ = igcdex(delta, beta1)
                    A=Phi*sp.Matrix([[delta],[beta1]])*sp.Matrix([[i2, -i1]])
                    T=sp.Matrix([[beta1, i1],[-delta,i2]])*sp.Matrix([[(Phi/alpha).q, 0],[0,1]])         
                elif potential_type == "landau":
                    A=sp.Matrix([[0,0],[Phi,0]])
                    T=sp.Matrix([[(Phi/alpha/beta1/gamma).q, 0],[0,1]])
        
        if all([t.is_integer for t in T]) == False:
            raise Exception("Exception in __init__:Invalid T")
        if potential_type == "linear" or potential_type == "landau":
            D=TBmodel.getD()
            DAT = D.T*A*T
            if DAT[0,0].q !=1 or DAT[0,1].q !=1 or DAT[1,0].q !=1 or DAT[1,1].q !=1:
                raise Exception("Exception in __init__:Invalid A or T")
        elif potential_type == "nonlinear":
            TAT = T.T*A*T
            if TAT[0,0].q !=1 or TAT[0,1].q !=1 or TAT[1,0].q !=1 or TAT[1,1].q !=1:
                raise Exception("Exception in __init__:Invalid A or T")

        detT = T.det()
        Tinv = T.inv()
        T_2=gcd(T[0,1],T[1,1])
        
        if symbolic:
            lattice = TBmodel.lattice
            latinv = lattice.inv()
            atoms_ = TBmodel.atoms
        else:
            T = np.matrix(T,dtype=float)
            A = np.matrix(A,dtype=float)
            Phi = float(Phi)
            lattice = np.matrix(TBmodel.lattice)
            latinv = np.linalg.inv(lattice)
            if TBmodel.symbolic:
                atoms_ = {atom:[float(t) for t in self.atoms[atom]] for atom in TBmodel.atoms}
            else:
                atoms_ = TBmodel.atoms
            
        
        superlattice = TBmodel.lattice * T
        J=set()
        atoms={}
        layers=[]
        for i in range(len(TBmodel.layers)):
            layers.append(set())
        for i in range(detT//T_2):
            for j in range(T_2):
                z=Tinv * matrix([[i],[j]])
                t=matrix([[i],[j]])-T*matrix([[floor(z[0,0])],[floor(z[1,0])]])
                J.add((int(t[0,0]),int(t[1,0])))
                for (n,atom) in atoms_:
                    v = matrix([atoms_[(n,atom)][0:2]]).T + lattice*t
                    atoms[(n,(atom,(int(t[0,0]),int(t[1,0]))))] = [v[0,0],v[1,0]]+atoms_[(n,atom)][2:]
                    layers[n].add((atom,(int(t[0,0]),int(t[1,0]))))
        hops=set()
        for j in J:
            for hop in TBmodel.hops:
                z=matrix(Tinv) * matrix([[j[0]+hop[2][0]],[j[1]+hop[2][1]]])
                z=matrix([[floor(z[0,0])],[floor(z[1,0])]])
                t=matrix([[j[0]+hop[2][0]],[j[1]+hop[2][1]]])-T*z
                Delta = T* matrix([[z[0,0]],[z[1,0]]])
                Ri = matrix([[j[0]],[j[1]]]) + latinv*matrix([[TBmodel.atoms[hop[0]][0]],[TBmodel.atoms[hop[0]][1]]])
                Rf = t+latinv*matrix([[TBmodel.atoms[hop[1]][0]],[TBmodel.atoms[hop[1]][1]]])
                if potential_type == "nonlinear":
                    phase = Delta.T*A*Delta + Phi*Rf.T*matrix([[0,-1],[1,0]])*Ri+ Phi*Delta.T*matrix([[0,-1],[1,0]])*(Ri+Rf)
                    phase = phase[0,0] /2
                else:
                    phase = (Delta+Rf-Ri).T*A*Delta +Delta.T*A*(Ri+Rf) +Phi*Rf.T*sp.Matrix([[0,-1],[1,0]])*Ri +Rf.T*A*Rf - Ri.T*A*Ri
                    phase = phase[0,0] /2
                phase -= floor(phase)
                hops.add(((hop[0][0],(hop[0][1],(int(j[0]),int(j[1])))), (hop[1][0],(hop[1][1],(int(t[0,0]),int(t[1,0])))), (z[0,0],z[1,0]), hop[3], phase))
        self.TBlattice = lattice
        self.lattice = superlattice
        self.atoms = atoms
        self.hops = hops
        self.Phi = Phi
        self.D = TBmodel.getD() if TBmodel.symbolic else None
        self.A = A
        self.T = T
        self.layers = layers
        self.symbolic = symbolic
        
    def duplicated(self):
        TBmodel = TB([[1,0],[0,1]],{},{},self.symbolic)
        model = MagneticTB(TBmodel, Phi=0)
        model.TBlattice = deepcopy(self.TBlattice)
        model.lattice = deepcopy(self.lattice)
        model.atoms = deepcopy(self.atoms)
        model.hops = deepcopy(self.hops)
        model.Phi = deepcopy(self.Phi)
        model.D = deepcopy(self.D)
        model.A = deepcopy(self.A)
        model.T = deepcopy(self.T)
        model.layers = deepcopy(self.layers)
        return model

    def Hamiltonian(self, k):
        k = sp.sympify(k)
        if self.symbolic == False:
            k = [float(x) for x in k]
        idx={}
        cnt = 0
        for atom in self.atoms:
            idx[atom] = cnt
            cnt = cnt+1
        H = [[0 for _ in range(cnt)] for _ in range(cnt)]
        for hop in self.hops:
            hop_term = - hop[3]*np.exp(-1j*float(2*pi*(k[0]*hop[2][0]+k[1]*hop[2][1])-2*pi*hop[4])) 
            H[idx[hop[0]]][idx[hop[1]]] = H[idx[hop[0]]][idx[hop[1]]] + hop_term
            H[idx[hop[1]]][idx[hop[0]]] = H[idx[hop[1]]][idx[hop[0]]] + np.conj(hop_term)
        return [H,idx]
            
    def symoff(self):
        self.TBlattice = np.matrix(self.TBlattice,dtype=float)
        self.lattice = np.matrix(self.lattice,dtype=float)
        self.atoms = {atom:[float(t) for t in self.atoms[atom]] for atom in self.atoms}
        self.hops = {(hop[0],hop[1],tuple(int(t) for t in hop[2]),hop[3], hop[4]) for hop in self.hops}
        self.Phi = float(self.Phi)
        self.D = np.matrix(self.D,dtype=float)
        self.A = np.matrix(self.A,dtype=float)
        self.T = np.matrix(self.T,dtype=float)
        self.symbolic = False
        return self
        
    def plot(self,save_dir=None):
        matrix = sp.Matrix if self.symbolic else np.matrix
        slinv = self.lattice.inv() if self.symbolic else np.linalg.inv(self.lattice)
        G = nx.DiGraph()
        xlim = [0,15]
        ylim = [0,10]
        xlist = []
        ylist = []
        for x in xlim:
            for y in ylim:
                idx = slinv*matrix([[x],[y]])
                xlist.append(idx[0])
                ylist.append(idx[1])
        xmin = int(min(xlist))
        xmax = int(max(xlist))
        ymin = int(min(ylist))
        ymax = int(max(ylist))

        half = sp.sympify('1/2') if self.symbolic else 0.5
        pos={}
        for idx in range(xmin-5,xmax+6):
            for idy in range(ymin-5,ymax+6):
                for (initial, final, delta, _, phase) in self.hops:
                    posi = matrix([self.atoms[initial][0:2]]).T+ self.lattice * matrix([[idx],[idy]])
                    posf = matrix([self.atoms[final][0:2]]).T+ self.lattice * (matrix([[idx],[idy]])+matrix([delta]).T)
                    if (posi[0]>=xlim[0] and posi[0]<=xlim[1] and posi[1]>=ylim[0] and posi[1]<=ylim[1]) or (posf[0]>=xlim[0] and posf[0]<=xlim[1] and posf[1]>=ylim[0] and posf[1]<=ylim[1]):
                        stri = (initial[0],(initial[1],(int(idx),int(idy))))
                        strf = (final[0],(final[1],int(idx+delta[0]),int(idy+delta[1])))
                        pos[stri]=(float(posi[0]),float(posi[1]))
                        pos[strf]=(float(posf[0]),float(posf[1]))
                        if phase==0 or phase==half:
                            G.add_weighted_edges_from([(stri,strf,2*pi*float(phase))])
                            G.add_weighted_edges_from([(strf,stri,2*pi*float(phase))])
                        elif phase<half:
                            G.add_weighted_edges_from([(stri,strf,2*pi*float(phase))])
                        else:
                            G.add_weighted_edges_from([(strf,stri,2*pi*float(1-phase))])
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        node_sizes = [1. for i in range(len(G))]
        M = G.number_of_edges()
        edge_colors = range(2, M + 2)
        edge_alphas = [1. for i in range(M)]
        cmap = plt.cm.Reds

        vmin = 0
        vmax = pi
        
        plt.figure(figsize=(10,5))
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=10,
            edge_color=weights,
            edge_cmap=cmap,
            width=2,
            edge_vmin = vmin,
            edge_vmax = vmax,
        )
        # set alpha value for each edge
        for i in range(M):
            edges[i].set_alpha(edge_alphas[i])

        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(weights)
        pc.set_clim(vmin,vmax)

        ax = plt.gca()
        ax.set_axis_off()
        plt.colorbar(pc, ax=ax)

        if save_dir != None:
            plt.savefig(save_dir,dpi=600)
        plt.show()

def Hofstadter_plot(TBmodel, q_max, k = [0,0], N= None, max_minute = 1,save_dir = None):
    start=time.time()
    E=[]
    B=[]
    if N == None:
        if TBmodel.D != None:
            N = 2* sp.Rational(TBmodel.D.det()).q
        elif TBmodel.symbolic:
            N = 2* sp.Rational(TBmodel.getD().det()).q
        else:
            N=1
    model = TBmodel.duplicated().symoff()
    for q in range(1,q_max+1):
        if(time.time()-start>=max_minute*60):
            break
        for p in range(0,N*q+1):
            if(time.time()-start>=max_minute*60):
                break
            if gcd(p,q)==1:
                H=MagneticTB(model, Phi=sp.Rational(p,q)).Hamiltonian(k)[0]
                Eigval=np.linalg.eigvalsh(H)
                E.extend(Eigval)
                B.extend([p/q for _ in range(len(Eigval))])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(B,E,s=0.1)
    ax.set_ylabel('$E$')
    ax.set_xlabel('$\Phi$')
    if save_dir != None:
        fig.savefig(save_dir)

def kE_plot(model,kroute = None, maxminute = 1, save_dir = None):
    model = model.duplicated().symoff()
    tstart=time.time()
    E=[]
    k=[]
    N=100
    k_nujeok=0
    record=[0]
    n=-1
    first= True
    for j in range(len(kroute)-1):
        [kxstart,kystart]=kroute[j][1]
        [kxend,kyend]=kroute[j+1][1]
        i=0
        while True:
            if i>=N:
                break
            if time.time()-tstart>60*maxminute:
                break
            k_nujeok=k_nujeok+sqrt(pow(kxend-kxstart,2)+pow(kyend-kystart,2))/N
            kx=kxstart*(N-i)/N+kxend*i/N
            ky=kystart*(N-i)/N+kyend*i/N
            if first:
                t1 = time.time()
            H=model.Hamiltonian([kx,ky])[0]
            Eigval=np.linalg.eigvalsh(H)
            if first:
                first = False
                t1 = time.time() - t1
                N = min(N, floor(60.0*maxminute/len(kroute)/max(t1,1e-10)))
            if n<0:
                n = len(Eigval)
            E.append(Eigval)
            k.append(k_nujeok)
            i=i+1
        if time.time()-tstart>60*maxminute:
            break
        record.append(k_nujeok)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    for i in range(n):
        ax.plot(k,[E[j][i] for j in range(len(E))])
    ax.set_ylabel('$E$')
    ax.set_xlabel('$k$')
    Emin=min([min(t) for t in E])
    Emax=max([max(t) for t in E])
    for k_ in record:
        ax.plot([k_,k_],[Emin,Emax],'--r',alpha=0.5,)
    ax.set_xticks(record, [kroute[i][0] for i in range(len(record))])
    if save_dir != None:
        fig.savefig(save_dir)

def plot_band(model, idx = None, mesh_num = 100, save_dir = None):
    model = model.duplicated().symoff()
    N = len(model.atoms)
    if idx == None:
        ni = 0
        nf = N
    elif isinstance(idx, int) and idx>=0 and idx<N:
        ni = idx
        nf = idx+1
    elif isinstance(idx,(tuple,list)) and len(idx)==2 and idx[0]>=0 and idx[0]<N and idx[1]>0 and idx[1]<=N and idx[1]>idx[0]:
        ni=idx[0]
        nf=idx[1]
    else:
        raise Exception('Only positive numbers or tuples and lists with len=2 can be given as an input.')
    n=mesh_num
    kx, ky = np.meshgrid(range(n), range(n))
    E=np.zeros((N,n,n))
    for i in range(n):
        for j in range(n):
            kx_ = i/n
            ky_ = j/n
            H = model.Hamiltonian([kx_,ky_])[0]
            E[:,i,j] = np.linalg.eigvalsh(H)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(ni,nf):
        ax.plot_surface(kx, ky, E[i,kx,ky], alpha=0.5)
    ax.set_xlabel('$k_1/|\mathbf{b}_1|$')
    ax.set_ylabel('$k_2/|\mathbf{b}_2|$')
    ax.set_zlabel('$E$')
    def normalize(value, tick_number):
        if value == 0:
            return "$0$"
        elif value == n-1:
            return "$1$"
        else:
            return f"${value / (n-1):.1g}$"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(normalize))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(normalize))
    if save_dir != None:
        fig.savefig(save_dir)

def Chern(v = None, model = None, idx_list = None, mesh_size = 30, gap_tolerance = 0.01):
    if (v== None and model == None) or (v!= None and model != None):
        raise Exception('Enter v or model, not both')
    if model == None and idx_list == None:
        raise Exception('idx_list need to be entered if v is given rather than model')
    if model != None:
        model = model.duplicated().symoff()
        N_=mesh_size
        N = len(model.atoms)
        v=np.zeros((N_,N_,N,N),dtype=np.complex_)
        if idx_list == None:
            ei = np.zeros((N_,N_,N))
        for ix in range(N_):
            for iy in range(N_):
                kx = ix/N_
                ky=iy/N_
                eig, ev = np.linalg.eigh(model.Hamiltonian([kx,ky])[0])
                v[ix,iy,:,:] = ev
                if idx_list == None:
                    ei[ix,iy,:]=eig
        if idx_list == None:
            i=0
            idx_list = []
            while i<N:
                j=i+1
                while j<N:
                    if (np.abs(ei[:,:,j]-ei[:,:,j-1]) >= gap_tolerance).all():
                        break
                    else:
                        j=j+1
                if j==i+1:
                    idx_list.append(i)
                else:
                    idx_list.append((i,j))
                i=j
    N1 = len(v)
    N2 = len(v[0])
    N = len(v[0][0])
    c=[]
    for i, idx in enumerate(idx_list):
        if isinstance(idx, int) and idx>=0 and idx<N:
            ni = idx
            nf = idx+1
        elif isinstance(idx,(tuple,list)) and len(idx)==2 and idx[0]>=0 and idx[0]<N and idx[1]>0 and idx[1]<=N and idx[1]>idx[0]:
            ni=idx[0]
            nf=idx[1]
        else:
            raise Exception('Only positive numbers or tuples and lists with len=2 can be given as an input.')
        c.append([])
        for x in range(N1):
            c[i].append([])
            for y in range(N2):
                v00 = np.matrix(v[x][y][:,ni:nf])
                v10 = np.matrix(v[(x+1)%N1][y][:,ni:nf])
                v01 = np.matrix(v[x][(y+1)%N2][:,ni:nf])
                v11 = np.matrix(v[(x+1)%N1][(y+1)%N2][:,ni:nf])
                v00c = np.conj(v00.T)
                v10c = np.conj(v10.T)
                v01c = np.conj(v01.T)
                v11c = np.conj(v11.T)
                t = np.linalg.det(v00c*v10)* np.linalg.det(v10c*v11)* np.linalg.det(v11c*v01)* np.linalg.det(v01c*v00)
                c[i][x].append(np.angle(t)/2/pi)
    C = np.sum(c,axis=(1,2))
    if np.allclose(C, np.round(C)):
        C = np.round(C)
    else:
        raise Exception('Wrong calculation : Increase mesh size or gap_tolerance')
    return [C,idx_list,[N1,N2,c]]

def moireTBG(t):
    if (t-1)%2!=0:
        raise Exception("Enter odd t")
    
    theta = sp.acos(sp.Rational(3*t*t-1,3*t*t+1))
    Bottom = Graphene.duplicated().extend([[-1-t, -2*t],[2*t, -1+t]]).translate([0,0,0])
    dict1 = dict()
    HoneyBottom = Honeycomb.duplicated().extend([[-1-t, -2*t],[2*t, -1+t]]).translate([0,0,0])
    for atom in Bottom.atoms:
        dict1[atom]=set()
    for init, final, delta, _ in HoneyBottom.hops:
        dict1[init].add( (final, delta) )
        dict1[final].add( (init, tuple(-x for x in delta)) )
    
    check = {atom: False for atom in Bottom.atoms}
    
    Top = Graphene.duplicated().extend([[1-t,-2*t],[2*t,1+t]]).rotate(angle=theta).translate([0,0,0.335/0.142])
    HoneyTop = Honeycomb.duplicated().extend([[1-t,-2*t],[2*t,1+t]]).rotate(angle=theta).translate([0,0,0.335/0.142])
    dict2 = dict()
    for (_,atom) in Top.atoms:
        dict2[(1,atom)]=set()
    for (_,init), (_,final), delta, _ in HoneyTop.hops:
        dict2[(1,init)].add( ((1,final), delta) )
        dict2[(1,final)].add( ((1,init), tuple(-x for x in delta)) )
    
    TBG = Bottom.merge(Top).symoff()
    
    q1 = queue.Queue()
    q2 = queue.Queue()
    q1.put( ((0, ('A',(0,0))) , (1, ('A',(0,0))), (0,0)) )
    while True:
        if q1.empty():
            break
        atom1, atom2, delta = q1.get()
        while check[atom1] and not q1.empty():
            atom1, atom2, delta = q1.get()
        if check[atom1]:
            break
        for atom,delta_ in dict1[atom1]:
            q1.put( (atom, atom2, tuple(a-b for a,b in zip(delta,delta_))) )
        d0 = np.matrix(delta) * np.matrix(TBG.lattice).T
        d = np.matrix([d0[0,0],d0[0,1],0]) + np.matrix(TBG.atoms[atom2])-np.matrix(TBG.atoms[atom1])
        t = graphene_hop([d[0,0],d[0,1],d[0,2]])
        if t == 0:
            for atom, delta_ in dict2[atom2]:
                d0 = np.matrix(tuple(a+b for a,b in zip(delta,delta_))) * np.matrix(TBG.lattice).T
                d = np.matrix([d0[0,0],d0[0,1],0]) + np.matrix(TBG.atoms[atom])-np.matrix(TBG.atoms[atom1])
                t = graphene_hop([d[0,0],d[0,1],d[0,2]])
                if t != 0:
                    pivot_atom = atom
                    pivot_delta = tuple(a+b for a,b in zip(delta,delta_))
                    break
        else:
            pivot_atom = atom2
            pivot_delta = delta
        chk = set()
        TBG.hops.add( (atom1, pivot_atom, pivot_delta,t) )
        chk.add( (pivot_atom, pivot_delta) )
        for atom, delta in dict2[pivot_atom]:
            q2.put( (atom, tuple(a+b for a,b in zip(pivot_delta,delta))))
        while not q2.empty():
            atom2, delta = q2.get()
            chk.add((atom2,delta))
            d0 = np.matrix(delta) * np.matrix(TBG.lattice).T
            d = np.matrix([d0[0,0],d0[0,1],0]) + np.matrix(TBG.atoms[atom2])-np.matrix(TBG.atoms[atom1])
            t = graphene_hop([d[0,0],d[0,1],d[0,2]])
            if t != 0:
                TBG.hops.add( (atom1, atom2, delta,t) )
                for atom, delta_ in dict2[atom2]:
                    atompair = (atom, tuple(a+b for a,b in zip(delta,delta_)))
                    if not (atompair in chk):
                        q2.put( atompair )
        check[atom1] = True
    return TBG
    
def graphene_hop(d):
    if len(d)==2:
        zratio = 0
        d = np.matrix([d[0],d[1]])
    else:
        zratio = d[2]*d[2]
        d = np.matrix([d[0],d[1],d[2]])
        zratio = zratio / (d*d.T)[0,0]
    dlen = sqrt(float((d*d.T)[0,0]))
    # print('dlen',dlen)
    if dlen >4:
        return 0
    delta = 0.184*sqrt(3)
    d0 = 0.335/0.142
    Vpi = 2.7 * exp(-(dlen-1)/delta)
    Vsigma = -4.8 * exp(-(dlen-d0)/delta)
    return Vpi*(1-zratio)+Vsigma*zratio

#Basic Crystals

Square = TB([[1,0],[0,1]], {'A':[0,0]}, {('A','A',(1,0),1),('A','A',(0,1),1)})
Triangle = TB([[1,0],['1/2','sqrt(3)/2']], {'A':[0,0]}, {('A','A',(1,0),1),('A','A',(0,1),1),('A','A',(1,-1),1)})
Honeycomb = TB([['3/2','sqrt(3)/2'],[0,'sqrt(3)']], {'A':[0,0],'B':[1,'sqrt(3)']}, {('B','A',(1,0),1),('B','A',(0,1),1),('B','A',(1,1),1)})
Kagome = TB([[2,0],[1,'sqrt(3)']], {'A':[0,0],'B':[1,0],'C':['1/2','sqrt(3)/2']}, {('A','B',(0,0),1),('A','C',(0,0),1),('B','C',(0,0),1),('B','A',(1,0),1),('B','C',(1,-1),1),('C','A',(0,1),1)})

Graphene = TB([['3/2','sqrt(3)/2'],[0,'sqrt(3)']], {'A':[0,0],'B':[1,'sqrt(3)']}, {})
chk = set()
for ix in range(-5,6):
    for iy in range(-5,6):
        d0 = np.matrix([ix,iy]) * np.matrix(Graphene.lattice).T
        if (d0*d0.T)[0,0]>(4+3)**2:
            continue
        for atom in Graphene.atoms:
            for atom2 in Graphene.atoms:
                if ix==0 and iy==0 and atom == atom2:
                    continue
                d = d0 + np.matrix(Graphene.atoms[atom2])-np.matrix(Graphene.atoms[atom])
                t = graphene_hop([d[0,0],d[0,1]])
                if t!=0 and ( (atom2,atom,-ix,-iy) in chk ) == False:
                    Graphene.hops.add( (atom, atom2, (ix,iy),t) )
                    chk.add( (atom,atom2,ix,iy) )
