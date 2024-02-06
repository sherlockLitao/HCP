import numpy as np
from scipy.sparse import coo_matrix
import base
from scipy.stats import ortho_group
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn import metrics
import scipy as sp
from hilbertcurve.hilbertcurve import HilbertCurve



#######################################################
def trans(X, k):

    Xs = (X-np.min(X,0))/(np.max(X,0)-np.min(X,0))
    Xs = Xs*(2**k)
    Xs = np.int64(Xs)
    Xs[Xs==(2**k)] = 2**k-1

    return Xs

## equal weights HCP
def HCP(X, Y, hc='Hm', k=5):

    """
    Implement Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        hc : string, impementation of the Hilbert curve, either 'Hm' or 'Hk'. 'Hm' is based on recursive sort, and 'Hk' is based on Hilbert indices. For large n, Hm is recommended.
        k : int, order of Hilbert curve when hc='Hk'

    Returns
    ----------
        float, Hilbert curve projection distance (p=2)
    """

    n,d = X.shape

    assert (hc == 'Hm' or hc == 'Hk'), \
        "Hilbert curve should only be Hm or Hk"

    if hc=='Hm':
        # recursive sort
        Xr = base.hilbert_order(X)
        Yr = base.hilbert_order(Y)
    else:
        # Hilbert indices based sort
        p=k
        hilbert_curve = HilbertCurve(p, d)

        Xi = trans(X, k)
        Yi = trans(Y, k)
        Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*p) - 1)
        Xr = np.argsort(Xdistances)
        Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*p) - 1)
        Yr = np.argsort(Ydistances)

    res = np.sum(((X[Xr,:]-Y[Yr,:])**2))/n

    return np.sqrt(res)





#######################################################
def general_plan(a=None, b=None, dense=False):

    GI = base.general_Plan(
        a.astype(np.float64),
        b.astype(np.float64)).T

    if dense:
        G = coo_matrix(
        (GI[:,0], (GI[:,1], GI[:,2])),
        shape=(a.shape[0], b.shape[0]))

        G = coo_matrix.todense(G)
        G = np.array(G)
        return G
   
    return GI

## unequal weights HCP
def gHCP(X, Y, a, b, is_plan=False, hc='Hm', k=3):

    """
    Implement Hilbert curve projection distance (p=2) for unequal sample size and unequal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (m, d), samples in the target domain
        a : array-like, shape (n), samples weights in the source domain
        b : array-like, shape (m), samples weights in the target domain
        is_plan : bool, True return transport plan, False return HCP distance
        hc : string, impementation of the Hilbert curve, either 'Hm' or 'Hk'. 'Hm' is based on recursive sort, and 'Hk' is based on Hilbert indices. For large n, Hm is recommended.
        k : int, order of Hilbert curve when hc='Hk'

    Returns
    ----------
        float, Hilbert curve projection distance (p=2) for unequal sample size and unequal weights
    """

    n,d = X.shape

    assert (hc == 'Hm' or hc == 'Hk'), \
        "Hilbert curve should only be Hm or Hk"

    if hc=='Hm':
        Xr = base.hilbert_order(X)
        Yr = base.hilbert_order(Y)
    else:
        p=k
        hilbert_curve = HilbertCurve(p, d)

        Xi = trans(X, k)
        Yi = trans(Y, k)
        Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*p) - 1)
        Xr = np.argsort(Xdistances)
        Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*p) - 1)
        Yr = np.argsort(Ydistances)

    aa = a[Xr]
    bb = b[Yr]

    if is_plan:
        G = general_plan(aa,bb,dense=True)
        ix = np.argsort(Xr)
        iy = np.argsort(Yr)

        res = G[ix,:]
        res = res[:,iy]
        return res

    else:
        GI = general_plan(aa,bb)
        # res = 0

        # for i in range(GI.shape[0]):
        #     res+=np.sum((X[Xr[int(GI[i,1])],:]-Y[Yr[int(GI[i,2])],:])**2)*GI[i,0]
        res = np.sum((X[Xr[GI[:, 1].astype(int)], :] - Y[Yr[GI[:, 2].astype(int)], :])**2 * GI[:, [0]])

        return np.sqrt(res)





#######################################################
def rand_projections(embedding_dim, num_samples=50):
    
    projections = [w / np.sqrt((w**2).sum())
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)

    return projections

## equal weights IPRHCP
def IPRHCP(X, Y, q=2, nslice=50, direction='ortho'):

    """
    Implement integral projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        q : int, dimension of subspace
        nslice : int, number of slices 
        direction : string, kind of the sliced direction, either 'ortho' or 'random'. 'ortho' means orthogonal directions, and 'random' means directions are independent and uniform to sphere.

    Returns
    ----------
        float, integral projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights
    """

    assert (direction == 'ortho' or direction == 'random'), \
        "Direction should only be orthogonal or random"
    
    d = X.shape[1]
    res = 0

    if direction=='ortho':
        ## to generate less orthogonal matrix(here we need d/q is integer)
        k = int(q*nslice/d)+1
        projM = np.zeros((d,d*k))
        for j in range(k):
            projM[:,(j*d):(j*d+d)] = ortho_group.rvs(dim=d)

        for i in range(nslice):
            proj = projM[:,(i*q):(i*q+q)]
            Xi = X@proj
            Yi = Y@proj
            res += HCP(Xi, Yi)**2

    else:
        ## random directions may be faster
        proj = rand_projections(d, nslice*q)
        Xp = X@proj.T
        Yp = Y@proj.T

        for i in range(nslice):
            Xi = Xp[:,(i*q):(i*q+q)]
            Yi = Yp[:,(i*q):(i*q+q)]
            res += HCP(Xi, Yi)**2

    return np.sqrt(res/nslice)

## unequal weights IPRHCP
def gIPRHCP(X, Y, a, b, q=2, nslice=50, direction='ortho'):

    """
    Implement integral projection robust Hilbert curve projection distance (p=2) for unequal sample size and unequal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (m, d), samples in the target domain
        a : array-like, shape (n), samples weights in the source domain
        b : array-like, shape (m), samples weights in the target domain
        q : int, dimension of subspace
        nslice : int, number of slices 
        direction : string, kind of the sliced direction, either 'ortho' or 'random'. 'ortho' means orthogonal directions, and 'random' means directions are independent and uniform to sphere.

    Returns
    ----------
        float, integral projection robust Hilbert curve projection distance (p=2) for unequal sample size and unequal weights
    """

    assert (direction == 'ortho' or direction == 'random'), \
        "Direction should only be orthogonal or random"
    
    d = X.shape[1]
    res = 0

    if direction=='ortho':
        ## to generate less orthogonal matrix(here we need d/q is integer)
        k = int(q*nslice/d)+1
        projM = np.zeros((d,d*k))
        for j in range(k):
            projM[:,(j*d):(j*d+d)] = ortho_group.rvs(dim=d)

        for i in range(nslice):
            proj = projM[:,(i*q):(i*q+q)]
            Xi = X@proj
            Yi = Y@proj
            res += gHCP(Xi, Yi, a, b)**2

    else:
        proj = rand_projections(d, nslice*q)
        Xp = X@proj.T
        Yp = Y@proj.T

        for i in range(nslice):
            Xi = Xp[:,(i*q):(i*q+q)]
            Yi = Yp[:,(i*q):(i*q+q)]
            res += gHCP(Xi, Yi, a, b)**2

    return np.sqrt(res/nslice)





#######################################################
## equal weights IPRHCP
def PRHCP(X, Y, q=2, niter=10):

    """
    Implement projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        q : int, dimension of subspace
        niter : int, maximum number of iterations 

    Returns
    ----------
        float, projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights
    """
    
    n, d = X.shape
    t = 0
    tau = 1
    i = 0
    Omega = np.eye(d)

    while i<niter:
        if i==0:
            Xr = base.hilbert_order(X)
            Yr = base.hilbert_order(Y)
            delta = X[Xr,:]-Y[Yr,:]
        else:
            Xu = X@U
            Yu = Y@U
            Xr = base.hilbert_order(Xu)
            Yr = base.hilbert_order(Yu)
            delta = X[Xr,:]-Y[Yr,:]
        
        disp = np.concatenate([delta, -delta])    
        pca = PCA(n_components = q, random_state = 1)
        pca.fit(disp)
        U = pca.components_.T
        Omega = (1-tau)*Omega + tau*U@U.T
        eigenvalues, eigenvectors = sp.linalg.eigh(Omega, eigvals=(d-q,d-1))
        U = eigenvectors
        t += 1
        tau = 2/(2+t)
        i += 1
    
    distance = HCP(X@U,Y@U)

    return U, distance

## unequal weights IPRHCP
def gPRHCP(X, Y, a, b, q=2, niter=10):

    """
    Implement projection robust Hilbert curve projection distance (p=2) for unequal sample size and unequal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (m, d), samples in the target domain
        a : array-like, shape (n), samples weights in the source domain
        b : array-like, shape (m), samples weights in the target domain
        q : int, dimension of subspace
        niter : int, maximum number of iterations 

    Returns
    ----------
        float, projection robust Hilbert curve projection distance (p=2) for unequal sample size and unequal weights
    """

    n, d = X.shape
    t = 0
    tau = 1
    i = 0
    a1 = np.sqrt(a.reshape((n,1)))
    Omega = np.eye(d)

    while i<niter:
        if i==0:
            G = gHCP(X, Y, a, b, is_plan=True)
            ida = a>0
            G[ida,:] = (G[ida,:].T/a[ida]).T
            ## weighted PCA
            delta = (X - G@Y)*a1
        else:
            Xu = X@U
            Yu = Y@U
            G = gHCP(Xu, Yu, a, b, is_plan=True)
            ida = a>0
            G[ida,:] = (G[ida,:].T/a[ida]).T
            ## weighted PCA
            delta = (X - G@Y)*a1
        
        disp = np.concatenate([delta, -delta])    
        pca = PCA(n_components = q, random_state = 1)
        pca.fit(disp)
        U = pca.components_.T
        Omega = (1-tau)*Omega + tau*U@U.T
        eigenvalues, eigenvectors = sp.linalg.eigh(Omega, eigvals=(d-q,d-1))
        U = eigenvectors
        t += 1
        tau = 2/(2+t)
        i += 1
    
    distance = gHCP(X@U,Y@U,a,b)

    return U, distance
