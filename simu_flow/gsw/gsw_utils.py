import numpy as np
import torch
import ot
from sklearn.datasets import make_swiss_roll, make_circles


def w2(X,Y):
    M=ot.dist(X,Y)
    a=np.ones((X.shape[0],))/X.shape[0]
    b=np.ones((Y.shape[0],))/Y.shape[0]
    return ot.emd2(a,b,M)


def load_data(name='swiss_roll', n_samples=1000):
    N=n_samples
    if name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == '25gaussians':
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'swiss_roll', 'circle' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X
