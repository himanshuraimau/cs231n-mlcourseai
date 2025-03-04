import numpy as np

def generate_spiral_data(N=100, D=2, K=3):
    """
    Generate spiral data for classification
    Args:
        N: number of points per class
        D: dimensionality of the data
        K: number of classes
    Returns:
        X: data points (N*K, D)
        y: labels (N*K,)
    """
    X = np.zeros((N*K,D))  # Initialize data matrix
    y = np.zeros(N*K, dtype='uint8')  # Initialize labels
    
    for j in range(K):
        ix = range(N*j,N*(j+1))  # Index range for current class
        r = np.linspace(0.0,1,N)  # Radius: linear spacing from 0 to 1
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2  # Angle: 4 rotations + noise
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  # Convert from polar to cartesian coordinates
        y[ix] = j  # Assign labels
    
    return X, y
