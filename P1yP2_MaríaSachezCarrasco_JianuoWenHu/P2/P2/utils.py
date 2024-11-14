import numpy as np
import pandas as pd


def cleanData(data):
    data['score'] = pd.to_numeric(data['score'], errors="coerce")
    data['users'] = pd.to_numeric(data['users'], errors="coerce")
    data['critics'] = pd.to_numeric(data['critics'], errors="coerce")
    data['user score'] = pd.to_numeric(data['user score'], errors="coerce")
    data.dropna(subset = ['score', 'users', 'critics', 'user score'], inplace=True)
    data["score"] = data["score"].astype(np.float64)    
    data["users"] = data["users"].astype(np.float64)    
    data["critics"] = data["critics"].astype(np.float64)    
    data["user score"] = data["user score"].astype(np.float64)    
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    return X, y

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    #X_norm, mu, sigma = 0

    X_norm = np.empty((0, X.shape[0]));
    mu=[]
    sigma=[]
    aux=0
    for c in X.T:
        mu.append(np.mean(X))
        sigma.append(np.std(X))
        a = (c-mu[aux]/sigma[aux])
        X_norm = np.vstack([X_norm,a])
        aux+=1
    
    return X_norm.T, mu, sigma

def load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    X = np.array([x1, x2, x3])
    X = X.T
    y = data[y_colum].to_numpy()
    return X, y