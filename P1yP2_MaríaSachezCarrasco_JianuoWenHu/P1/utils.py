import numpy as np
import pandas as pd

def cleanData(data):
    #TO-DO clean the dataframe. Return the dataframe cleaned.
    data = data.drop(data[data['user score'] == 'tbd'].index)
    data['score'] = data['score'].astype(np.float32)
    data['user score'] = data['user score'].astype(np.float32)
    data['score'] = data['score'].apply(lambda x: x/10)
    
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    #display(data)
    return X, y