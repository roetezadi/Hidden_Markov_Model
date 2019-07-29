import copy
import numpy as np
import pandas as pd
from numpy.core.defchararray import lstrip

K = 3

def manhatan_distance(num1, num2):
    return np.abs(num1 - num2)

def assignment(df, centers):
    for i in centers.keys():
        df['distance_from_{}'.format(i)] = (manhatan_distance(df['x'], centers[i]))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centers.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df

def update(df, centroid):
    for i in centroid.keys():
        centroid[i] = np.mean(df[df['closest'] == i]['x'])
    return centroid

def K_Means(numbers):
    df = pd.DataFrame({'x': numbers})
    N = len(numbers)

    # defining centers for clusters
    # centers[i] = x
    centers = {
        i: numbers[np.random.randint(0,N-1)]
        for i in range(K)
    }

    old_centers = copy.deepcopy(centers)
    df = assignment(df, old_centers)
    new_centers = update(df, old_centers)

    while True:
        closest_centroid = df['closest'].copy(deep=True)
        old_centers = copy.deepcopy(new_centers)
        df = assignment(df, old_centers)
        new_centers = update(df, old_centers)
        if closest_centroid.equals(df['closest']):
            break

    return df.loc[:, ['x','closest']]


# numbers = [2,3,88,5,123,10,88]
# print(K_Means(numbers))
# df = pd.DataFrame({'x': numbers})
# print(df)
# centers_index = {
#         i: numbers[np.random.randint(0,6)]
#         for i in range(K)
#     }
# print("centers:", centers_index)
# df = assignment(df, centers_index)
# print(df)
# print(update(centers_index))