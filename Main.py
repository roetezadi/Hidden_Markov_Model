import ClusteringKMeans as cluster
import Forward_Backward as fb
import numpy as np
import pandas as pd

def read_data():
    file = open('data/data_p1.txt')
    data = file.read().split()
    data = list(map(lambda x: int(x), data))
    return data


data = read_data()
df = cluster.K_Means(data)
# print(df[df['closest'] == 0].count())
tags = {'low':df['closest'][0], 'medium':df['closest'][3], 'high':df['closest'][12]}
print("low is: ", tags['low'])
print("medium is: ", tags['medium'])
print("high is: ", tags['high'])

print("initializing...")
T, E = fb.probabilities(df)
print("training...")
T, E = fb.hmm_train(df, T, E)
print("over")
print("Enter yout sequence:")
seq = 5
while True:
    a= []
    for i in range(seq+1):
        x = input()
        a.append(int(x))
    a = np.array(a)
    print(fb.hmm_test(a, T, E))
