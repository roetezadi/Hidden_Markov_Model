import numpy as np

def probabilities(data):
    N = len(data)
    # transition matrix
    # T = np.random.random((N, N))
    # T *= 0.01
    T = np.zeros((N,N))
    for i in range(N):
        T[i,:] = np.random.dirichlet(np.ones(N), size=1)
    # emission matrix
    # E = np.random.random((N, 3))
    E = np.ones((N, 3))
    E *= 0.33
    # E = np.zeros((N, 3))
    # for i,x in enumerate(data['closest']):
    #     E[i][x] = 1
    return T, E

def hmm_train(df, T, E):
    N = len(df)
    Time = 193
    alpha = np.zeros((N, Time))
    beta = np.ones((N, Time))
    epsilon = np.zeros((Time, N, N))
    gamma = np.zeros((N, Time))
    delta = np.zeros((N, Time))

    begin_index = 0
    end_index = N
    sequence = 5

    # initialization
    alpha[0][0] = E[0][0]*1
    alpha[0][1] = E[0][1]*0
    alpha[0][2] = E[0][2]*0
    delta[0][0] = E[0][0]*1
    delta[0][1] = E[0][1]*0
    delta[0][2] = E[0][2]*0
    iteration = 0
    # while (begin_index + sequence) < N:
    while iteration < 10:
        print(iteration)
        # forward
        for t in range(Time):
            if t != 0:
                for i in range(N):
                    alpha[i][t] = E[i][df['closest'][t]]*(np.dot(T[:,i], alpha[:, t-1].T))

        # print(alpha[0][4])

        # backward
        for t in reversed(range(t)):
            if t+1 < Time:
                for i in range(N):
                    beta[i][t] = np.dot(T[i, :],(np.multiply(beta[:,t+1], E[:, df['closest'][t+1]])))

        # epsilon
        for t in range(Time):
            for i in range(N):
                for j in range(N):
                    tmp = (np.dot(alpha[:, t], beta[:, t].T))
                    if tmp!=0:
                        epsilon[t][i][j] = (alpha[i, t]*beta[i, t]*T[i, j])/tmp

        # gamma
        for i in range(N):
            for t in range(Time):
                gamma[i][t] = np.sum(epsilon[t,i,:])

        # delta
        for t in range(Time):
            if t != 0:
                for i in range(N):
                    delta[i][t] = E[i][df['closest'][t]]*np.max((np.dot(T[:,i], delta[:, t-1].T)))

        # update T and E
        for i in range(N):
            for j in range(N):
                T[i, j] = (np.sum(epsilon[:, i, j]))/(np.sum(gamma[i, :]))
        for t in range(Time):
            for i in range(N):
                E[t, df['closest'][i]] = (np.dot(delta[i, :], gamma[i, :].T))/(np.sum(gamma[i, :]))

        begin_index += 1
        iteration += 1

    return T, E

def hmm_training2(df, T, E):
    N = len(df)
    Time = 5
    alpha = np.zeros((N, Time))
    beta = np.ones((N, Time))
    epsilon = np.zeros((Time, N, N))
    gamma = np.zeros((N, Time))
    delta = np.zeros((N, Time))

    begin_index = 0
    end_index = N
    sequence = 5

    # initialization
    alpha[0][0] = E[0][0] * 1
    alpha[0][1] = E[0][1] * 0
    alpha[0][2] = E[0][2] * 0
    delta[0][0] = E[0][0] * 1
    delta[0][1] = E[0][1] * 0
    delta[0][2] = E[0][2] * 0
    iteration = 0
    while (begin_index + sequence) < N:
        # print(iteration)
        # forward
        for t in range(Time):
            if t != 0:
                for i in range(N):
                    alpha[i][t] = E[i][df['closest'][t+begin_index]] * (np.dot(T[:, i], alpha[:, t - 1].T))

        print(alpha[0][4])

        # backward
        for t in reversed(range(t)):
            if t + 1 < Time:
                for i in range(N):
                    beta[i][t] = np.dot(T[i, :], (np.multiply(beta[:, t + 1], E[:, df['closest'][t + 1 +begin_index]])))

        # epsilon
        for t in range(Time):
            for i in range(N):
                for j in range(N):
                    tmp = (np.dot(alpha[:, t], beta[:, t].T))
                    if tmp != 0:
                        epsilon[t][i][j] = (alpha[i, t] * beta[i, t] * T[i, j]) / tmp

        # gamma
        for i in range(N):
            for t in range(Time):
                gamma[i][t] = np.sum(epsilon[t, i, :])

        # delta
        for t in range(Time):
            if t != 0:
                for i in range(N):
                    delta[i][t] = E[i][df['closest'][t+begin_index]] * np.max((np.dot(T[:, i], delta[:, t - 1].T)))

        # update T and E
        for i in range(N):
            for j in range(N):
                T[i, j] = (np.sum(epsilon[:, i, j])) / (np.sum(gamma[i, :]))
        for t in range(Time):
            for i in range(N):
                E[t, df['closest'][i+begin_index]] = (np.dot(delta[i, :], gamma[i, :].T)) / (np.sum(gamma[i, :]))

        begin_index += 1
        iteration += 1
    return T, E

def hmm_test(df, T, E):
    # print(T)
    N = len(df)
    Time = 5
    Threshold = 0.9
    alpha = np.zeros((N, Time))
    alpha[0][0] = E[0][0]*1
    alpha[0][1] = E[0][1]*0
    alpha[0][2] = E[0][2]*0
    begin_index = 0
    # forward
    for t in range(Time):
        if t != 0:
            for i in range(N):
                alpha[i][t] = E[i][df[t+begin_index]] * (np.dot(T[0:6, i], alpha[:, t - 1].T))
    alpha1 = alpha[N-1][Time-1]
    begin_index += 1
    alpha = np.zeros((N, Time))
    alpha[0][0] = E[0][0] * 1
    for t in range(Time):
        if t != 0:
            for i in range(N):
                alpha[i][t] = E[i][df[t+begin_index]] * (np.dot(T[0:6, i], alpha[:, t - 1].T))
    alpha2 = alpha[N-1][Time-1]

    state = ''
    if alpha1 != 0:
        r = (alpha1 - alpha2)/alpha1
    print(alpha1)
    print(alpha2)
    print(np.abs(alpha1 - alpha2))
    print(np.abs(r))
    if np.abs(r) < Threshold:
        state = "normal"
    else:
        state = "fraud"
    return state

