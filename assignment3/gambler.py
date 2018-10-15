import numpy as np
import matplotlib.pyplot as plt

def value_iteration(goal,V,pi,ph):
    theta = 1e-100  #a small positive threshold
    while 1:
        delta = 0
        for state in range(1,goal): #loop for each s in States
            v = V[state]
            Evalue = [ph * V[state + action] + (1-ph) * V[state - action] for action in range(1,min(state, 100 - state) + 1)]
            V[state] = np.max(Evalue)
            pi[state] = np.argmax(Evalue)+1
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break


def main():
    goal = 100
    values = np.zeros(goal+1)
    values[100] = 1
    policy = np.zeros(goal+1)
    value_iteration(goal, values, policy, 0.25)

    plt.figure(1)
    plt.suptitle('ph = 0.25', fontsize=16)
    plt.subplot(2,1,1)
    plt.plot(values)
    plt.ylabel('Value\nestimates',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([0,.2,.4,.6,.8,1])
    plt.xticks([1,25,50,75,99])

    plt.subplot(2,1,2)
    plt.scatter(np.arange(goal+1),policy,s=7)
    plt.ylabel('Final\npolicy\n(stake)',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([1,10,20,30,40,50])
    plt.xticks([1,25,50,75,99])



    value_iteration(goal, values, policy, 0.55)
    plt.figure(2)
    plt.suptitle('ph = 0.55', fontsize=16)
    plt.subplot(2,1,1)
    plt.plot(values)
    plt.ylabel('Value\nestimates',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([0,.2,.4,.6,.8,1])
    plt.xticks([1,25,50,75,99])

    plt.subplot(2,1,2)
    plt.scatter(np.arange(goal+1),policy,s =7)
    plt.ylabel('Final\npolicy\n(stake)',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([1,10,20,30,40,50])
    plt.xticks([1,25,50,75,99])
    
    plt.show()



if __name__ == '__main__':
    main()
