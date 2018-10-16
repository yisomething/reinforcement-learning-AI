import numpy as np
import matplotlib.pyplot as plt

def value_iteration(goal,V,pi,ph):
    theta = 1e-100  #a small positive threshold
    sweep = 0
    while 1:
        delta = 0
        if sweep == 1:
            graph(V,ph,goal,pi,'blue')
        elif sweep == 2:
            graph(V,ph,goal,pi,'red')
        elif sweep == 3:
            graph(V,ph,goal,pi,'green')
        elif sweep == 32:
            graph(V,ph,goal,pi,'orange')

        for state in range(1,goal): #loop for each s in States
            v = V[state]
            Evalue = [ph * V[state + action] + (1-ph) * V[state - action] for action in range(1,min(state, 100 - state) + 1)]
            V[state] = np.max(Evalue)
            pi[state] = np.argmax(Evalue)+1
            delta = max(delta, abs(v - V[state]))
        sweep += 1
        if delta < theta:
            break
    graph(V,ph,goal,pi,'black')

def graph(V,ph,goal,pi,color):
    plt.suptitle('ph = '+str(ph),fontsize=16)
    plt.subplot(2,1,1)
    plt.ylabel('Value\nestimates',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([0,.2,.4,.6,.8,1])
    plt.xticks([1,25,50,75,99])
    plt.plot(V,color = color)


def main():

    goal = 100
    values = np.zeros(goal+1)
    values[100] = 1
    policy = np.zeros(goal+1)
    ph = [0.25,0.55]
    value_iteration(goal, values, policy, ph[0])
    #value_iteration(goal, values, policy, ph[1])

    plt.show()



if __name__ == '__main__':
    main()
