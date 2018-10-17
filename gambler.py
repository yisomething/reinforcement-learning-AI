import numpy as np
import matplotlib.pyplot as plt
def value_iteration(goal,V,pi,ph):
    theta = 1e-100  #a small positive threshold
    sweep = 0
    while 1:
        delta = 0
        if sweep == 1:
            draw_value(V,'blue')
        elif sweep == 2:
            draw_value(V,'red')
        elif sweep == 3:
            draw_value(V,'green')
        elif sweep == 32:
            draw_value(V,'orange')

        for state in range(1,goal): #loop for each s in States
            v = V[state]
            Evalue = [ph * V[state + action] + (1-ph) * V[state - action] for action in range(1,min(state, 100 - state) + 1)]
            V[state] = np.max(Evalue)
            pi[state] = np.argmax(Evalue)+1
            delta = max(delta, abs(v - V[state]))
        sweep += 1
        if delta < theta:
            break
    draw_value(V,'grey')
    draw_policy(goal,pi)


def graph(i,ph):
    plt.figure(i)
    plt.suptitle('ph = '+str(ph),fontsize=16)
    plt.subplot(2,1,1)
    plt.ylabel('Value\nestimates',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([0,.2,.4,.6,.8,1])
    plt.xticks([1,25,50,75,99])
    plt.text(85,0.4,'----sweep 1',color = 'blue',fontsize=10)
    plt.text(85,0.3,'----sweep 2',color = 'red',fontsize=10)
    plt.text(85,0.2,'----sweep 3',color = 'green',fontsize=10)
    plt.text(85,0.1,'----sweep 32',color = 'orange',fontsize=10)
    plt.text(85,0,'----Final value function',color = 'grey',fontsize=10)


def draw_value(V,color):
    plt.plot(V,color = color)


def draw_policy(goal,policy):
    plt.subplot(2,1,2)
    plt.scatter(np.arange(goal+1),policy,s=7)
    plt.ylabel('Final\npolicy\n(stake)',rotation = 0)
    plt.xlabel('Captial')
    plt.yticks([1,10,20,30,40,50])
    plt.xticks([1,25,50,75,99])
    plt.show()


def main():
    goal = 100
    values = np.zeros(goal+1)
    values[100] = 1
    policy = np.zeros(goal+1)
    ph = [0.25,0.55]
    graph(1,ph[0])
    value_iteration(goal, values, policy, ph[0])
    graph(2,ph[1])
    value_iteration(goal, values, policy, ph[1])


if __name__ == '__main__':
    main()
