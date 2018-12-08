#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent

from rl_glue import RLGlue
from env_hw6 import Environment
from tile3 import*
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d

def question_1():
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 5
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            # print(steps[r, e])
    np.save('steps', steps)

def question_3():
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)
    max_eps_steps = 100000
    num_episodes = 1000
    num_runs = 1
    numActions=3

    rlglue.rl_init()
    for e in range(num_episodes):
        rlglue.rl_episode(max_eps_steps)

    weights = rlglue.rl_agent_message("3D plot of the cast-to-go")

    fout = open('value','w')
    steps = 50
    z = np.zeros((50,50))
    for i in range(steps):
        for j in range(steps):
            values = []
            for a in range(numActions):
                tile = [8*(-1.2+(i*1.7/steps))/1.7,8*(-0.07+(j*0.14/steps))/0.14]
                inds =  agent.get_index(tile,a)
                values.append(np.sum([weights[i] for i in inds]))
            height = max(values)
            z[j][i]=-height
            fout.write(repr(-height)+' ')
        fout.write('\n')
    fout.close()

    fig = plt.figure()
    ax = fig.add_subplot(111,projection ='3d')
    x = np.arange(-1.2,0.5,1.7/50)
    y = np.arange(-0.07,0.07,0.14/50)
    x,y = np.meshgrid(x,y)
    ax.set_xticks([-1.2, 0.5])
    ax.set_yticks([0.07, -0.07])
    ax.set_ylabel('Velocity')
    ax.set_xlabel('Position')
    ax.set_zlabel('Cost-To-Go')
    ax.plot_surface(x,y,z)
    plt.savefig('cost-to-go.png')
    plt.show()
    np.save('steps', steps)

if __name__ == "__main__":
    #question_1()
    question_3()
    print("Done")
