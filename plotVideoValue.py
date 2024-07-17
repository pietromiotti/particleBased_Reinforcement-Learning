import random
from env import Env
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from dqn import DQN
from ppo import ActorCritic

DISCOUNT_FACTOR = 0.999
random.seed()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


'''
CHOOSE THE ACTIVITY THAT YOU WANT TO PERFORM: 
    print value fuction -> activity =0
    plot real time -> activity =1
    save video -> activity =2
'''
value_func = 1


env = Env(continuous=False)

'''
    CHOOSE THE MODEL that you want to check, make sure to 
'''
''' 1 door 0 obstacle '''
dqn = 0
policy = ActorCritic(env.num_state, env.num_action)
policy.load_state_dict(torch.load('./ppo_state_1_door_0_obstacle'))

#dqn = 1
#policy = DQN(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./netstate_dqn_1_door_0_obstacle'))

lenghtWall_up=2
lenghtWall_bottom=0
obstacleLenght = 0

''' 1 door 1 obstacle'''

#dqn = 0
#policy = ActorCritic(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./ppo_state_1_door_1_obstacle'))

#dqn = 1
#policy = DQN(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./netstate_dqn_1_door_1_obstacle'))

#lenghtWall_up=2
#lenghtWall_bottom=0
#obstacleLenght = 2

''' 2 door 0 obstacle'''
#dqn = 0
#policy = ActorCritic(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./ppo_state_2_door_0_obstacle'))

#dqn = 1
#policy = DQN(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./netstate_dqn_2_doors_0_obstacle'))

#lenghtWall_up=2
#lenghtWall_bottom=2
#obstacleLenght = 0

''' 2 door 1 obstacle'''
#dqn = 0
#policy = ActorCritic(env.num_state, env.num_action)
#policy.load_state_dict(torch.load('./ppo_state_2_door_1_obstacle'))

#dqn = 1
#policy = DQN(env.num_state,env.num_action)
#policy.load_state_dict(torch.load('./netstate_dqn_2_doors_1_obstacle'))

#lenghtWall_up=2
#lenghtWall_bottom=2
#obstacleLenght = 2

#reset the environment with the desired parameters
env = Env(continuous=False, lenghtWall_up= lenghtWall_up, lenghtWall_bottom=lenghtWall_bottom, obstacleLenght=obstacleLenght)

x = np.arange(env.WALL_LEFT, env.WALL_RIGHT, 0.5)
y = np.arange(env.WALL_LEFT, env.WALL_RIGHT, 0.5)

state = env.reset()
done = False
totreward = 0

def update(i):
    global state
    global done
    global ani
    global writer
    state = torch.from_numpy(state.astype(np.float32))

    if(dqn):
        q_actions = policy(state)

        #print(q_actions)
        action = torch.argmax(q_actions).item()

        state, reward, done, info = env.step(action)
    else:
        action, dist = policy.act(state)
        state, reward, done, info = env.step(action.numpy())

    scatter.set_offsets(np.c_[env.pos[0, :], env.pos[1, :]])

    if(done):
        anim.event_source.stop()
        exit(1)


if(value_func==0):
    xx = np.arange(env.WALL_LEFT, env.WALL_RIGHT, 0.01)
    yy = np.arange(env.WALL_LEFT, env.WALL_RIGHT, 0.01)

    mystates_2 = np.zeros(shape=((len(xx)) ** 2, env.num_state))
    for i in range(len(xx)):
        for j in range(len(yy)):
            mystates_2[i * len(xx) + j, 0] = xx[i]
            mystates_2[i * len(xx) + j, 1] = yy[j]
            mystates_2[i * len(xx) + j, 2] = 0
            mystates_2[i * len(xx) + j, 3] = 0

    mystates_2 = torch.from_numpy(mystates_2.astype(np.float32))

    if(dqn):
        output = ((policy(mystates_2)).detach()).numpy()
        output = np.amax(output, axis=1)
        output = output.reshape(-1, xx.shape[0])
    else:
        output = ((policy.critic(mystates_2)).detach()).numpy()
        output = output.reshape(-1,xx.shape[0])

    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, output, color='black')
    plt.show()

elif (value_func==1):

    for i in range(1000):

        state = torch.from_numpy(state.astype(np.float32))
        if(dqn):
            q_actions = policy(state)
            action = torch.argmax(q_actions).item()
            state, reward, done, info = env.step(action)
        else:
            action, dist = policy.act(state)
            state, reward, done, info = env.step(action.numpy())



        plt.scatter(env.pos[0, :], env.pos[1, :], 50, env.total_particles[:])
        totreward += reward * (DISCOUNT_FACTOR ** i)
        plt.show(block=False)
        plt.pause(0.00001)
        plt.close()
        if(done):
            print('done', totreward)

elif(value_func==2):
    fig = plt.figure()
    ax = plt.axes(xlim=(0,env.L-1), ylim=(0,env.L-1))
    scatter = ax.scatter(env.pos[0, :], env.pos[1, :], 50, env.total_particles[:])

    #return ax
    anim = FuncAnimation(fig, update,  interval=10)
    anim.save('./particle_clever.mp4', dpi=200, writer=writer)
    plt.show()



