# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
mpl.rc('animation', html='jshtml')

#%% CartPole egyszerű verziója
env = gym.make('CartPole-v1')
obs = env.reset()
print(obs)
print(env.action_space)
env.render()

print(env.action_space)

obs, reward, done, info = env.step(1)

print(obs)
print(reward)
print(done)
print(info)

for i in range(10):
    env.step(1)
    env.render()

env.close()

#%% CartPole egyszerű Hardcode szabályrendszerrel
import gym
env = gym.make('CartPole-v1')

def basic_policy(obs): # jutalmak 
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500): # 500 iteráció
    episode_rewards = 0 # kezdeti jutalom
    obs = env.reset() # környezet init
    for step in range(200): # 200 válasz iterációnként
        action = basic_policy(obs) # jelenlegi szög
        # env.render()
        obs, reward, done, info = env.step(action) # válasz
        episode_rewards += reward # jutalmak mentése
        if done: # kilépés
            break
    totals.append(episode_rewards) 

print('átlag', np.mean(totals))
print('szórás', np.std(totals))
print('min', np.min(totals))
print('max', np.max(totals))
env.close()


#%% Vizualizálási függvények
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


#%% Vizualizáljunk egy epizódot!
env.seed(42)

frames = []
obs = env.reset()
for step in range(200):
    img = env.render(mode="rgb_array")
    frames.append(img)
    action = basic_policy(obs)

    obs, reward, done, info = env.step(action)
    if done:
        env.close()
        break
    
plot_animation(frames)
