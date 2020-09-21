import csv
import datetime
import json, os, shutil, sys
import os
import psutil
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import gym
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds

#Configure tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#Set working directory
checkpoint_root = r"C:\Users\Alexandre\Desktop\Project"
wd=os.chdir(checkpoint_root)

#load data into dataframe
df = pd.read_csv('./data/data.csv')

#set my evaluation function
def evaluate(model, env,n_cpu,graph = False):
    
    score = []
    score_last = []
    actions = []
    
    #to compute the time elapsed
    t0 = time.time()
    
    #let say 10 episodes (or 100) to compute the mean and the standard deviation at the end
    episodes=10
    
    for e in range(episodes):
        obs = env.reset()
        
        #For lstm in Stable Baseline with multi env/cpu (subprocvecenv)
        state = None
        done = [False for _ in range(env.num_envs)]
      
        while True:
          
            action, _states = model.predict(obs,state=state, mask=done)

            obs, rewards, dones, infos = env.step(action)

            actions.append(infos[0]["action"])
            
            #to plot the score later
            score.append(infos[0]["Score"])
            
            if dones[0]:
                break
        #append the end score to the score_last list
        score_last.append(score[-1])

    #once episodes end, compute the average score
    average = sum(score_last)/len(score_last)
    
    #optional : to compute the time elapsed
    delay = time.time() - t0 
    
    print("Evaluation took: " , round(delay,1),  " seconds, i.e ", round(delay/60,2), " minutes." )
    print("Final Score= ",str(round(score[-1],2))+ 'Pts')
    print("Average net worth= ",str(round(average,2))+ 'Pts') 
    
    if graph == True:
        pd.DataFrame(score).plot(title= (" Average score: "+str(round(average,2))+ 'Pts'))
    
        plt.show()
    
    return round(score[-1],2)

#standard function from Standard Baseline to create env
def make_env(env_id, rank, config, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        time.sleep(0.5)
        env = gym.make(env_id,env_config = config)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    time.sleep(0.5)
    return _init


#enter the parameters you want here (the more cpu the more parallel environments generated)
num_cpu = 6
#the batchsize (num_cpu%nminibatches == 0)
nminibatches=1


#the configuration of the env if any
config = {
         'df': df,"use_lstm": True,  'reward_func': "CustomRewardFunction_v3","n_cpu": num_cpu
          }

env_id = 'CustomEnv-v1'

train_env = SubprocVecEnv([make_env(env_id, i,config)  for i in range(num_cpu)])

#the parameters of the lstm used
config = {
  "learning_rate": 0.00001,
  "policy": "MlpLnLstmPolicy", #MlpLnLstmPolicy
  "policy_kwargs": {
    "net_arch": [
      128,
      128,
      128,
      128,
      
       "lstm",
    ]
  }
}

#initialize a model
model = PPO2(config["policy"],train_env, verbose=0, nminibatches=nminibatches,policy_kwargs=config["policy_kwargs"],learning_rate = config["learning_rate"])
    
#if a saved model exist with the configuration above, load it.
if os.path.exists(checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(num_cpu) + "_nminibatches"+str(nminibatches)+".zip"):
    model = PPO2.load(checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(num_cpu) + "_nminibatches"+str(nminibatches)+".zip",env=train_env,policy=MlpLnLstmPolicy,verbose=0)
    print("*"*100)
    print("Model loaded from:",checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(num_cpu) + "_nminibatches"+str(nminibatches)+".zip")
    print("*"*100)


#launch the training and evaluation
episodes = 10

results= []

t0 = time.time()

for i in range(10):
   
    t1 = time.time()
    
    print("-"*50)
    print("Episode n°",i+1, "/", episodes)
    print(str(datetime.datetime.now()),"- Learning starts")
    model.learn(total_timesteps = (len(df)-1))#,callback = eval_callback)
    print(str(datetime.datetime.now()),"- Learning ends")
    
    #save the model when you want. You can also add condition such as save it if the reward is greater than the previous ones.
    if episodes % 5 == 0:
        model.save("PPO2_Stable_Baseline_cpu"+str(num_cpu)+"_nminibatches"+str(nminibatches))
        print("Model saved")
    
    print()
    print(str(datetime.datetime.now()),"- Evaluating...")    
    result = evaluate(model,  train_env,num_cpu)#num_cpu )
    print(str(datetime.datetime.now()),"- Evaluating ends...")
    
    results.append(result)
    delay1 = time.time() - t1 
    
    print()
    print(str(datetime.datetime.now()),"- Episode took: " , float(round(delay1,1)),  " seconds, i.e ", round(delay1/60,2), " minutes." )
    
delay = time.time() - t0 

#you may want to add the results in a csv file.
with open("results.csv","a",newline = "") as file:
    writer = csv.writer(file)
    
    writer.writerow([str(datetime.datetime.now()),
                     str(float(round(delay,1))) + " seconds i.e "+ str(round(delay/60,2))+ " minutes." ,
                     "PPO2_Stable_Baseline_cpu" + str(n_cpu) + "_nminibatches"+str(nminibatches)+".zip",
                     np.mean(results),
                     np.std(results)])
    
print("*" * 100)
print("n_cpu= ", str(n_cpu))
print("nminibatches= ", str(nminibatches))
print("Results: ",results)
print(str(datetime.datetime.now())," - Evaluating Ends. Mean= ", np.mean(results), ". SD= ", np.std(results))
print("Training took: " , float(round(delay,1)),  " seconds, i.e ", round(delay/60,2), " minutes." )
print(round(delay,2)/episodes, " sec/episodes")
print("*" * 100)


#if you want to shutdown the computer after the training
import os
os.system("shutdown.exe /h")
 
