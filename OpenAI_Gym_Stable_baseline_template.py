

import csv
import datetime
import json, os, shutil, sys
import os
checkpoint_root = r"C:\Users\Alexandre\Desktop\Bourse\Stock-Trading-Visualization-attemps"
wd=os.chdir(checkpoint_root)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gym
# import ray
import pandas as pd
import numpy as np
import time
from typing import Dict
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy,CnnPolicy,CnnLstmPolicy,CnnLnLstmPolicy
from stable_baselines import PPO2,ACER, ACKTR
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common.env_checker import check_env

import tensorflow as tf

import psutil

import matplotlib.pyplot as plt

print(psutil.cpu_count())
print(psutil.cpu_count(logical=False))


# K.set_session(session)    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
# print(env_ids)

#to consult metrics
#tensorboard --logdir ~/ray_results

longueur = 1000
df = pd.read_csv('./data/MSFT.csv')
df = df.sort_values('Date')
df = df.reset_index(drop=True)

# print("removing Unnamed column")
df = df.drop('Unnamed: 0', 1)
df = df.dropna()
# print(df.columns)
df = df.reset_index(drop=True)
# df = df.set_index('Date')

 
df1 = df[['Date','Open','Close']]
df1["Open_diff"] =df["Open"].diff()
df1["Close_diff"] =df["Close"].diff()
df1["High_diff"] =df["High"].diff()
df1["Low_diff"] =df["Low"].diff()
df1["Volume_diff"] =df["Volume"].diff()
df1 = df1.set_index('Date')


df2 = df.set_index('Date')
df2 = df2[['High', 'Low' ,'Adjusted_Close', 'Volume',
   'Dividends']].diff() 


result = pd.concat([df1, df2], axis=1, sort=False)

result.reset_index(level=0, inplace=True)

df = result.dropna()
 


def evaluate(model, env,n_cpu,graph = False):
    
    ep_r = []
    ep_l = []
    networth = []
    networth_last = []
    actions = []
    
    t0 = time.time()
    episodes=1
    
    for e in range(episodes):
        obs = env.reset()

        zeros= np.zeros((n_cpu-1,obs.shape[1],obs.shape[2]))

        state = None

        done = [False for _ in range(env.num_envs)]
        print('env.num_envs= ', env.num_envs)

        total_r = 0.
        total_l = 0.
        while True:

            action, _states = model.predict(obs,state=state, mask=done,deterministic=True)

            obs, rewards, dones, infos = env.step(action)

            actions.append(infos[0]["action"])
            # print("infos",infos[0])
            networth.append(infos[0]["networth"])
            
            # print("obs: ",obs[0])
            total_l += 1.
            total_r += rewards[0]
            if dones[0]:
                break
        networth_last.append(networth[-1])
        ep_r.append(total_r)
        ep_l.append(total_l)

    
    average = sum(networth_last)/len(networth_last)
    delay = time.time() - t0 
    
    print("Evaluation took: " , float(round(delay,1)),  " seconds, i.e ", round(delay/60,2), " minutes." )
    print("Final net worth= ",str(round(networth[-1],2))+ '€')
    print("Average net worth= ",str(round(average,2))+ '€') 
    
    if graph == True:
        pd.DataFrame(networth).plot(title= (" Average gain: "+str(round(average,2))+ '€'))
        pd.DataFrame(df["Adjusted_Close"]).plot(title= (" Average gain: "+str(round(average,2))+ '€'))
        pd.DataFrame(networth).plot(title= (" Average gain: "+str(round(average,2))+ '€'))
        
        #plot actions
        
        buy = []
        sell = []
        wait = []
        
        for i in actions:
            if i == 1:
                buy.append(1)
                sell.append(0)
                wait.append(0)
            elif i == -1:
                buy.append(0)
                sell.append(1)
                wait.append(0)
            elif i== 0:
                buy.append(0)
                sell.append(0)
                wait.append(1)
            
        print("Buy: ", sum(buy), ' i.e ', round(sum(buy)/len(buy)*100,2), "%")
        print("Sell: ", sum(sell), ' i.e ', round(sum(sell)/len(sell)*100,2), "%")
        print("Wait: ", sum(wait), ' i.e ', round(sum(wait)/len(wait)*100,2), "%")
        
        l = len(actions)-1
        
        buy = [buy[i] * df.iloc[i]["Adjusted_Close"] for i in range(l)]
        sell = [sell[i] * df.iloc[i]["Adjusted_Close"] for i in range(l)]
        wait = [wait[i] * df.iloc[i]["Adjusted_Close"] for i in range(l)]
        
        plt.figure(figsize = (30,8))
        plt.xlabel("Days")
        plt.ylabel("Price")
        
        plt.plot(list(range(l+1)),df["Adjusted_Close"][:l+1],'-', label = 'Adj_Close',linewidth=0.5)
        
        plt.plot(list(range(l+1)),df["MA_5"][:l+1],'-', label = 'MA_5',linewidth=1)
        plt.plot(list(range(l+1)),df["MA_10"][:l+1],'-', label = 'MA_10',linewidth=1)
        plt.plot(list(range(l+1)),df["MA_20"][:l+1],'-', label = 'MA_20',linewidth=1)
        plt.plot(list(range(l+1)),df["MA_50"][:l+1],'-', label = 'MA_50',linewidth=1)
        plt.plot(list(range(l+1)),df["MA_100"][:l+1],'-', label = 'MA_100',linewidth=1)
        
        
        plt.plot(list(range(l)),buy, '.',c="red",markersize=5,label = 'Buy')
        plt.plot(list(range(l)),sell, '.',c="green",markersize=8,label = 'Sell')
        plt.plot(list(range(l)),wait, '.',c="orange",markersize=2,label = 'wait')
        plt.legend(loc = 'best')
        # plt.plot(list(range(l+1)),sell_bt_cant, '.',c="orange",markersize=4,label = 'Wanted to sell',)
        
        plt.show()
    
    return round(networth[-1],2)



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
        # env = env.__init__({"config":config})


        time.sleep(0.5)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    time.sleep(0.5)
    return _init

# print(env.observation_space)  # (8 * 5 * 5)

num_cpu = 6
nminibatches=1
n_cpu = num_cpu

config = {
         'df': df,"use_lstm": True,  'reward_func': "omega","n_cpu": n_cpu
          }


env_id = 'StockTradingEnv-v0'

                    
train_env = SubprocVecEnv([make_env(env_id, i,config)  for i in range(num_cpu)])



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


model = PPO2(config["policy"],train_env, verbose=0, nminibatches=nminibatches,policy_kwargs=config["policy_kwargs"],learning_rate = config["learning_rate"])
    

if os.path.exists(checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(n_cpu) + "_nminibatches"+str(nminibatches)+".zip"):
    model = PPO2.load(checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(n_cpu) + "_nminibatches"+str(nminibatches)+".zip",env=train_env,policy=MlpLnLstmPolicy,verbose=0)
    print()
    print("*"*100)
    print("Model loaded from:",checkpoint_root+"\\"+ "PPO2_Stable_Baseline_cpu" + str(n_cpu) + "_nminibatches"+str(nminibatches)+".zip")
    print("*"*100)
    print()


episodes = 10
results= []
t0 = time.time()

for i in range(10):
    
    t1 = time.time()
    print("-"*50)
    print("Episode n°",i+1, "/", episodes)
    print(str(datetime.datetime.now()),"- Learning starts")
    model.learn(total_timesteps = (len(df)-1)*episodes)#,callback = eval_callback)
    print(str(datetime.datetime.now()),"- Learning ends")
    
     # if episodes % 5 == 0:
    model.save("PPO2_Stable_Baseline_cpu"+str(n_cpu)+"_nminibatches"+str(nminibatches))
    print("Model saved")
    
    
    print()
    print(str(datetime.datetime.now()),"- Evaluating...")    
    result = evaluate(model,  train_env,n_cpu)#num_cpu )
    print(str(datetime.datetime.now()),"- Evaluating ends...")
    
    results.append(result)
    delay1 = time.time() - t1 
    
   
    print()
    print(str(datetime.datetime.now()),"- Episode took: " , float(round(delay1,1)),  " seconds, i.e ", round(delay1/60,2), " minutes." )
    
delay = time.time() - t0 




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




