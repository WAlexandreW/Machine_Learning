import time
import pandas as pd

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

#just in case, to clear it
ray.shutdown()

#import data
df = pd.read_csv('./data/MSFT.csv')

#initialize ray for parallelized computations, testing
ray.init(num_cpus=6, num_gpus=1,ignore_reinit_error=True)

#configuration of your env, if any
config = {
         'df': df,"use_lstm": False,
         'reward_func': CustomFunc-v0
          }

env = DummyVecEnv([lambda: env(env_config=config)])  

#the function with the score to maximize
def get_score(agent,env):
        score = []
        obs = env.reset()
      
        while True:
            action, _states = agent.predict(obs)
            obs[0], rewards, dones, infos = env.step(action)

            score.append(infos[0]["score"])
            if dones[0]:
                break
        return score[-1]

#the function to be launched in hyperopt
def f(config):
    #create an agent
    agent = PPO2(policy =MlpPolicy,
                 env = env,
                 learning_rate = config["learning_rate"],
                 policy_kwargs = config["policy_kwargs"]
                 )
    #get its score
    score = get_score(agent,env)
    #send the score to tune
    tune.report(score = score)



# Create a HyperOpt search space : all the possibles values during the hyperparameter tuning
space = {

    'policy' : hp.choice('policy', ["MlpLnLstmPolicy"]),
    'learning_rate':hp.choice('learning_rate', [0.000001,0.00005,0.00010,0.00025,0.00050,0.001]),
    'policy_kwargs' : hp.choice("net_arch" ,[{"net_arch":[16, 16]},{"net_arch":[32, 32]},{"net_arch":[64, 64]},{"net_arch":[128, 128]},{"net_arch":[256, 256]},{"net_arch":[512, 512]},
                                            {"net_arch":[16, 16, 16]},{"net_arch":[32, 32, 32]},{"net_arch":[64, 64, 64]},{"net_arch":[128, 128,128]},{"net_arch":[256, 256,256]},{"net_arch":[512, 512,512]},
                                            {"net_arch":[16, 16, 16,16]},{"net_arch":[32, 32, 32,32]},{"net_arch":[64, 64, 64,64]},{"net_arch":[128, 128,128,128]},{"net_arch":[256, 256,256,256]},{"net_arch":[512, 512,512,512]}])
}

#set a timer
t0 = time.time()

# Specify the search space and maximize score
hyperopt = HyperOptSearch(space, metric="score", mode="max")

# I tested 3 possibilities below and the last one had the best results: 1) normal hyperopt 2) hyperopt with ASHASScheduler 3) Hyperopt with PBT
#decomment the one you want to test. 
#the resources per trial can be increased or decreased depending on your computer specs


# analysis = tune.run(f, search_alg=hyperopt, num_samples=50,resources_per_trial={"cpu": 1, "gpu": 0.2})
# analysis = tune.run(f, search_alg=hyperopt, num_samples=50,resources_per_trial={"cpu": 1, "gpu": 0.1},scheduler=ASHAScheduler(metric="score", mode="max"))

#PBT must have its mutations values in "hyperparam_mutations"
analysis = tune.run(f,verbose=0, search_alg=hyperopt,reuse_actors= True, num_samples=1000,resources_per_trial={"cpu": 1, "gpu": 0.1},scheduler=PopulationBasedTraining(metric="score", mode="max",hyperparam_mutations={
            "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5,5e-6, 1e-6],
            "net_arch": [{"net_arch":[16, 16]},{"net_arch":[32, 32]},{"net_arch":[64, 64]},{"net_arch":[128, 128]},{"net_arch":[256, 256]},{"net_arch":[512, 512]},
                                            {"net_arch":[16, 16, 16]},{"net_arch":[32, 32, 32]},{"net_arch":[64, 64, 64]},{"net_arch":[128, 128,128]},{"net_arch":[256, 256,256]},{"net_arch":[512, 512,512]},
                                            {"net_arch":[16, 16, 16,16]},{"net_arch":[32, 32, 32,32]},{"net_arch":[64, 64, 64,64]},{"net_arch":[128, 128,128,128]},{"net_arch":[256, 256,256,256]},{"net_arch":[512, 512,512,512]}]
        }))

#save the results in a Dataframe
df1 = analysis.dataframe()

#compute the delay
delay1 = time.time() - t0
print("HP Optimization took: " , float(round(delay1,1)),  " seconds, i.e ", round(delay1/60,2), " minutes." )
    
# Get the best hyperparameters
best_hyperparameters = analysis.get_best_config(metric = "score")
print(best_hyperparameters)

#you may want to save the results in an excel file, concat with previous data (its an append actually)
result = pd.read_excel('C:/Users/Alexandre/Desktop/Projects/Hyperparameter tuning results.xlsx')
result = pd.concat([result,df1])
result.to_excel('Hyperparameter tuning results.xlsx') 

