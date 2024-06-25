from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.env_util import make_vec_env

from linear_environment import LinearEnvironment
from global_environment import GlobalEnvironment
from local_environment import LocalEnvironment
import networkx as nx
import numpy as np

def wagner_value(graph, _={}):
    n = graph.shape[0]
    nxgraph = nx.Graph(graph)
    for i in range(n):
        if nxgraph.has_edge(i,i):
            nxgraph.remove_edge(i,i)
    # print(nxgraph.edges)
    if not nx.is_connected(nxgraph): #or nx.is_isomorphic(nxgraph, self.nxstar)):
        score = -np.Inf    
    else:
        constant = 1 + np.sqrt(n - 1)
        lambda_1 = max(np.real(nx.adjacency_spectrum(nxgraph)))
        mu = len(nx.max_weight_matching(nxgraph,maxcardinality=True))
        score = constant - (lambda_1 + mu)
        
    # if True:
    #     return -np.sum(np.sum(graph))
    return score

if __name__ == '__main__':
    number_of_nodes=18
    env = LinearEnvironment(number_of_nodes,wagner_value,dense_reward=True,start_with_complete_graph=True)
    total_timesteps = 1_000_000
    
    #model = PPO("MlpPolicy", env, verbose=0, n_steps=128)
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=dict(net_arch=[256,256,256,256,256,256,256,256,256,256,256,256]),learning_rate=1e-4)
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    model.save(f"WagnerLinearEnvironment{number_of_nodes}PPO{total_timesteps}")
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(reward)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         break