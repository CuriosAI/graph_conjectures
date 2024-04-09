import numpy as np
from linear_environment import LinearEnvironment

def value_fun(graph, timestep):
    return np.sum(np.sum(graph))

n = 5
e = n*n
linear_environment = LinearEnvironment(n,value_fun, True, True)
observation, info = linear_environment.reset()
expected_starting_observation = np.zeros((2*e,),dtype=np.int8)
expected_starting_observation[0:(e+1)] = 1
assert np.array_equal(observation, expected_starting_observation), "error, invalid starting state"
observation, reward, done, _, info = linear_environment.step(1)
print(f"reward={reward}")
expected_starting_observation[e] = 0
expected_starting_observation[e+1] = 1
expected_starting_observation[e+n] = 1
print(observation[0:e])
print(observation[e:])
assert np.array_equal(observation, expected_starting_observation), "error, invalid starting state"
assert reward==0, "error, invalid reward"
assert not(done), "error, invalid done"
observation, reward, done, _, info = linear_environment.step(0)
print(f"reward={reward}")
print(observation[0:e])
print(observation[e:])


