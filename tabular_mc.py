from collections import defaultdict
import numpy as np
import sys
import copy

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1, max_steps=1000, save=True):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    if save == True:
        # Initialize an empty list to hold all episodes
        episodes = []

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # Initialize a dictionary to store the Q-value differences
    Q_diff = defaultdict(lambda: np.zeros(env.nA))

    # Initialize a list to store the norms of Q_diff after each episode
    Q_diff_norms = []

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is a list of (state, action, reward, next_state) tuples
        episode = []
        env.reset() # This initializes self.state with the empty graph
        for t in range(max_steps):
            state = copy.deepcopy(env.state) # Backup the original state, before step modify it
            probs = policy(tuple(state))
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done = env.step(action)
            next_state = copy.deepcopy(next_state) # Step updates env.state and returns it, thus next_state will change
            episode.append((state, action, reward, next_state))
            if done:
                break
        
        if save == True:
            # Add the episode to the episodes list
            episodes.append(episode)

        # The policy is improved implicitly by changing the Q dictionary
        for i, (state, action, reward, _) in enumerate(episode):
            sa_pair = (tuple(state), action)
            # Sum up all rewards starting from time i
            G = sum([x[2] * (discount_factor ** j) for j, x in enumerate(episode[i:])])
            # Calculate average return for this state over all sampled episodes
            old_Q = Q[tuple(state)][action]
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            new_Q = returns_sum[sa_pair] / returns_count[sa_pair]
            Q[tuple(state)][action] = new_Q
            # Store the absolute difference between old and new Q values
            Q_diff[tuple(state)][action] = abs(new_Q - old_Q)

                # After each episode, compute the norm of Q_diff and store it

        Q_diff_values = np.array(list(Q_diff.values()))
        Q_diff_norm = np.linalg.norm(Q_diff_values)
        Q_diff_norms.append(Q_diff_norm)
        
    if save == True:
        return Q, policy, episodes, Q_diff_norms
    else:
        return Q, policy