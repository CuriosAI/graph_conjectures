from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


from helpers import make_normalized_linenv, create_experiment_folder
from envs import LinEnv
from save_and_load import CheckOnTrainEnvCallback

def a2c_run(number_of_nodes):
    # This is an A2C attempt
    number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2

    # register_linenv(number_of_nodes=number_of_nodes, normalize_reward=True) # Needed by rl_zoo3. This register 'LinEnv-v0' with normalization. To change this name we need to change it also in rl_zoo3/hyperparams/ppo.yml

    # Create a list of training environments for multiprocessing
    number_of_envs = 1 # Set this value = number of cores to enable multiprocess
    train_env = make_vec_env(make_normalized_linenv(number_of_nodes), seed=1, n_envs=number_of_envs, vec_env_cls=DummyVecEnv) # Replace DummyVecEn  with SubprocVecEnv for true multiprocess

    episode_length = number_of_edges # LinEnv has a fixed horizon: every episode lasts exactly number_of_edges steps

    # Create the A2C agent. net_arch = [128, 64, 4] is Wagner choice.
    net_arch = [128, 64, 4] # To be tuned

    # Generate a unique folder name for saving experiment data
    experiment_folder = create_experiment_folder("A2C", number_of_nodes, net_arch)

    model = A2C('MlpPolicy', train_env, verbose=1, policy_kwargs={"net_arch": net_arch}, tensorboard_log=f"./{experiment_folder}/tensorboard/")

    # Since we are interested in a single graph, and not in the whole policy producing that graph, it makes sense to check the graphs explored in train_env
    # Be careful that stop_on_star=False will produce a non-stopping training, and if star_check=True the disk will be probably filled with pickle files of the star. It can be useful to check if training is working, because the star should be found by the greedy policy by a close-to-optimal policy
    check_freq = 1 # Check frequency for the callback: check every 1 call to the env, that is, every step
    # check_freq = episode_length * 1 # Check every 1 episode
    check_callback = CheckOnTrainEnvCallback(check_freq=check_freq, log_folder=experiment_folder, star_check=True, stop_on_star=False, stop_on_counterexample=False, verbose=0)

    # If we want to evaluate the policy, we need a separate (and not normalized) env because we do not want to interfere with train_env by performing episodes
    eval_env = LinEnv(number_of_nodes, normalize_reward=False) # For evaluation we don't want normalization
    eval_freq = 10 * episode_length # Evaluation is performed every 10 episodes
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=eval_freq, log_path=f"./{experiment_folder}/eval_callback/", best_model_save_path=f"./{experiment_folder}/best_model/", deterministic=True, verbose=1)
    # check_callback = CheckCallback(eval_env, check_freq=check_freq, log_file='log.txt', verbose=1)

    # Save a checkpoint every 100 episodes
    checkpoint_callback = CheckpointCallback(save_freq=100*episode_length, save_path=f"./{experiment_folder}/checkpoints/", name_prefix="model")

    # Train the agent until a star or a counterexample is found
    total_timesteps = 10E9
    model.learn(total_timesteps=total_timesteps, callback=[check_callback, eval_callback, checkpoint_callback], progress_bar=True)

    # load_results("log.txt")
    return()