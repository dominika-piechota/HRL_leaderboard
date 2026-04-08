from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


import argparse
import ast
import json
import logging
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections     import deque
from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import clear_SUMO_files
from utils           import print_agent_counts

from routerl import Keychain as kc


from dqn_cb_temp_utils import GlobalObservation, Network, DQN, run_episode


# TODO: check for 'todo' and 'note' & clean
DEBUG = True



####################################   OBSERVATION AND Q-NETWORK IMPLEMENTATION   #####################################################



##################################################################################################################################






# Main script to run the centralized DQN experiment
if __name__ == "__main__":

    if DEBUG:
        print(f"\nDEBUG flag set to {DEBUG}.")
        print(f"Turn off manually in {__file__} to disable debug messages and assertions.\n")


    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "dqn_cb"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value


    # Define input / output paths
    custom_network_folder = f"../networks/{network}"
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)

    num_machines = int(num_agents * ratio_machines)
    experience_collecting_episodes = 0 if num_machines == 0 else math.ceil(min_buffer_size/num_machines) # num of episodes needed to min fill the buffer
    total_episodes = human_learning_episodes + experience_collecting_episodes + training_episodes + test_eps
    
    # Phases for plotting
    phases = [
        1,
        human_learning_episodes,
        human_learning_episodes + int(experience_collecting_episodes),
        human_learning_episodes + int(experience_collecting_episodes) + int(training_episodes)
    ] # NOTE: check this in context of collecting experience phase
    phase_names = ["Human stabilization", "Mutation and experience collecting", "AV learning", "Testing"]




    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    

            
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    if DEBUG:
        assert update_every_k_episodes >=1 # (mid-episode training not supported)


    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : "previous_agents_plus_start_time"
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo"
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)


    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning") #TODO: include collecting experience here
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()


    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    
    # Initialize GlobalObservation. Define Q-Net and agent lookup
    global_observation = GlobalObservation(env.machine_agents)
    q_net = DQN(
            input_dim = len(env.machine_agents) * global_observation.num_table_columns, ##
            output_dim = env.environment_params[kc.ACTION_SPACE_SIZE], ##
            hidden_dims = hidden_dims,

            batch_size = batch_size,
            num_batches = num_batches,

            min_buffer_size = min_buffer_size,
            max_buffer_size = max_buffer_size,

            eps_init = eps_init,
            eps_decay = eps_decay,

            lr=lr, 
            device=device
        )
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}




    ### Collect experience for training (random sampling of actions)
    pbar.set_description("Collecting experience samples.")
    os.makedirs(plots_folder, exist_ok=True)

    global_observation.enable_transition_collection()

    for episode in range(experience_collecting_episodes):
        print(f"\nExperience collecting episode {episode}/{experience_collecting_episodes}")
        # NOTE: enable qnet push in eval mode?

        assert global_observation.collect_transitions == True

        # --- Run single episode and collect transitions ---
        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )
        
        # --- Move collected (s, a, r) transitions to DQN replay buffer ---
        for (s,a,r) in global_observation.flush_transitions():
            q_net.push(s,a,r)

        # --- Plot visualization ---
        #NOTE: this is probably redundant and can be done once at the end of the episode (as data for plots are read from files and plot is overwritten each call)
        #NOTE: check how plot_every interacts with save_every (what if n/k mismatch?)
        # Is this param only for generating intermediate plots in case experiment breaks?
        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()
    
    if DEBUG:
        assert len(q_net.memory) >= q_net.min_buffer_size



    ### Learning phase ###
    pbar.set_description("AV learning")
    q_net.set_train()
    global_observation.enable_transition_collection() #NOTE(3): maybe move this inside run_episode (with phase arg in run_episode?)?

    for episode in range(training_episodes):
        print(f"\nTraining episode: {episode}/{training_episodes}") ##NOTE: rm prints before PR

        assert global_observation.collect_transitions == True

        # --- Run single episode and collect transitions ---
        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )
        
        # --- Move collected (s, a, r) transitions to DQN replay buffer ---
        for (s,a,r) in global_observation.flush_transitions(): 
            q_net.push(s,a,r)

        # --- Network parameter update ---
        if episode > 0 and episode % update_every_k_episodes == 0:
            q_net.learn() 

        # --- Plot visualization ---
        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()
    
    
    ### Testing phase ###
    pbar.set_description("Testing")

    q_net.set_eval()
    global_observation.disable_transition_collection()


    for episode in range(test_eps):
        print(f"\nTest episode: {episode}/{test_eps}")

        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_df = pd.DataFrame({"losses": q_net.loss})
    losses_df.to_csv(os.path.join(records_folder, "losses.csv"))
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)


    print("\n\nREMEMBER TO COMMIT ALGO CONFIG (and maybe env_config/test.txt (changed plotting to every episode)\n\n")

