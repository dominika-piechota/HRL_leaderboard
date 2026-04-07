# fill imports
from __future__ import annotations

import os
import sys

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)


import argparse
import ast
import json
import logging
import random

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



###############################
# Global observation
################################
# NOTE: docstrs reviewed
# TODO: view remaining NOTE/TODO annotations in this class
class GlobalObservation:
    """
    Manage the global observation of the AV fleet.

    This class maintains a global state table (pd.DataFrame) that tracks
    the state of all AV agents in the environment. Each row corresponds
    to an agent, and each column represents an observed feature such as:
    start time, origin, destination, chosen route, travel time, completion status.

    Key responsibilities:
        - Maintain agent data in a pd.DataFrame of shape (num_agents, num_features).
        - Keep track of the currently departing agent.
        - Update the state table at each environment timestep using information from the TrafficEnvironment.
        - Provide agents with a view of the global state conditioned on their start time.

    Attributes:
        features (dict):
            Mapping of column names to their default values.
        state_table (pd.DataFrame):
            Global observation table where rows correspond to agent IDs
            and columns correspond to features.
        _transitions (dict):
            Cache of per-agent episode experience. Keys are agent IDs, and values
            store (observation, action, reward) used to populate the replay
            buffer in the DQN after episode completion.
        collect_transitions (bool):
            Whether to cache transition data in _transitions during state updates.
        acting_agent_id (int):
            ID of the agent currently taking an action.
        episode_finished (bool):
            Indicates whether the current episode has finished.
    """



    ##############################
    ### Initialization & reset ###
    def __init__(self, agents: list[BaseAgent], collect_transitions:bool=True)->None:

        
        self.features = {
            'start_time': -1,
            'origin': -1,
            'destination': -1,
            'route': -1,
            'travel_time': -1.0,
            'has_finished': 0,
            'is_known': 0,
            #'is_acting': 0, #<- kept in self.acting_agent_id and appended only when generating agent's observation
        }
        self.state_table = self._initialize_state_table(agents)

        self._transitions = {agent.id : {} for agent in agents} # {agentid: {obs: , action: , reward: }} #NOTE: rm comment? # store episode (s,a,r) info
        self.collect_transitions = collect_transitions

        self.acting_agent_id = None
        self.episode_finished = False

    def reset(self)->None:
        """
        Reset the global observation:
            - Fill state table (pd.DataFrame of shape (n_machine_agents,n_features)) with default values for each feature column.
            - Clear transition memory.
            - Set self.episode_finished flag to False.
            - Set currently acting agent to None.

        Args:
            None
        Returns:
            None
        """

        self.state_table[:] = pd.DataFrame(self.features, index=self.state_table.index) # NOTE: add permuting agents with the same start times here or somwhere else (e.g. compare _initialize_state_table)
        self._transitions = {agentid: {} for agentid in self._transitions}
        
        self.acting_agent_id = None
        self.episode_finished = False 
        return

    def _initialize_state_table(self, agents: list[BaseAgent] )->pd.DataFrame:
        """
        Initialize state table.
        Set row indices to agent IDs (integers) sorted by start times.
        Set column names as in self.features.
        Fill with default values for each column.

        Args:
            agents (list[BaseAgent]): list of agents to be included in the global state table.

        Returns:
            pd.DataFrame indexed with agent IDs (integers) filled with default values for each column.
        """

        idx_time_sorted = [agent.id for agent in sorted(agents, key=lambda x: x.start_time)] # note: some permutations on agents with equal start times can be added

        empty_columns = {
            key: [value] * len(idx_time_sorted)
            for key, value in self.features.items()
        }

        return pd.DataFrame(empty_columns, index=idx_time_sorted)


    #################################################################
    ### Registering departing agent, getting agent's observation  ###
    #################################################################
    def register_starting_agent(self, agent: BaseAgent)->None:
        """
        Add agent `start_time`, `origin` and `destination` to the agent row.
        Move `self.acting_agent_id` indicator to registered agent.

        Args:
            agent (BaseAgent): An agent whose departure data is to be logged.
        Returns:
            None
        """

        assert agent.id in self.state_table.index
        assert self.state_table.at[agent.id, 'is_known'] == 0, "Trying to overwrite information for known (already registered) agent"

        # Ensure that agents are sorted by travel time and added in this order (-> start time in prev row is defined and not greater than current)
        idx_iloc, col = self.state_table.index.get_loc(agent.id), 'start_time' 
        assert isinstance(idx_iloc, int)
        assert (idx_iloc == 0) or (self.state_table.iloc[idx_iloc-1][col] != self.features[col] and self.state_table.iloc[idx_iloc-1][col] <= agent.start_time)
        

        # Register agent
        self.state_table.at[agent.id, 'is_known'] = 1
        self.state_table.loc[agent.id, ['start_time', 'origin', 'destination']] = [agent.start_time, agent.origin, agent.destination]
        self.acting_agent_id = agent.id
        return

    def register_starting_agent_action(self, agentid: int, action: int)->None:
        """
        Add action (route identifier) to 'route' column for currently acting agent.
        """
        assert agentid == self.acting_agent_id
        self.state_table.at[agentid, 'route'] = action
        return

    def generate_agent_observation(self, agentid: int) -> np.ndarray:
        """
        Generate an agent-specific snapshot of the global state table at their departure time.

        The observation is constructed as a snapshot of the global state table
        at the timestep when the specified agent departs.
        The snapshot reflects only information available at that time:
            - agents that depart later are not included (no future information),
            - earlier agents may be either completed or still in transit,
            depending on their status at that timestep.

        An additional `is_acting` indicator column is appended to mark the acting agent.

        Args:
            agentid (int):
                Identifier of the agent for whom the observation is generated.

        Returns:
            np.ndarray:
                Array representation of the state table snapshot, preserving the original
                column order, with an additional one-hot `is_acting` column appended at the end.
                This added column has value of 1 in the row corresponding to the specified agent.
        """


        assert agentid in self.state_table.index, f"Agent ID {agentid} not in self.state_table.index\nself.state_table.index: {self.state_table.index}"
        assert agentid == self.acting_agent_id

        #######################################################################################################################################################################
        # Potential more sanity checks to perform (when ensuring corectness after changes; swithed off for efficiency) TODO: check this comment; to remove?s
        #   - check if agent travel_times are sorted: assert self._is_column_nondescending(colname='start_time')
        #   - optionally: check if all rows below current agent are filled with empty vals (assumed in current version; scheduled future agents may be added in next versions)
        #######################################################################################################################################################################


        # Get agent's view of state table
        obs = self.state_table.copy()
        obs['is_acting'] = 0
        obs.at[agentid, 'is_acting'] = 1

        return obs.to_numpy()

    def get_flattened_agent_observation(self, agentid: int)->np.ndarray:
        return self.generate_agent_observation(agentid).flatten()



    ############################################
    ### Communication with TrafficEvironment ###
    ############################################
    def update_state_with_recently_finished_machines(self, env: TrafficEnvironment)->None:
        """
        Update the global observation table with changes in the environment since the last environment snapshot.

        Specifically:
            - register travel times for agents that have finished their trips since last snapshot,
            - set the 'has_finished' indicators for those agents.

        Args:
            env (TrafficEnvironment): environment for which observation is registered.
        
        Returns:
            None
        """ 

        # Get travel times for recently finished agents
        driving_agents = self.driving_agents() # <- IDs of agents that were active in the last snapshot of the observation
        finished_agents_times = { # <- Agents that finished after last snapshot - according to update from TrafficEnvironment
            info[kc.AGENT_ID] : info[kc.TRAVEL_TIME]
            for info in env.travel_times_list
            if
                info[kc.AGENT_ID] in driving_agents and
                kc.TRAVEL_TIME in info and 
                info[kc.TRAVEL_TIME] != self.features['travel_time'] # check if assigned travel time is not 'empty value'
        } 

        # Update state table
        self.state_table['travel_time'].update(pd.Series(finished_agents_times))
        self.state_table['has_finished'].update(pd.Series({agent: 1 for agent in finished_agents_times}))

        if self.state_table['has_finished'].all():
            self.episode_finished = True
        return 

    
    ###########################################
    ### Experience (s,a,r) cache management ###
    ###########################################

    def flush_transitions(self)->Iterator[Tuple[np.ndarray, int, float]]:
        """
        Export episode (s,a,r) data from _transitions dict and clear transitions cache.
        """
        if not self.collect_transitions:
            raise RuntimeError("Transition recording was disabled (collect_transitions=False). Set to True to enable recording and export.")
        assert self._transitions

        for transition in self._transitions.values():
            yield transition['observation'], transition['action'], transition['reward']
        
        # Execute when iterator exhausted
        for transition in self._transitions.values():
            transition.clear()


    def cache_transition_observation(self, agentid: int, observation: np.ndarray)->None:
        self._cache_transition_field(agentid, 'observation', observation)

    def cache_transition_action(self, agentid: int, action: int)->None:
        self._cache_transition_field(agentid, 'action', action)

    def cache_transition_reward(self, agentid: int, reward: float)->None:
        self._cache_transition_field(agentid, 'reward', reward)

    def _cache_transition_field(self, agentid: int, field: str, value) -> None:

        if not self.collect_transitions:
            raise RuntimeError("Transition recording is disabled (collect_transitions=False)")

        if field not in ['observation', 'action', 'reward']:
            raise ValueError(f"Invalid transition field name ({field}), must be one of: 'observation', 'action', 'reward' ")
        if field in self._transitions[agentid]:
            raise KeyError(f"{field} for agent {agentid} already exists")

        self._transitions[agentid][field] = value





    #################################
    ####### Auxiliary methods #######

    ### Accessing state table info ###
    @property
    def num_table_columns(self)->int:
        return len(self.features) + 1 # features + 'is_acting' column

    def driving_agents(self) -> pd.Index:
        """
        Get IDs of agents that are marked as 'started' but not 'finished' in the observation table.
        """
        df = self.state_table
        return df.index[df['is_known'] & ~df['has_finished']]

    def get_agent_feature(self, agentid: int, feature: str)->Any: #TODO(2): change to property(?)
        return self.state_table.at[agentid, feature]

    def set_agent_feature(self, agentid: int, feature: str, value: Any)->None: #TODO(2): change to property(?)

        current_value = self.state_table.at[agentid, feature]
        default_value = self.features[feature]

        if current_value != default_value:
            raise valueError(f"Feature '{feature}' for agent '{agentid}' is already set (value: {current_value}).")
            
        self.state_table.at[agentid, feature] = value
        return

    def is_empty_cell(self, agentid: int, feature: str)->bool:
        """
        Check whether the table cell contains a default (empty) value.

        Args:
            agentid (int): row index.
            feature (str): column name.

        Returns:
            True if cell equals to the default empty value for that feature.
        """
        return self.state_table.at[agentid, feature] == self.features[feature]

    ### Object state ###
    def disable_transition_collection(self) -> None:
        self.collect_transitions = False
    def enable_transition_collection(self) -> None:
        self.collect_transitions = True

    ### Checking correctness ###
    def _is_column_nondescending(self, colname: str)->bool:
        """
        Check if state table column is sorted in non-descending order.
        Ignore suffix filled with default empty values.
        """

        # Get column and default value; raises error if column not present
        col = self.state_table[colname]
        default_val = self.features[colname]

        if len(col) <= 1:
            return True


        # Verify if empty and nonempty values are not mixed; get non-empty prefix
        isempty_mask = col.eq(default_val)
        if isempty_mask.any():

            first_empty = isempty_mask.idxmax()  # index of first True ( True is argmax in T/F boolean series) occurence

            # Check that all values after first empty are also empty
            clean_suffix = isempty_mask.loc[first_empty:].all()
            if not clean_suffix:
                return False
            
            # Get prefix
            prefix = col.iloc[:first_empty]

        else:
            prefix = col

        is_nondesc = (prefix.diff().iloc[1:] >= 0).all()  # drop NaN for first row
        return is_nondesc




        

###############################
# DQN Network
################################

### Simplified single-DQN implementation for single-step decision-making
class DQN(BaseLearningModel):
    """
    DQN structure:
        - predicting network
        - replay buffer
        -
    """ #TODO: curate docstring
    def __init__(self,
                input_dim: int,
                output_dim: int,
                hidden_dims: list[int],

                batch_size=64,
                num_batches= 10,

                min_buffer_size=256,
                max_buffer_size=256, 

                eps_init=0.99,
                eps_decay=0.998,

                lr=0.003, 
                device="cpu"
                ):

        super().__init__()
        self.device = device

        # Q-network
        self.action_space_size = output_dim
        self.q_network = Network(input_dim, output_dim, hidden_dims).to(self.device)
        

        # Replay buffer
        self.min_buffer_size = min_buffer_size
        self.memory = deque(maxlen=max_buffer_size)

        # Behavior policy
        self.epsilon = eps_init
        self.epsilon_decay = eps_decay

        # Training 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.loss = list() # TODO: check this; may be wrong -> break plots --all (compare with iql implementation)

        self.is_training = True

        # Validation
        assert min_buffer_size > 0
        assert min_buffer_size <= max_buffer_size #NOTE: change to value errors?
        
        if min_buffer_size < batch_size:
            raise ValueError(f"min_buffer_size ({self.min_buffer}) must be >= batch_size ({self.batch_size}) to allow sampling a full batch.")


    def set_train(self)->None:
        self.is_training = True
        self.q_network.train()

        # TODO - what with 
        # ?loss
        # ? epsilon
        # ? memory buffer
        # ? network params?

    def set_eval(self) -> None:
        """
        Set DQN .is_training = False.

        Effects:
            - Set network to .eval() -> norm layers behavior.
            - Only greedy policy in act() (q-net argmax action)
            - Prevent from running learn() and pushing to buffer.(?) #TODO: decide
        """ #NOTE: curate
        self.is_training = False
        self.q_network.eval()

        # NOTE: these probably keep unchanged here, reset in reset() [check how done in pytorch]
        # ? clear buffer
        # ? cleaar loss logging
        # ? reset epsilon
        # ? network params?

    def reset(self)->None:
        # reset buffer, loss logging, epsilon, all training changeable params, (weights and biases?) -> check how this is done in pytorch
        # set training flag to true
        raise NotImplementedError

    def act(self, state: np.ndarray) -> int:
        """
        Act epsilon-greedy.

        Evaluation mode -> exploitation.
        Training mode -> exploration (random action) with probability epsilon, exploitation (Q-net argmax) with probability (1-epsilon)
        """

        if not self.is_training:
            return self._argmax_action(state)

        if np.random.rand() < self.epsilon:
                return np.random.choice(self.action_space_size)

        return self._argmax_action(state)
    
    def push(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Add (s,a,r) tuple to the buffer.
        """
        if not self.is_training:
            raise RuntimeError("Cannot call `push()` in evaluation mode. Set `self.is_training = True` to update DQN buffer.")

        self.memory.append((state, action, reward)) # All interactions are single-step, so we only store the last state, action, and reward
        return

    def learn(self):
        """
        Update network parameters.
        """

        # Prevent learning in evaluation mode
        if not self.is_training:
            raise RuntimeError("Cannot call `learn()` in evaluation mode. Set `self.is_training = True` to update network.")

        # Skip learning if not enough data in the buffer
        if len(self.memory) < self.min_buffer_size:
            print(f"Skipping learn: memory size {len(self.memory)} < min_buffer_size {self.min_buffer}")
            return


        step_loss = list()
        for _ in range(self.num_batches):

            # Get batch of states, actions and rewards
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device) #NOTE(2): change to as_tensor(...)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device) ## TODO: check how actions are encoded in buffer (int or one-hot)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            # Predict Q-values (travel times) for actions, compare with recorded travel times
            predicted_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            # Backpropagate & optimize
            loss = self.loss_fn(predicted_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item()) #NOTE: compare this with new dev loss logging

        self.loss.append(sum(step_loss)/len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def _argmax_action(self, state: np.ndarray) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) ## NOTE: changed from FloatTensor; no copying now
        
        # Ensure (batch, dim1, ..., dimk)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0) #NOTE: add checking if earlier (without ndim==1) first dim wasnt taken as batch in URB(?)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        action = torch.argmax(q_values).item()
        return action


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Network, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x







def run_episode(
    env: TrafficEnvironment,
    dqn: DQN,
    global_observation: GlobalObservation,
    agent_lookup: dict
    )->None:
    """
    Execute a full multi-agent episode in the TrafficEnvironment.

    Responsibilities:
        - Reset environment and global observation state
        - Iterate over agents using env.agent_iter() (runs two loops - in first agents are active and select actions, in second agents are terminated - reward collection)
            - Active agents loop:
                - update global observation with environment state change (since last agent departure timestamp)
                - update global observation with current (starting) agent
                - construct observation for current agent
                - select action for current agent via DQN policy
                - Optinally: cache agent observation and action for replay buffer (if enabled)
                
            - Terminated/truncated agents loop:
                - record agent travel time in global observation (if not available earlier, during env state update)
                - Optionally: cache agent reward for replay buffer (if enabled)

    Side effects:
        - Mutates global_observation
        - Steps environment

    Args:
        env (TrafficEnvironment):
            Multi-agent traffic simulation environment.
            Implemented in the Petting-Zoo style API.
            Responsible for managing agent lifecycle, state transitions, and reward signaling.
        dqn (DQN):
            Deep Q-Network object encapsulating the policy network and experience replay buffer. #TODO: review, curate
            Provides methods for action selection (ε-greedy), learning updates, and sampling from memory.
        global_observation (GlobalObservation):
            TODO
        agent_lookup (dict): 
            Mapping of TrafficEnvironment agent identifiers to agent objects. 

    Returns:
        None
    """

    global_observation.reset()
    env.reset()


    # Iterate over agents sorted by travel time
    for agent_id_str in env.agent_iter():
        
        # if DEBUG: assert isinstance(agent_id_str, str) and agent_id.isnumeric() #TODO
        agent = agent_lookup[agent_id_str]
        agent_id = int(agent_id_str)

        _, reward, termination, truncation, info = env.last() # observation, reward, termination, truncation, info

        if termination or truncation: # All agents finished, collect rewards

            # if DEBUG:
            #     assert isinstance(reward, (np.floating, float)) and reward<0, f"Reward: {reward} ({type(reward)}); agent: {agent_id}"

            # Update table with travel times for the last chunk of agents (arriving after the last agent's departure)
            if global_observation.is_empty_cell(agentid=agent_id, feature='travel_time'): ## NOTE: new, test
                travel_time = -reward
                global_observation.set_agent_feature(agentid=agent_id, feature='travel_time', value=-travel_time)

            # Collect reward data for experience buffer
            if global_observation.collect_transitions:
                global_observation.cache_transition_reward(agent_id, reward)

            action = None

        else: # Episode in progress, select actions for agents

            # Update global observation with env state change (add travel times for recently finished agents)
            global_observation.update_state_with_recently_finished_machines(env)
            global_observation.register_starting_agent(agent)
            
            # Get observation and action for current agent
            obs = global_observation.get_flattened_agent_observation(agent_id)
            action = dqn.act(obs) # acts epsilon-greedy or armax(q-values), depends on DQN.is_training flag
            global_observation.register_starting_agent_action(agent_id, action)

            # Collecting (s,a,_) data for experience buffer
            if global_observation.collect_transitions:
                global_observation.cache_transition_observation(agent_id, obs)
                global_observation.cache_transition_action(agent_id, action)

        env.step(action)

    
    # if DEBUG:
    #     global_observation._is_column_nondescending(colname='start_time') # Sanity check #TODO: keep this or discard?
        #raise NotImplementedError("Check if state table format corresponds to finished episode.") #TODO check if all travel times filled, all is_acting==False (all is_known==True)

    # Set global observation flag to finish
    global_observation.episode_finished = True







if __name__=="__main__":
    pass