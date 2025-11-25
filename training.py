import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################








#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # Environment settings
    max_bot_pos = 100
    max_cat_pos = 100
    max_velocity_states = 5
    num_actions = env.action_space.n

    learnRate = 0.1 #default rate but can be changed
    discFactor = 0.99 
    exploreRate = 1 #how often the bot tries random actions
    minEpsilon = 0.05
    epsilonDecay = 0.995 #hm epsilon reduces every episode


    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        # Inside your training loop
        state, _ = env.reset()
        done = False
        prev_catPos = state % 100  # store initial cat position

        while not done:
            botPos = state // 100
            catPos = state % 100
            
            # Compute velocity dynamically
            catVel = np.clip(catPos - prev_catPos + 2, 0, 4)  # shift to range 0-4
            prev_catPos = catPos  # update for next step

            # Decide whether to explore or exploit
            if random.random() < exploreRate:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])  # still use your existing Q-table

            # Take action
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward logic
            reward = -1
            nextBotPos = next_state // 100
            nextCatPos = next_state % 100
            if nextBotPos == nextCatPos:
                reward = 100

            # Q-learning update
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            q_table[state][action] = old_value + learnRate * (reward + discFactor * next_max - old_value)

            state = next_state

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table