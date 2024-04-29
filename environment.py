import os
import gzip
import random
import numpy as np
from agent2 import Agent
from mind import Mind



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


from multiprocessing import Queue, Lock


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class Business:
    def __init__(self, location, owner_type, owner_ethnicity=None):
        self.location = location  
        self.owner_type = owner_type 
        self.owner_ethnicity = owner_ethnicity if owner_ethnicity else owner_type  
        self.value = 0  
        self.consumers = [] 



    def visit(self, agent):
        print(f"Attempting to visit restaurant at {self.location} by agent {agent.id}")
        self.consumers.append(agent.type)

        if agent.type == self.owner_type:
            self.value += 0  
            print(f"Restaurant at {self.location} new value: {self.value}")
        else:
            self.value += 0.1 
            print(f"Restaurant at {self.location} new value: {self.value}")
        return self.calculate_point(agent)

    def calculate_point(self, agent):
        points = 0
        points += self.value
        return points


    def get_summary(self):
        return {
            "location": self.location,
            "owner_type": self.owner_type,
            "owner_ethnicity": self.owner_ethnicity,
            "value": self.value,
            "consumers": self.consumers
        }





class Environment:
    def __init__(self, size, num_restaurants, agent_max_age, agent_range, num_actions, p_agents=0.1):
        self.size = size  
        self.num_restaurants = num_restaurants  
        self.agent_max_age = agent_max_age  
        self.agent_range = agent_range  
        self.actions = num_actions 
        self.p_agents = p_agents 
        self.agents = [] 
        self.businesses = [] 
        self.map = np.zeros(size) 

        self.input_size = (2 * agent_range + 1) ** 2 
        self.num_actions = 4 

        self._generate_map()
        self._initialize_agents()
        self._initialize_businesses()



    def _generate_map(self):
        self.map = np.zeros(self.size) 
        for _ in range(int(self.size[0] * self.size[1] * self.num_restaurants)):
            loc = self._choose_free_location()
            self.map[loc] = 2
            business_type = random.choice(['A', 'B'])
            business = Business(location=loc, owner_type=business_type)
            self.businesses.append(business)


    def _initialize_agents(self):
        num_agents = int(self.size[0] * self.size[1] * self.p_agents)
        for _ in range(num_agents):
            loc = self._choose_free_location()
            lock = Lock()  
            queue = Queue()
            agent_type = np.random.choice(['A', 'B'], p=[0.7, 0.3])
            mind = Mind(self.input_size, self.num_actions, lock, queue, self)
            agent = Agent(id=len(self.agents), loc=loc, type=agent_type, mind=mind, p=1.0, max_time=self.agent_max_age, tolerance=np.random.rand(), agent_range=self.agent_range)
            mind.set_agent(agent)
            self.agents.append(agent)

    def get_observation_window(self, agent):
        observation_window = []
        for other_agent in self.agents:
            if other_agent is not None and agent is not None:
                if other_agent is not agent: 
                    distance = np.sqrt((agent.loc[0] - other_agent.loc[0])**2 + (agent.loc[1] - other_agent.loc[1])**2)
                    if distance <= self.agent_range:  
                        observation_window.append(other_agent)
        return observation_window
    
    def _choose_free_location(self):
        free_locations = [(i, j) for i in range(self.size[0]) for j in range(self.size[1]) if self.map[i, j] == 0]
        if not free_locations:
            raise Exception("No free locations available.")
        return random.choice(free_locations)


    def _initialize_businesses(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.map[i, j] == 2:
                    business_type = random.choice(['A', 'B'])
                    business = Business(location=(i, j), owner_type=business_type)
                    self.businesses.append(business)
    

    def get_accessible_restaurants(self, agent):
        accessible_restaurants = self.businesses
        return accessible_restaurants



    def _choose_location(self):
        free_locations = [(i, j) for i in range(self.size[0]) for j in range(self.size[1]) if self.map[i, j] == 0]
        return random.choice(free_locations)


    def step(self):
        for agent in self.agents:
            if agent.is_alive():
                agent_state = agent.get_state()
                action = agent.decide(agent_state)
                self.try_move(agent, action)

    def try_move(self, agent, initial_direction):
        directions = {
            'haut': (-1, 0),
            'bas': (1, 0),
            'gauche': (0, -1),
            'droite': (0, 1),
        }

        ordered_directions = [initial_direction] if initial_direction in directions else []
        ordered_directions += [d for d in directions.keys() if d != initial_direction]

        for direction in ordered_directions:
            dx, dy = directions[direction] 
            new_loc = (agent.loc[0] + dx, agent.loc[1] + dy)
            if self._is_valid_location(new_loc):
                self.map[agent.loc] = 0  
                agent.loc = new_loc  
                self.map[new_loc] = 1  
                agent.moved = True
                print(f"L'agent {agent.id} a bougé vers {direction}.")
                return True  
        return False 

    def move(self, agent, direction):
        pass

    def _is_valid_location(self, loc):
        i, j = loc
        return 0 <= i < self.size[0] and 0 <= j < self.size[1] and self.map[i, j] == 0

    
