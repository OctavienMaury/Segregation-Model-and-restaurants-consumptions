import os
import gzip
import random
import sys
import json
import numpy as np
from environment import Environment, Business
from agent2 import Agent



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class Schelling(Environment):
    def __init__(self, size, p_business, agent_max_age, agent_range, num_actions, p_agents=0.5, name=None, num_restaurants=5, alpha=0.5, beta=1., gamma=1):
        super().__init__(size, p_business, agent_max_age, agent_range, num_actions, p_agents)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_episode = 1
        self.num_restaurants = num_restaurants
        self.directions = ['haut', 'bas', 'gauche', 'droite'] 
        self.initialize_restaurants()
        self.visit_history = []



    def initialize_restaurants(self):
        self.businesses = []  
        for _ in range(self.num_restaurants):
            location = self._choose_free_location()
            owner_type = 'A' if self.current_episode == 1 else ('B' if random.random() < 0.3 else 'A')
            restaurant = Business(location, owner_type)
            self.businesses.append(restaurant)
            self.map[location[0], location[1]] = 2 
        print(f"Restaurants initialized with their locations marked as 2 on self.map.")
        print(f"Initialized {len(self.businesses)} restaurants.")



    def step(self):
        print("Début de l'étape de simulation")
        for agent in self.agents:
            if agent.is_alive():
                decided_restaurant = agent.decide_and_visit(self)
                if decided_restaurant:  
                    self.record_visit(agent, decided_restaurant)
                else:
                    print("Aucun restaurant visité par l'agent", agent.id) 
        print("Fin de l'étape de simulation")


    def record_visit(self, agent, restaurant):
        if restaurant: 
            visit_info = {
                "agent_id": agent.id,
                "agent_type": agent.type,
                "restaurant_type": restaurant.owner_type,
                "episode": self.current_episode,
                "iteration": self.current_iteration
            }
            self.visit_history.append(visit_info)
            print("Visit recorded:", visit_info)


    def save_visit_history(self):
        print("Saving visit history:", self.visit_history)
        user_home_dir = os.path.expanduser('~')
        documents_dir = os.path.join(user_home_dir, 'Desktop', '.OctavienCODE')
        file_path = os.path.join(documents_dir, f"zvisit_history.json")
        with open(file_path, 'w') as file:
            json.dump(self.visit_history, file, indent=4)


    def save_additional_data(self, episode, iteration, businesses, agents):

        data = {

         "restaurants": [{ "location": business.location, "score": business.value, "type_owner": business.owner_type } for business in businesses],
            "agents": [{ "location": agent.loc, "reward": agent.get_reward(), "type_agent": agent.type, "last_visited_restaurant_type": agent.last_visited_restaurant} for agent in agents],  # get_reward() est hypothétique

        }

        user_home_dir = os.path.expanduser('~')
        documents_dir = os.path.join(user_home_dir, 'Desktop', '.OctavienCODE')
        file_path = os.path.join(documents_dir, f"additional_data_episode{episode}_iteration{iteration}.json")
        with open(file_path, 'w') as f:

            json.dump(data, f)



    def save(self, episode, iteration):

        print(f"Début de la sauvegarde pour l'épisode {episode}, itération {iteration}.")
        env_map = np.zeros(self.size)
        for business in self.businesses:
            i, j = business.location
            if business.owner_type == 'A':
                env_map[i, j] = 3
            elif business.owner_type == 'B':
                env_map[i, j] = 4

        for agent in self.agents:
            i, j = agent.loc
            if agent.type == 'A':
                env_map[i, j] = 1  
            elif agent.type == 'B':
                env_map[i, j] = 2 



        user_home_dir = os.path.expanduser('~')
        documents_dir = os.path.join(user_home_dir, '', '')
        spatial_file_path = os.path.join(documents_dir, f"z100.15x15.alpha1_episode{episode}_iteration{iteration}_spatial.npy.gz")
        with gzip.open(spatial_file_path, 'wb') as f:

            np.save(f, env_map)




        additional_data = {
            "restaurants": [{"location": business.location, "score": business.value, "type_owner": business.owner_type} for business in self.businesses],
            "agents": [{"location": agent.loc, "reward": agent.get_last_reward(), "type_agent" : agent.type, "last_visited_restaurant_type": agent.last_visited_restaurant} for agent in self.agents]
        }


        additional_file_path = os.path.join(documents_dir, f"z100.15x15.alpha1_episode{episode}_iteration{iteration}_additional.json")
        with open(additional_file_path, 'w') as f:
            json.dump(additional_data, f)



    print("Sauvegarde terminée.")


def play(society, episodes, iterations):

    save_iterations = [1, 20, 40,  60, 80, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000 ]  # Exemple d'itérations à sauvegarder
    for episode in range(episodes):
        society.current_episode = episode + 1
        society.initialize_restaurants()     
        for iteration in range(iterations):
            print(f"Début de l'itération {iteration+1} de l'épisode {episode+1}.")
            society.step()
            print(f"Fin de l'itération {iteration+1} de l'épisode {episode+1}.")
            if iteration + 1 in save_iterations:  
                society.save(episode, iteration)
    society.save_visit_history()
    print("SIMULATION IS FINISHED.")



if __name__ == '__main__':


    name = "testestes5"  
    iterations =5000  
    episodes = 2  
    agent_max_age = 80  
    agent_range = 5 
    num_actions = 4 
    num_restaurants = 5  
    alpha = 0.5 
    beta = 1 
    gamma = 1  



    if len(sys.argv) > 1:
        try:
            name = sys.argv[1]
            iterations = int(sys.argv[2])
            agent_range = int(sys.argv[3])
            agent_max_age = int(sys.argv[4])
            alpha = float(sys.argv[5])
            beta = float(sys.argv[6])
            gamma = float(sys.argv[7])
            num_restaurants = int(sys.argv[8])
        except (IndexError, ValueError) as e:
            print("Erreur dans les arguments de la ligne de commande. Utilisation des valeurs par défaut.")
    society = Schelling(size=(10, 10), p_business=0.1, agent_max_age=80, agent_range=3, num_actions=4, p_agents=0.5, num_restaurants=5, alpha=0.5, beta=1., gamma=1)
    play(society, episodes=2, iterations=5000)


     
