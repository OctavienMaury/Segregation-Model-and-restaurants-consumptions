import numpy as np
import torch


class Agent:

    def __init__(self, id, loc, type, mind, p, max_time, tolerance, agent_range, alpha=0.5, recent_visits=None):
        self.id = id
        self.alive = True
        self.loc = loc
        self.type = type
        self.alpha =alpha
        self.current_state = None
        self.action = None
        self.next_state = None
        self.p_void = p
        self.mind = mind
        self.input_size = mind.get_input_size()
        self.output_size = mind.get_output_size()
        self.age = 0
        self.moved = False
        self.max_time = max_time
        self.time_remaining = max_time
        self.decision = None
        self.tolerance = tolerance
        self.agent_range = agent_range
        self.recently_visited_restaurants = []
        self.last_reward = 0
        self.last_visited_restaurant = None
        self.recently_visited_restaurants = recent_visits if recent_visits else []


    def update(self, reward, done):
        assert reward is not None, 'No Reward'
        if self.action == 'none':
            print(f"Agent {self.id}: Aucune action significative à mettre à jour.")
            self.last_reward = reward
            if not done:
                self.current_state = self.next_state
            self.next_state = None
            return  
    
        current_state_tensor = torch.tensor(self.current_state, dtype=torch.float32)
        next_state_tensor = torch.tensor(self.next_state, dtype=torch.float32)

        self.mind.memory.push(current_state_tensor, self.action, reward, next_state_tensor, done)

        self.last_reward = reward
        loss = self.mind.train()
        self.action = None
        if not done:
            self.current_state, self.next_state = self.next_state, None
        else:
            self.current_state, self.next_state = None, None
        print(f"Agent {self.id} a mis à jour son état après une visite avec une récompense de {reward}.")



    def get_observation_window(self, environment):
        return environment.get_observation_window(self)
    
    
    def get_losses(self):
        return self.mind.get_losses()

    def agent_step(self):
        if not self.a_visite_restaurant:
            self.a_visite_restaurant = True
        else:
            pass

    def get_last_reward(self):
        return self.last_reward

    def get_state_representation(self, environment, post_visit=False):
        state = []

        nearby_restaurants = []
        nearby_agents = []
        for restaurant in environment.businesses:
            distance = np.linalg.norm(np.array(restaurant.location) - np.array(self.loc))
            if distance <= self.agent_range:
                nearby_restaurants.append(restaurant)

        for agent in environment.agents:
            if agent.id != self.id:  
                distance = np.linalg.norm(np.array(agent.loc) - np.array(self.loc))
                if distance <= self.agent_range:
                    nearby_agents.append(agent)

        state.append(len(nearby_restaurants))
        state.append(len(nearby_agents))


        return np.array(state)
 
    def decide_and_visit(self, environment):
        self.current_state = self.get_state_representation(environment)
        self.moved = False
        accessible_restaurants = [restaurant for restaurant in environment.get_accessible_restaurants(self)
                                  if restaurant not in self.recently_visited_restaurants]  
        if not accessible_restaurants:  
            accessible_restaurants = environment.get_accessible_restaurants(self) 

        best_total_reward = -np.inf
        best_action = None
        restaurant_to_visit = None

        for restaurant in accessible_restaurants:
            self.next_state = self.get_state_representation(environment, post_visit=False)
            total_reward = self.mind.calculate_reward(self, restaurant, self.type, self.moved)
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                restaurant_to_visit = restaurant
                best_action = self.calculate_optimal_direction_to(restaurant.location)

        if best_action and restaurant_to_visit:
            successful_move = environment.try_move(self, best_action)
            self.moved = successful_move
            if successful_move:
                reward = restaurant_to_visit.visit(self)
                self.recently_visited_restaurants.append(restaurant_to_visit) 
                if len(self.recently_visited_restaurants) > 5:
                    self.recently_visited_restaurants.pop(0)  
                self.next_state = self.get_state_representation(environment, post_visit=True)
                self.update(reward, False)
                self.last_visited_restaurant = restaurant_to_visit.owner_type
            else:
                print(f"L'agent {self.id} n'a pas pu se déplacer.")
        else:
            self.next_state = self.current_state
            self.update(0, False)


    def visit_restaurant(self, restaurant):
        self.recently_visited_restaurants.append(restaurant)
        if len(self.recently_visited_restaurants) > 5:
            self.recently_visited_restaurants.pop(0)

        self.mind.update_recent_visits(self.recently_visited_restaurants)

        reward = restaurant.visit(self)
        self.update(reward, False)

    
    def get_state(self):
        return self.current_state

    def calculate_optimal_direction_to(self, target_location):
        x_diff = target_location[0] - self.loc[0]
        y_diff = target_location[1] - self.loc[1]
    
        if abs(x_diff) > abs(y_diff):
            return 'haut' if x_diff < 0 else 'bas'
        else:
            return 'gauche' if y_diff < 0 else 'droite'
        
    def get_state(self):
        return self.current_state

    def get_age(self):
        return self.age


    def get_time_remaining(self):
        return self.time_remaining


    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc

    def get_type(self):
        return self.type

    def get_decision(self):
        assert self.decision != None, "Decision is requested without setting."
        return self.decision


    def set_decision(self, decision):
        self.decision = decision


    def clear_decision(self):
        self.decision = None


    def set_loc(self, loc):
        self.loc = loc


    def set_current_state(self, state):
        self.current_state = state


    def set_next_state(self, state):
        self.next_state = state


    def reset(self):
        self.time_remaining = self.max_time
        self.age = 0

    def is_alive(self):
        return self.alive

    
