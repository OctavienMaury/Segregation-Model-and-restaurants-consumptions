import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import random
import numpy as np





class Mind:
    BATCH_SIZE = 256
    GAMMA = 0.98
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100000
    TAU = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __init__(self, input_size, num_actions, lock, queue, environment, agent=None, alpha = 0.5, destination=None, memory_length=1000000):
        self.network = DQN(input_size, num_actions)
        self.target_network = DQN(input_size, num_actions)
        self.lock = lock
        self.queue = queue
        self.losses = []
        self.network.share_memory()
        self.target_network.share_memory()
        self.recent_visit =[]
        self.alpha = alpha 
        self.environment = environment
        self.agent = agent
        self.input_size, self.num_actions = input_size, num_actions

        self.memory = ReplayMemory(memory_length)
        self.optimizer = optim.Adam(self.network.parameters(), 0.001)
        self.steps_done = 0
        self.num_actions = num_actions

        self.target_network.load_state_dict(self.network.state_dict())
        self.input_size = input_size
        self.num_cpu = mp.cpu_count() // 2


    def save(self, name, type):
        # torch.save(self.network.state_dict(), "%s/%s_network.pth" % (name, type))
        # torch.save(self.target_network.state_dict(), "%s/%s_target_network.pth" % (name, type))
        # torch.save(self.optimizer.state_dict(), "%s/%s_optimizer.pth" % (name, type))
        # states, ages, actions, next_states, rewards, dones = zip(*self.memory.memory)
        """
        np.save("%s/%s_states.npy" % (name, type), states)
        np.save("%s/%s_ages.npy" % (name, type), ages)
        np.save("%s/%s_actions.npy" % (name, type), actions)
        np.save("%s/%s_next_states.npy" % (name, type), next_states)
        np.save("%s/%s_rewards.npy" % (name, type), rewards)
        np.save("%s/%s_dones.npy" % (name, type), dones)

        np.save("%s/%s_memory_pos.npy" % (name, type), np.array([self.memory.position]))
        """
        # np.save("%s/%s_loss.npy" % (name, type), np.array(self.losses))
        pass



    def load(self, name, type, iter):
        """
        self.network.load_state_dict(torch.load("%s/%s_network.pth" % (name, type)))
        self.target_network.load_state_dict(torch.load("%s/%s_target_network.pth" % (name, type)))
        self.optimizer.load_state_dict(torch.load("%s/%s_optimizer.pth" % (name, type)))

        self.losses = list(np.load("%s/%s_loss.npy" % (name, type)))
        states = np.load("%s/%s_states.npy" % (name, type))
        ages = np.load("%s/%s_ages.npy" % (name, type))
        actions = np.load("%s/%s_actions.npy" % (name, type))
        next_states = np.load("%s/%s_next_states.npy" % (name, type))
        rewards = np.load("%s/%s_rewards.npy" % (name, type))
        dones = np.load("%s/%s_dones.npy" % (name, type))

        self.memory.memory = list(zip(states, ages, actions, next_states, rewards, dones))
        self.memory.position = int(np.load("%s/%s_memory_pos.npy" % (name, type))[0])
        self.steps_done = iter
        """
        pass

    def get_input_size(self):
        return self.input_size


    def get_output_size(self):
        return self.num_actions

    def set_agent(self, agent):
        self.agent = agent
    
    def get_losses(self):
        return self.losses

    def decide(self, state, age, type):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor([[state]], device=self.device)
                age = torch.FloatTensor([[age]], device=self.device)
                q_values = self.network(type * state, age)
                return q_values.max(1)[1].view(1, 1).detach().item()
        else:
            rand = [[random.randrange(self.num_actions)]]
            return torch.tensor(rand, device=self.device, dtype=torch.long).detach().item()
    
    def calculate_sd(self, agent, observation_window):
        s, d = 0, 0  
        for other_agent in observation_window:
            if agent.get_type == other_agent.get_type:
                s += 1 
            else:
                d += 1
        return s, d
    
    def calculate_SR(self, s, d, alpha):
        total_agents = s + d
        if total_agents == 0:  
            return 0
        similarity_ratio = s / total_agents
        tolerance_effect = 1 / (1 + np.exp(-20 * (similarity_ratio - alpha))) - 0.5
        return tolerance_effect * 20
        

    def calculate_TR(self, moved):
        """Calcul de la récompense de stillness."""
        return -1 if moved else 1
    
    def calculate_reward(self, agent, visited_restaurant, agent_type, moved):
        reward = 0

        observation_window = self.environment.get_observation_window(agent)
        s, d = self.calculate_sd(agent, observation_window)
        reward += self.calculate_SR(s, d, self.alpha)
        
        reward += self.calculate_TR(moved)
      

        if visited_restaurant not in self.recent_visit[-5:]:
            self.recent_visit.append(visited_restaurant)
            if len(self.recent_visit) > 5:
                self.recent_visit.pop(0)

        if visited_restaurant.owner_type == agent.type:
            reward  += 0.7 * (self.alpha - 0.5) 
        else:
            reward += 0.7 * (0.5 - self.alpha)
        
        return reward


    def update_rewards(self, reward):
        pass

    def remember(self, vals):
        self.memory.push(vals)

    def copy(self):
        net = DQN(self.input_size, self.num_actions)
        target_net = DQN(self.input_size, self.num_actions)
        optimizer = optim.Adam(net.parameters(), 0.001)
        optimizer.load_state_dict(self.optimizer.state_dict())
        net.load_state_dict(self.network.state_dict())
        target_net.load_state_dict(self.target_network.state_dict())

        return net, target_net, optimizer



    def opt(self, data, lock, queue, type):
        batch_state, batch_age, batch_action, batch_next_state, batch_done, expected_q_values = data
        current_q_values = self.network(type * batch_state, batch_age).gather(1, batch_action)
        max_next_q_values = self.target_network(type * batch_next_state, batch_age).detach().max(1)[0]

        for i, done in enumerate(batch_done):
            if not done:
                expected_q_values[i] += (self.GAMMA * max_next_q_values[i])

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        queue.put(loss.item())

        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.TAU * param.data + target_param.data * (1.0 - self.TAU))


    def get_data(self):
        transitions = self.memory.sample(self.BATCH_SIZE)
    
        if not transitions or len(transitions) < self.BATCH_SIZE:
            print("Pas assez de données pour l'entraînement.")
            return None

        try:
            filtered_transitions = [t for t in transitions if None not in t and t[1] is not None]
            if not filtered_transitions:
                print("Aucune transition valide après filtrage.")
                return None

            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*filtered_transitions)
        except ValueError as e:
            print(f"Erreur lors du déballage des transitions: {e}")
            return None

        if batch_action and all(action is not None for action in batch_action):
            batch_state = torch.tensor(batch_state, dtype=torch.float32)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
            batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(-1)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
            batch_done = torch.tensor(batch_done, dtype=torch.float32)
            return batch_state, batch_action, batch_reward, batch_next_state, batch_done
        else:
            print("Des actions None ont été détectées après filtrage.")
            return None


    def train(self):

        if len(self.memory) < self.BATCH_SIZE:
            print("Pas assez de données pour l'entraînement.")
            return 1  

        data = self.get_data()
        if data is None:
            print("Aucune donnée valide pour l'entraînement.")
            return 1  

        processes = []
        for _ in range(self.num_cpu):
            p = mp.Process(target=self.opt, args=(data, self.lock, self.queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        losses = []
        while not self.queue.empty():
            loss = self.queue.get()
            losses.append(loss)

        if losses:
            average_loss = sum(losses) / len(losses)
            self.losses.append(average_loss)
            print(f"Perte moyenne pour cet entraînement: {average_loss}")

        return 0  


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)  
        if len(self.memory) < self.capacity:
            self.memory.append(None)  
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        current_size = len(self.memory)
        if current_size == 0:
            raise ValueError("La mémoire est vide, impossible de prendre un échantillon.")
        batch_size = min(batch_size, current_size)  
        sampled_transitions = random.sample(self.memory, batch_size)
        return [t for t in sampled_transitions if t is not None]
    
    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    hidden = 16



    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.l1 = nn.Conv2d(1, self.hidden, kernel_size=3, padding=1)  
        self.l2 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)  
        self.l3 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)  
        self.l4 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)  
        self.l5 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)  
        self.out = nn.Linear(self.hidden + 1, num_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, age, relu=False):
        [N, a, b, c] = x.size()
        x = F.relu(self.l5(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))))
        x = x.mean(-1).mean(-1)
        x = torch.cat([x, age], dim=1)
        out = self.out(x)
        return F.relu(out) if relu else out


    def decide(self, observations):
        for business in self.environment.businesses:
            if self.can_see(business.location):  
                if not business.visited:
                    return business.visit(self.agent) 


