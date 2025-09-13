import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class TaxiDQN():
    discount_factor_g = 0.99
    replay_memory_size = 75000
    mini_batch_size = 64

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['S','N','E','W','P','D']

    def __init__(self, is_rainy=False, fickle_passenger=False):
        """
        Initialize TaxiDQN with stochastic environment options
        
        Args:
            is_rainy (bool): If True, movement actions have 80% success rate, 
                             10% chance each of moving left/right of intended direction
            fickle_passenger (bool): If True, passenger has 30% chance of changing 
                                   destinations after first movement away from pickup
        """
        self.is_rainy = is_rainy
        self.fickle_passenger = fickle_passenger
        self.is_stochastic = is_rainy or fickle_passenger
        
        # Adaptive hyperparameters based on environment complexity
        # Deterministic environment
        if not self.is_stochastic:
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.hidden_nodes = 128
            self.max_grad_norm = 5.0     
            self.default_episodes = 1000 
            self.learning_rate_a = 0.0008 

        # Stochastic environments
        else:
            self.epsilon_min = 0.05          
            self.epsilon_decay = 0.9985      
            self.hidden_nodes = 128          
            self.max_grad_norm = 5.0         
            self.default_episodes = 2000     
            self.learning_rate_a = 0.0012    
        
        print(f"Environment: is_rainy={is_rainy}, fickle_passenger={fickle_passenger}")
        print(f"Hyperparameters: eps_min={self.epsilon_min}, eps_decay={self.epsilon_decay}, "
              f"hidden={self.hidden_nodes}, episodes={self.default_episodes}")

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.default_episodes
        # Create environment with stochastic parameters
        env = gym.make('Taxi-v3', is_rainy=self.is_rainy, fickle_passenger=self.fickle_passenger)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        epsilon = 1.0
        memory = ReplayMemory(self.replay_memory_size)
        policy_dqn = DQN(in_states=num_states, h1_nodes=self.hidden_nodes, 
                        h2_nodes=self.hidden_nodes, out_actions=num_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        env_type = "Stochastic" if self.is_stochastic else "Deterministic"
        print(f'\nTraining {env_type} Policy:')
        print(f'Network: {num_states} -> {self.hidden_nodes} -> {self.hidden_nodes} -> {num_actions}')
        print(f'Device: {DEVICE}')
        rewards_per_episode = []
        epsilon_history = []
        episode_lengths = []
        successful_episodes = []
        for i in range(episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            while not terminated and not truncated:
                episode_length += 1
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = self.state_to_dqn_input(state, num_states).unsqueeze(0).to(DEVICE)
                        q_values = policy_dqn(state_tensor)
                        action = q_values.argmax(dim=1).item()
                new_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn)
            rewards_per_episode.append(episode_reward)
            episode_lengths.append(episode_length)
            successful_episodes.append(1 if episode_reward > 0 else 0)
            if epsilon > self.epsilon_min:
                epsilon *= self.epsilon_decay
            epsilon_history.append(epsilon)
            if (i + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                success_rate = np.mean(successful_episodes[-100:]) * 100
                print(f"Episode {i+1:4d}: Reward={avg_reward:6.2f}, Length={avg_length:5.1f}, "
                      f"Success={success_rate:5.1f}%, ε={epsilon:.3f}")
        env.close()
        # Save model
        model_name = f"taxi_dqn{'_stochastic' if self.is_stochastic else ''}.pt"
        torch.save(policy_dqn.state_dict(), model_name)
        self.plot_training_results(rewards_per_episode, epsilon_history, episode_lengths, successful_episodes)
        final_success_rate = np.mean(successful_episodes[-100:]) * 100
        final_avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"\nTraining Complete!")
        print(f"Final 100-episode average: Reward={final_avg_reward:.2f}, Success Rate={final_success_rate:.1f}%")

    def plot_training_results(self, rewards, epsilon_history, episode_lengths, successful_episodes):
        plt.figure(figsize=(15, 10))
        
        # Plot rewards with moving average
        plt.subplot(221)
        plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(rewards) >= 100:
            moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
            plt.plot(moving_avg, color='red', linewidth=2, label='100-Episode Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        env_type = "Stochastic" if self.is_stochastic else "Deterministic"
        plt.title(f'Training Rewards - {env_type} Environment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success rate
        plt.subplot(222)
        if len(successful_episodes) >= 100:
            success_rate_ma = [np.mean(successful_episodes[max(0, i-99):i+1]) * 100 
                             for i in range(len(successful_episodes))]
            plt.plot(success_rate_ma, color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate (100-Episode Moving Average)')
        plt.grid(True, alpha=0.3)
        
        # Plot episode lengths
        plt.subplot(223)
        plt.plot(episode_lengths, alpha=0.3, color='purple', label='Episode Length')
        if len(episode_lengths) >= 100:
            moving_avg_length = [np.mean(episode_lengths[max(0, i-99):i+1]) 
                               for i in range(len(episode_lengths))]
            plt.plot(moving_avg_length, color='orange', linewidth=2, label='100-Episode Average')
        plt.xlabel('Episode')
        plt.ylabel('Steps to Complete')
        plt.title('Episode Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        plt.subplot(224)
        plt.plot(epsilon_history, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.title('Exploration Rate Decay')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"taxi_dqn{'_stochastic' if self.is_stochastic else ''}_training.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def optimize(self, mini_batch, policy_dqn):
        """Optimize the DQN with experience replay"""
        num_states = policy_dqn.fc1.in_features

        states = [t[0] for t in mini_batch]
        actions = torch.tensor([t[1] for t in mini_batch], dtype=torch.long, device=DEVICE)
        next_states = [t[2] for t in mini_batch]
        rewards = torch.tensor([t[3] for t in mini_batch], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([t[4] for t in mini_batch], dtype=torch.bool, device=DEVICE)

        state_batch = self.state_to_dqn_input(states, num_states).to(DEVICE)
        next_state_batch = self.state_to_dqn_input(next_states, num_states).to(DEVICE)

        pred_q_all = policy_dqn(state_batch)
        pred_q = pred_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = policy_dqn(next_state_batch)
            max_next_q, _ = next_q_all.max(dim=1)
            target_q = rewards + (~dones).float() * (self.discount_factor_g * max_next_q)

        loss = self.loss_fn(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

    def state_to_dqn_input(self, states, num_states: int) -> torch.Tensor:
        """Convert integer state or list of states to one-hot float tensor."""
        if isinstance(states, (list, tuple, np.ndarray)):
            idx = torch.tensor(states, dtype=torch.long)
            one_hot = F.one_hot(idx, num_classes=num_states).float()
            return one_hot
        else:
            idx = torch.tensor([states], dtype=torch.long)
            one_hot = F.one_hot(idx, num_classes=num_states).float().squeeze(0)
            return one_hot

    def test(self, episodes=5, render=False):
        """Test the trained policy"""
        render_mode = 'human' if render else None
        env = gym.make('Taxi-v3', is_rainy=self.is_rainy, fickle_passenger=self.fickle_passenger, render_mode=render_mode)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        # Load appropriate model
        model_name = f"taxi_dqn{'_stochastic' if self.is_stochastic else ''}.pt"
        policy_dqn = DQN(in_states=num_states, h1_nodes=self.hidden_nodes, \
                        h2_nodes=self.hidden_nodes, out_actions=num_actions).to(DEVICE)
        if not os.path.exists(model_name):
            print(f"No trained model found ({model_name}). Please run training first.")
            return
        policy_dqn.load_state_dict(torch.load(model_name, map_location=DEVICE))
        policy_dqn.eval()
        env_type = "stochastic" if self.is_stochastic else "deterministic"
        print(f'\nTesting trained policy on {env_type} environment...')
        total_rewards = []
        total_steps = []
        successful_deliveries = 0
        for episode in range(episodes):
            state, info = env.reset()
            if render:
                env.render()
            terminated = False
            truncated = False
            episode_reward = 0
            steps = 0
            print(f"\nEpisode {episode + 1}:")
            while not terminated and not truncated and steps < 500:
                with torch.no_grad():
                    state_tensor = self.state_to_dqn_input(state, num_states).unsqueeze(0).to(DEVICE)
                    q_values = policy_dqn(state_tensor)
                    action = q_values.argmax(dim=1).item()
                # print(f"  Step {steps + 1}: Action {self.ACTIONS[action]}")
                if render:
                    env.render()
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            if episode_reward > 0:
                successful_deliveries += 1
                print(f"  ✓ SUCCESS! Reward: {episode_reward}, Steps: {steps}")
            else:
                print(f"  ✗ Failed. Reward: {episode_reward}, Steps: {steps}")
        env.close()
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        success_rate = successful_deliveries / episodes * 100
        print(f"\n{'='*50}")
        print(f"TEST RESULTS ({env_type.upper()} ENVIRONMENT)")
        print(f"{'='*50}")
        print(f"Episodes:      {episodes}")
        print(f"Success Rate:  {success_rate:.1f}% ({successful_deliveries}/{episodes})")
        print(f"Avg Reward:    {avg_reward:.2f}")
        print(f"Avg Steps:     {avg_steps:.1f}")
        print(f"Reward Range:  {min(total_rewards):.1f} to {max(total_rewards):.1f}")

if __name__ == '__main__':
    # Choose one configuration to test
    print("Choose environment configuration:")
    print("1. Deterministic (original)")
    print("2. Rainy weather only") 
    print("3. Fickle passenger only")
    print("4. Both rainy and fickle (hardest)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    config_map = {
        '1': (False, False, "deterministic"),
        '2': (True, False, "rainy"), 
        '3': (False, True, "fickle"),
        '4': (True, True, "rainy_and_fickle")
    }
    
    if choice not in config_map:
        print("Invalid choice, using deterministic environment")
        choice = '1'
    
    is_rainy, fickle_passenger, name = config_map[choice]
    
    print(f"\n{'='*60}")
    print(f"TRAINING {name.upper().replace('_', ' + ')} TAXI ENVIRONMENT")
    print(f"{'='*60}")
    
    taxi_agent = TaxiDQN(is_rainy=is_rainy, fickle_passenger=fickle_passenger)
    
    # Train the agent
    taxi_agent.train()
    
    # Test the trained agent
    print("\nWould you like to visualize the testing? (y/n): ", end="")
    show_render = input().strip().lower() == 'y'
    
    if show_render:
        print("Rendering enabled - you'll see a visual window!")
        taxi_agent.test(episodes=5, render=True)
    else:
        taxi_agent.test(episodes=5, render=False)