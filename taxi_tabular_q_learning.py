import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class TaxiQLearning:
    def __init__(self, is_rainy=False, fickle_passenger=False):
        
        """
        Initialize Tabular Q-Learning with stochastic environment options
        
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
        if self.is_stochastic:
            self.learning_rate_a = 0.7           # Slightly lower learning rate for stability
            self.discount_factor_g = 0.98        # Higher discount factor to value future rewards more in a stochastic setting
            self.epsilon_decay_rate = 0.0001     
            self.min_learning_rate = 0.05        # Prevent learning rate from dropping too low
            self.default_episodes = 20000        # More episodes for convergence in stochastic environment
        else:
            self.learning_rate_a = 0.75           # Your original values
            self.discount_factor_g = 0.9
            self.epsilon_decay_rate = 0.0001
            self.min_learning_rate = 0.0001
            self.default_episodes = 15000

        self.ACTIONS = ['S', 'N', 'E', 'W', 'P', 'D']
        
        print(f"Environment: is_rainy={is_rainy}, fickle_passenger={fickle_passenger}")
        print(f"Hyperparameters: α={self.learning_rate_a}, γ={self.discount_factor_g}, "
              f"ε_decay={self.epsilon_decay_rate}, episodes={self.default_episodes}")

    def run(self, episodes=None, is_training=True, render=False):
        if episodes is None:
            episodes = self.default_episodes if is_training else 5

        # Create environment with stochastic parameters
        render_mode = 'human' if render else None
        env = gym.make('Taxi-v3', is_rainy=self.is_rainy, 
                      fickle_passenger=self.fickle_passenger, render_mode=render_mode)

        # Load or initialize Q-table
        model_name = f"taxi{'_stochastic' if self.is_stochastic else ''}.pkl"
        
        if is_training:
            q = np.zeros((env.observation_space.n, env.action_space.n))  # 500 x 6 array
        else:
            if not os.path.exists(model_name):
                print(f"No trained model found ({model_name}). Please run training first.")
                return
            with open(model_name, 'rb') as f:
                q = pickle.load(f)

        # Initialize training parameters
        learning_rate_a = self.learning_rate_a
        discount_factor_g = self.discount_factor_g
        epsilon = 1.0
        epsilon_decay_rate = self.epsilon_decay_rate
        rng = np.random.default_rng()

        # Tracking arrays
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        episode_lengths = []
        successful_episodes = []

        env_type = "Stochastic" if self.is_stochastic else "Deterministic"
        if is_training:
            print(f'\nTraining {env_type} Tabular Q-Learning Policy:')
            print(f'Q-table size: {env.observation_space.n} x {env.action_space.n}')

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            if not is_training:
                print(f"\nEpisode {i + 1}:")

            while not terminated and not truncated:
                episode_length += 1

                if is_training and rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state, :])

                # if not is_training:
                #     print(f"  Step {episode_length}: Action {self.ACTIONS[action]}")

                new_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if is_training:
                    # Q-learning update
                    q[state, action] = q[state, action] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                    )

                state = new_state

            # Decay epsilon and adjust learning rate
            if is_training:
                epsilon = max(epsilon - epsilon_decay_rate, 0)
                if epsilon == 0:
                    learning_rate_a = self.min_learning_rate

            rewards_per_episode[i] = episode_reward
            epsilon_history.append(epsilon)
            episode_lengths.append(episode_length)
            successful_episodes.append(1 if episode_reward > 0 else 0)

            # Progress reporting during training
            if is_training and (i + 1) % 1000 == 0:
                avg_reward = np.mean(rewards_per_episode[max(0, i-999):i+1])
                avg_length = np.mean(episode_lengths[max(0, i-999):i+1])
                success_rate = np.mean(successful_episodes[max(0, i-999):i+1]) * 100
                print(f"Episode {i+1:5d}: Reward={avg_reward:6.2f}, Length={avg_length:5.1f}, "
                      f"Success={success_rate:5.1f}%, ε={epsilon:.4f}, α={learning_rate_a:.4f}")

            # Testing episode results
            if not is_training:
                if episode_reward > 0:
                    print(f"  ✓ SUCCESS! Reward: {episode_reward}, Steps: {episode_length}")
                else:
                    print(f"  ✗ Failed. Reward: {episode_reward}, Steps: {episode_length}")

        env.close()

        # Save model after training
        if is_training:
            with open(model_name, "wb") as f:
                pickle.dump(q, f)
            
            self.plot_training_results(rewards_per_episode, epsilon_history, 
                                     episode_lengths, successful_episodes)

            final_success_rate = np.mean(successful_episodes[-1000:]) * 100
            final_avg_reward = np.mean(rewards_per_episode[-1000:])
            print(f"\nTraining Complete!")
            print(f"Final 1000-episode average: Reward={final_avg_reward:.2f}, Success Rate={final_success_rate:.1f}%")

        # Testing statistics
        if not is_training:
            self.print_test_statistics(rewards_per_episode, episode_lengths, successful_episodes)

        return rewards_per_episode

    def plot_training_results(self, rewards, epsilon_history, episode_lengths, successful_episodes):
        """Plot comprehensive training results"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards with moving average
        plt.subplot(221)
        plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(rewards) >= 1000:
            moving_avg = [np.mean(rewards[max(0, i-999):i+1]) for i in range(len(rewards))]
            plt.plot(moving_avg, color='red', linewidth=2, label='1000-Episode Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        env_type = "Stochastic" if self.is_stochastic else "Deterministic"
        plt.title(f'Training Rewards - {env_type} Environment (Tabular Q-Learning)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success rate
        plt.subplot(222)
        if len(successful_episodes) >= 1000:
            success_rate_ma = [np.mean(successful_episodes[max(0, i-999):i+1]) * 100 
                             for i in range(len(successful_episodes))]
            plt.plot(success_rate_ma, color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate (1000-Episode Moving Average)')
        plt.grid(True, alpha=0.3)
        
        # Plot episode lengths
        plt.subplot(223)
        plt.plot(episode_lengths, alpha=0.3, color='purple', label='Episode Length')
        if len(episode_lengths) >= 1000:
            moving_avg_length = [np.mean(episode_lengths[max(0, i-999):i+1]) 
                               for i in range(len(episode_lengths))]
            plt.plot(moving_avg_length, color='orange', linewidth=2, label='1000-Episode Average')
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
        filename = f"taxi_qlearning{'_stochastic' if self.is_stochastic else ''}_training.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def print_test_statistics(self, rewards, episode_lengths, successful_episodes):
        """Print detailed test statistics"""
        episodes = len(rewards)
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(episode_lengths)
        success_rate = np.mean(successful_episodes) * 100
        successful_deliveries = int(np.sum(successful_episodes))

        env_type = "stochastic" if self.is_stochastic else "deterministic"
        
        print(f"\n{'='*50}")
        print(f"TEST RESULTS ({env_type.upper()} ENVIRONMENT - TABULAR Q-LEARNING)")
        print(f"{'='*50}")
        print(f"Episodes:      {episodes}")
        print(f"Success Rate:  {success_rate:.1f}% ({successful_deliveries}/{episodes})")
        print(f"Avg Reward:    {avg_reward:.2f}")
        print(f"Avg Steps:     {avg_steps:.1f}")
        print(f"Reward Range:  {min(rewards):.1f} to {max(rewards):.1f}")

    def train(self, episodes=None):
        """Train the Q-learning agent"""
        return self.run(episodes=episodes, is_training=True, render=False)

    def test(self, episodes=5, render=False):
        """Test the trained Q-learning agent"""
        return self.run(episodes=episodes, is_training=False, render=render)

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
    
    # Create Q-learning agent
    taxi_agent = TaxiQLearning(is_rainy=is_rainy, fickle_passenger=fickle_passenger)
    
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

