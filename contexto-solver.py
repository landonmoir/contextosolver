import numpy as np
import random
from collections import defaultdict
import logging
import pandas as pd
# Add matplotlib imports
import matplotlib
# Use Agg backend to work without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Set up logging with more visible format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Ensure our logging configuration takes precedence
)

class ContextoGame:
    def __init__(self, word_vectors: Dict[str, np.ndarray]):
        """Initialize game with word vectors."""
        self.wordmap = word_vectors
        self.secret = None
        logging.info(f"Initialized game with {len(word_vectors)} words")
        self.reset()

    def reset(self) -> str:
        """Reset the game with a new secret word."""
        self.secret = random.choice(list(self.wordmap.keys()))
        return self.secret

    def guess(self, word: str) -> int:
        """Make a guess and return the number of words closer to the target."""
        if word not in self.wordmap:
            return len(self.wordmap)
        
        secret_vec = self.wordmap[self.secret]
        guess_vec = self.wordmap[word]
        guess_distance = np.linalg.norm(guess_vec - secret_vec)
        
        words_closer = sum(1 for w in self.wordmap.keys()
            if np.linalg.norm(self.wordmap[w] - secret_vec) < guess_distance)
        return words_closer

class ContextoAgent:
    def __init__(self, word_vectors: Dict[str, np.ndarray], 
            learning_rate: float = 0.2,
            discount_factor: float = 0.9,
            epsilon: float = 1.0,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.999):
        self.wordmap = word_vectors
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table and word distance cache
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.distance_cache = {}
        
        # Precompute embedding matrix for faster similarity computations
        logging.info("Precomputing embedding matrix...")
        start_time = time.time()
        self.words = list(word_vectors.keys())
        self.embedding_matrix = np.vstack([word_vectors[word] for word in self.words])
        logging.info(f"Embedding matrix computed in {time.time() - start_time:.2f} seconds")
        
    def get_state(self, last_guess: Optional[str], words_closer: Optional[int]) -> Tuple[str, int]:
        """Convert current game state to a state representation."""
        if last_guess is None:
            return ('', -1)
        return (last_guess, words_closer)
    
    def get_similar_words(self, word: str, n: int = 5) -> List[str]:
        """Find n most similar words to the given word."""
        if word not in self.distance_cache:
            word_vec = self.wordmap[word]
            distances = np.linalg.norm(self.embedding_matrix - word_vec, axis=1)
            similar_indices = np.argsort(distances)[1:n+1]
            self.distance_cache[word] = [self.words[i] for i in similar_indices]
        return self.distance_cache[word]
    
    def choose_action(self, state: Tuple[str, int], previous_guesses: List[str]) -> str:
        """Choose next word to guess using epsilon-greedy strategy."""
        available_words = [w for w in self.words if w not in previous_guesses]
        if not available_words:
            logging.warning("No available words left to guess!")
            return random.choice(self.words)  # Fallback to any word if somehow we run out
            
        if random.random() < self.epsilon:
            if state[0] and random.random() < 0.5:
                candidates = self.get_similar_words(state[0])
                candidates = [w for w in candidates if w not in previous_guesses]
                if candidates:
                    return random.choice(candidates)
            return random.choice(available_words)
        else:
            available_actions = {w: self.q_table[state][w] 
                for w in available_words}
            if not available_actions:
                return random.choice(available_words)
            return max(available_actions.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple[str, int], action: str, reward: float, 
            next_state: Tuple[str, int], done: bool) -> None:
        """Update Q-values using Q-learning update rule."""
        if done:
            next_max_q = 0
        else:
            next_state_values = self.q_table[next_state].values()
            next_max_q = max(next_state_values) if next_state_values else 0
            
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, game: ContextoGame, max_steps: int = 100) -> Tuple[int, List[str], List[int]]:
        """Train for one episode."""
        start_time = time.time()
        secret_word = game.reset()
        state = self.get_state(None, None)
        previous_guesses = []
        distances = []
        
        for step in range(max_steps):
            action = self.choose_action(state, previous_guesses)
            previous_guesses.append(action)
            
            words_closer = game.guess(action)
            distances.append(words_closer)
            done = words_closer == 0
            
            reward = -words_closer if not done else 2000
            next_state = self.get_state(action, words_closer)
            self.update(state, action, reward, next_state, done)
            
            if done:
                episode_time = time.time() - start_time
                logging.debug(f"Found word '{secret_word}' in {step + 1} steps, "
                            f"time: {episode_time:.2f}s")
                return step + 1, previous_guesses, distances
            
            state = next_state
            
            # Add timeout warning
            if time.time() - start_time > 60:  # 1 minute timeout
                logging.warning(f"Episode taking too long! Current step: {step}")
        
        logging.warning(f"Failed to find word '{secret_word}' in {max_steps} steps")
        return max_steps, previous_guesses, distances

def load_word_vectors(file_path: str) -> Dict[str, np.ndarray]:
    """Load word vectors from file."""
    logging.info(f"Loading word vectors from {file_path}...")
    start_time = time.time()
    try:
        wordmap = {}
        with open(file_path) as infile:
            for line in infile:
                split = line.strip().split(" ")
                word = split[0]
                vecs = np.array([float(x) for x in split[1:]])
                wordmap[word] = vecs
        logging.info(f"Loaded {len(wordmap)} word vectors in {time.time() - start_time:.2f} seconds")
        return wordmap
    except Exception as e:
        logging.error(f"Error loading word vectors: {str(e)}")
        raise

def plot_learning_progress(df: pd.DataFrame, save_path: str = 'learning_progress.png'):
    """Plot learning metrics to visualize improvement over time."""
    # Set style for better-looking plots
    plt.style.use('seaborn')
    
    # Create figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate rolling averages for smoother plots
    window = 50
    rolling_steps = df['Steps'].rolling(window=window, min_periods=1).mean()
    rolling_success = df['Found_Word'].rolling(window=window, min_periods=1).mean() * 100
    rolling_distance = df['Average_Distance'].rolling(window=window, min_periods=1).mean()
    
    # Plot average steps per episode
    ax1.plot(df['Episode'], df['Steps'], 'b.', alpha=0.1, label='Raw')
    ax1.plot(df['Episode'], rolling_steps, 'r-', label=f'{window}-ep average')
    ax1.set_title('Steps per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax1.legend()
    
    # Plot success rate
    ax2.plot(df['Episode'], df['Found_Word'] * 100, 'b.', alpha=0.1, label='Raw')
    ax2.plot(df['Episode'], rolling_success, 'r-', label=f'{window}-ep average')
    ax2.set_title('Success Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.legend()
    
    # Plot average distance
    ax3.plot(df['Episode'], df['Average_Distance'], 'b.', alpha=0.1, label='Raw')
    ax3.plot(df['Episode'], rolling_distance, 'r-', label=f'{window}-ep average')
    ax3.set_title('Average Distance per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Distance')
    ax3.legend()
    
    # Plot final distances histogram
    ax4.hist(df['Final_Distance'], bins=50, color='blue', alpha=0.7)
    ax4.set_title('Distribution of Final Distances')
    ax4.set_xlabel('Final Distance')
    ax4.set_ylabel('Count')
    
    # Adjust layout and save
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Learning progress plots saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving plot: {str(e)}")
    finally:
        plt.close(fig)

def save_additional_plots(df: pd.DataFrame):
    """Save additional individual plots for more detailed analysis."""
    try:
        # Plot epsilon decay
        plt.figure(figsize=(10, 6))
        plt.plot(df['Episode'], df['Epsilon'], 'b-')
        plt.title('Epsilon Decay Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.savefig('epsilon_decay.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot success rate heatmap by episode ranges
        plt.figure(figsize=(12, 6))
        episode_ranges = range(0, len(df), 50)
        success_rates = [df['Found_Word'][i:i+50].mean() * 100 for i in episode_ranges]
        plt.bar(range(len(success_rates)), success_rates, alpha=0.7)
        plt.title('Success Rate by Episode Ranges (50 episodes each)')
        plt.xlabel('Episode Range')
        plt.ylabel('Success Rate (%)')
        plt.xticks(range(len(success_rates)), [f'{i}-{i+49}' for i in episode_ranges], rotation=45)
        plt.grid(True)
        plt.savefig('success_rate_ranges.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Additional plots saved successfully")
    except Exception as e:
        logging.error(f"Error saving additional plots: {str(e)}")

def main():
    # Load word vectors
    word_vectors = load_word_vectors('./vectors.txt')
    
    # Initialize game and agent
    game = ContextoGame(word_vectors)
    agent = ContextoAgent(word_vectors)
    
    # Training parameters
    episodes = 200
    max_steps = 200
    
    # Training loop
    episode_data = []
    start_time = time.time()
    logging.info("Starting training...")
    
    # Track performance metrics
    best_episode = None
    best_steps = float('inf')
    success_streak = 0
    best_streak = 0
    
    try:
        for episode in range(episodes):
            episode_start = time.time()
            steps, guesses, distances = agent.train_episode(game, max_steps)
            episode_time = time.time() - episode_start
            
            # Update performance tracking
            if distances[-1] == 0:  # Successful guess
                success_streak += 1
                best_streak = max(best_streak, success_streak)
                if steps < best_steps:
                    best_steps = steps
                    best_episode = episode + 1
            else:
                success_streak = 0
            
            # Record episode data
            episode_data.append({
                'Episode': episode + 1,
                'Steps': steps,
                'Final_Distance': distances[-1],
                'Average_Distance': np.mean(distances),
                'Guesses': len(guesses),
                'Found_Word': distances[-1] == 0,
                'Secret_Word': game.secret,
                'Time': episode_time,
                'Epsilon': agent.epsilon
            })
            
            # Log progress more frequently at start, then every 10 episodes
            if episode < 10 or (episode + 1) % 10 == 0:
                recent_success_rate = (sum(1 for i in episode_data[-10:] 
                    if i['Found_Word']) / min(10, len(episode_data))) * 100
                logging.info(
                    f"Episode {episode + 1}/{episodes}: "
                    f"Steps={steps}, "
                    f"Final Distance={distances[-1]}, "
                    f"Epsilon={agent.epsilon:.3f}, "
                    f"Recent Success Rate={recent_success_rate:.1f}%, "
                    f"Time={episode_time:.2f}s"
                )
            
            # Add some basic progress indication every episode
            if (episode + 1) % 1 == 0:
                print(".", end="", flush=True)
                if (episode + 1) % 50 == 0:
                    print(f" {episode + 1}")
    
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user. Saving partial results...")
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise
    finally:
        # Save whatever results we have
        if episode_data:
            df = pd.DataFrame(episode_data)
            df.to_csv('training_results.csv', index=False)
            logging.info(f"Results saved to training_results.csv")
            
            # Plot learning progress
            plot_learning_progress(df)
            
            # Save additional plots
            save_additional_plots(df)
            
            # Plot learning progress
            plot_learning_progress(df)
            
            # Print summary statistics
            total_time = time.time() - start_time
            success_rate = (df['Found_Word'].sum() / len(df)) * 100
            avg_steps = df['Steps'].mean()
            recent_success_rate = (df['Found_Word'].tail(100).mean()) * 100
            
            logging.info(f"\nTraining Summary:")
            logging.info(f"Total time: {total_time:.2f} seconds")
            logging.info(f"Episodes completed: {len(df)}")
            logging.info(f"Overall success rate: {success_rate:.1f}%")
            logging.info(f"Recent success rate (last 100 episodes): {recent_success_rate:.1f}%")
            logging.info(f"Average steps per episode: {avg_steps:.1f}")
            logging.info(f"Best episode: #{best_episode} ({best_steps} steps)")
            logging.info(f"Longest success streak: {best_streak}")
            logging.info(f"Final epsilon value: {agent.epsilon:.3f}")

if __name__ == "__main__":
    main()