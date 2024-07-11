import numpy as np
import random
from collections import defaultdict
import json
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextoGame:
    def __init__(self, word_vectors):
        self.wordmap = word_vectors
        self.reset()

    def reset(self):
        self.secret = random.choice(list(self.wordmap.keys()))
        return self.secret

    def get_distance(self, guess):
        secret_vec = self.wordmap[self.secret]
        guess_vec = self.wordmap[guess]

        words_closer = sum(np.linalg.norm(secret_vec - self.wordmap[word]) < np.linalg.norm(guess_vec - secret_vec) for word in self.wordmap.keys())
        return words_closer

    def get_recommendations(self, word_vec):
        dists = [(word, np.linalg.norm(word_vec - self.wordmap[word])) for word in self.wordmap.keys()]
        dists = np.array(dists, dtype=[('word', object), ('dist', np.float32)])
        return np.sort(dists, order='dist')

def load_word_vectors(file_path):
    wordmap = {}
    with open(file_path) as infile:
        for line in infile:
            split = line.strip().split(" ")
            word = split[0]
            vecs = np.array(list(map(float, split[1:])))
            wordmap[word] = vecs
    return wordmap

def choose_action(state, q_table, epsilon, word_list):
    if np.random.rand() < epsilon:
        return random.choice(word_list)
    else:
        return max(q_table[state], key=q_table[state].get, default=random.choice(word_list))

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, prev_words_closer):
    q_next_max = max(q_table[next_state].values(), default=0)
    reward = prev_words_closer - reward
    q_table[state][action] = q_table[state].get(action, 0) + alpha * (reward + gamma * q_next_max - q_table[state].get(action, 0))

def main():
    word_vectors = load_word_vectors('./vectors.txt')
    game = ContextoGame(word_vectors)

    q_table = defaultdict(dict)
    epsilon = 0.1  # Exploration rate
    alpha = 0.1    # Learning rate
    gamma = 0.9    # Discount factor
    episodes = 1000
    max_guesses = 100  # Limit the number of guesses per game

    word_list = list(word_vectors.keys())

    # Initialize an empty list to store episode data
    episode_data = []

    for episode in range(episodes):
        secret = game.reset()
        state = secret
        total_words_closer = 0
        num_guesses = 0
        total_avg_decrease = 0
        prev_words_closer = 0  # Initialize previous words closer

        for _ in range(max_guesses):
            action = choose_action(state, q_table, epsilon, word_list)
            words_closer = game.get_distance(action)
            total_words_closer += words_closer
            num_guesses += 1

            reward = words_closer  # Reward based on improvement in words closer
            next_state = action

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma, prev_words_closer)

            state = next_state
            prev_words_closer = words_closer  # Update previous words closer

            if words_closer == 0:
                break  # Guessed the correct word

        # Calculate average decrease in words closer per episode
        if episode > 0:
            avg_decrease = (total_words_closer - episode_data[-1]['Total_Words_Closer']) / num_guesses
        else:
            avg_decrease = 0

        # Store episode data in a dictionary
        episode_data.append({
            'Episode': episode + 1,
            'Secret_Word': secret,
            'Num_Guesses': num_guesses,
            'Total_Words_Closer': total_words_closer,
            'Average_Words_Closer': total_words_closer / num_guesses,
            'Correct_Guess': state == secret,
            'Avg_Decrease_Words_Closer_Per_Episode': avg_decrease
        })

        # Logging information about the current episode
        logging.info(f"Episode {episode + 1}/{episodes}: Secret Word={secret}, Total Words Closer={total_words_closer}, Average Words Closer per Episode={total_words_closer / num_guesses}, Avg Decrease Words Closer per Episode={avg_decrease}")

    # Convert episode_data to a DataFrame
    episode_data_df = pd.DataFrame(episode_data)

    # Save the DataFrame to a CSV file
    episode_data_df.to_csv('episode_data.csv', index=False)
    logging.info("Episode data saved to episode_data.csv")

    # Save the Q-table for future use
    with open('q_table.json', 'w') as f:
        json.dump(q_table, f)
        logging.info("Q-table saved to q_table.json")

if __name__ == "__main__":
    main()
