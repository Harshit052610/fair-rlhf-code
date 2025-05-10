import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from google.colab import files
import io
import gymnasium as gym
from gymnasium import spaces

class ShoppingEnv(gym.Env):
    def __init__(self, data, feedback=None):
        super(ShoppingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.feedback = feedback if feedback is not None else [0] * len(data)
        self.current_step = 0
        self.max_steps = len(data)
        
        self.action_space = spaces.Discrete(5)
        self.feature_cols = [col for col in data.columns if col != 'Customer ID' and 'Purchase Amount (USD)' not in col]
        self.observation_space = spaces.Box(low=-5, high=5, shape=(len(self.feature_cols),), dtype=np.float32)
        
        self.category_mapping = {i: cat for i, cat in enumerate(['Clothing', 'Footwear', 'Outerwear', 'Accessories', 'Jewelry'])}
        self.gender_col = 'Gender_Male'
        self.action_counts = {0: {'Male': 0, 'Female': 0}, 1: {'Male': 0, 'Female': 0},
                             2: {'Male': 0, 'Female': 0}, 3: {'Male': 0, 'Female': 0},
                             4: {'Male': 0, 'Female': 0}}

    def reset(self, seed=None):
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_step >= self.max_steps:
            self.current_step = 0
        return self.data.iloc[self.current_step][self.feature_cols].values.astype(np.float32)

    def _calculate_fairness_penalty(self, gender, action):
        gender_label = 'Male' if gender == 1 else 'Female'
        self.action_counts[action][gender_label] += 1
        male_count = self.action_counts[action]['Male']
        female_count = self.action_counts[action]['Female']
        total = male_count + female_count
        if total == 0:
            return 0
        imbalance = abs(male_count / total - female_count / total)
        penalty = -10 * imbalance
        return penalty

    def step(self, action):
        customer = self.data.iloc[self.current_step]
        actual_category = customer.filter(like='Category_').idxmax().replace('Category_', '')
        purchase_amount = customer['Purchase Amount (USD)']
        gender = customer[self.gender_col]

        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        
        if self.category_mapping[action] == actual_category:
            base_reward = purchase_amount
        else:
            base_reward = 0

        fairness_penalty = self._calculate_fairness_penalty(gender, action)
        feedback_reward = self.feedback[self.current_step] * 10
        reward = base_reward + fairness_penalty + feedback_reward

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        if done:
            self.current_step = 0
        
        return self._get_observation(), reward, done, truncated, {}

def load_preprocessed_data():
    print("Please upload the 'preprocessed_shopping_data.csv' file if not already uploaded.")
    uploaded = files.upload()
    
    target_file = None
    for filename in uploaded.keys():
        if 'preprocessed_shopping_data' in filename.lower():
            target_file = filename
            break
    
    if target_file is None:
        print("Error: No file containing 'preprocessed_shopping_data' was found in uploaded files.")
        return None
    
    try:
        df = pd.read_csv(io.BytesIO(uploaded[target_file]))
        print(f"Preprocessed data loaded from '{target_file}' with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model(model_path='ppo_shopping_model'):
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def simulate_human_feedback(data):
    category_mapping = {i: cat for i, cat in enumerate(['Clothing', 'Footwear', 'Outerwear', 'Accessories', 'Jewelry'])}
    feedback = []
    
    for idx in range(len(data)):
        actual_category = data.iloc[idx].filter(like='Category_').idxmax().replace('Category_', '')
        if actual_category in ['Clothing', 'Footwear']:
            feedback.append(1.0)
        else:
            feedback.append(0.5)
    return feedback

def fine_tune_with_feedback(model, data, feedback):
    env = ShoppingEnv(data, feedback=feedback)
    model.set_env(env)
    
    total_timesteps_per_episode = 2000
    num_episodes = 5
    
    for episode in range(num_episodes):
        model.learn(total_timesteps=total_timesteps_per_episode, reset_num_timesteps=False)
        print(f"Completed fine-tuning episode {episode + 1}/{num_episodes}")
    
    model.save("ppo_shopping_model_with_feedback")
    print("Model fine-tuned with human feedback and saved as 'ppo_shopping_model_with_feedback'.")
    files.download('ppo_shopping_model_with_feedback.zip')
    print("Fine-tuned model downloaded as 'ppo_shopping_model_with_feedback.zip'.")
    return model

def download_script():
    script_content = __file_content__
    with open('incorporate_human_feedback.py', 'w') as f:
        f.write(script_content)
    files.download('incorporate_human_feedback.py')
    print("Script downloaded as 'incorporate_human_feedback.py'.")

def main():
    df = load_preprocessed_data()
    if df is None:
        return
    
    model = load_model()
    if model is None:
        return
    
    feedback = simulate_human_feedback(df)
    fine_tune_with_feedback(model, df, feedback)
    download_script()

if __name__ == "__main__":
    main()
