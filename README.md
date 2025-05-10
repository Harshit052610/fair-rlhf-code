# Fair-RLHF: Fairness-Aware Offline Reinforcement Learning with Human Feedback

This repository contains the code for the paper *"Fairness-Aware Offline Reinforcement Learning with Human Feedback (Fair-RLHF): Mitigating Bias in Human Preferences"*. The project focuses on developing a fairness-aware reinforcement learning model for shopping recommendations, incorporating human feedback to mitigate bias in preferences, specifically across `Gender`.

## Project Overview
The goal of this project is to create a fairness-aware recommendation system using offline reinforcement learning (RL) with human feedback (RLHF). The system recommends product categories (e.g., `Clothing`, `Footwear`) to customers while ensuring fairness across sensitive attributes like `Gender`. The pipeline includes data preprocessing, model training, fairness evaluation, and fine-tuning with human feedback.

### Key Features
- **Offline RL**: Trains a PPO model using a static dataset (`shopping_behavior_updated.csv`).
- **Fairness**: Incorporates a fairness penalty in the reward function to balance recommendations across `Gender`.
- **Human Feedback**: Fine-tunes the model with simulated human preferences (preferring `Clothing` and `Footwear`) to align with the RLHF framework.
- **Fairness Evaluation**: Uses `fairlearn` to evaluate fairness metrics like Demographic Parity Difference (DPD).

## Repository Structure
- `/src`:
  - `preprocess_data.py`: Preprocesses the raw dataset (`shopping_behavior_updated.csv`) and saves it as `preprocessed_shopping_data.csv`.
  - `train_model.py`: Trains a PPO model using the preprocessed data and saves it as `ppo_shopping_model.zip`.
  - `evaluate_fairness.py`: Evaluates the fairness of the model's recommendations using `fairlearn` and visualizes results.
  - `incorporate_human_feedback.py`: Fine-tunes the model with simulated human feedback and saves it as `ppo_shopping_model_with_feedback.zip`.
- `README.md`: This file, providing an overview and instructions.

## Prerequisites
- Python 3.8 or higher
- Google Colab (recommended for running the scripts)
- Install dependencies manually (no `requirements.txt` provided):
  ```bash
  pip install pandas numpy gymnasium stable-baselines3 scikit-learn fairlearn matplotlib seaborn google-colab
