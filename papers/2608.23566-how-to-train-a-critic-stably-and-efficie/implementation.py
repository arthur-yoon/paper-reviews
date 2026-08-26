import numpy as np
import math

class SimpleTokenizer:
    """
    A very simple, deterministic pseudo-tokenizer for simulation purposes.
    """
    def __init__(self, vocab_size=10000, max_len=512):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text):
        # Simple hash-based encoding for simulation
        tokens = [self.bos_token_id]
        for char in text:
            tokens.append(abs(hash(char)) % (self.vocab_size - 100) + 100)
        tokens.append(self.eos_token_id)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return tokens

    def pad(self, tokens, max_length=None):
        if max_length is None:
            max_length = self.max_len
        padding = [self.pad_token_id] * (max_length - len(tokens))
        return np.array(tokens + padding, dtype=np.int32)

class SimplePolicyModel:
    """
    Simulates a Policy Model using a random walk with a learnable bias.
    """
    def __init__(self, hidden_dim=64, seed=42):
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)
        # Simulated parameters
        self.weight_matrix = self.rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.bias_vector = np.zeros(hidden_dim)
        self.logits_head = self.rng.normal(0, 0.1, size=(hidden_dim, 100)) # Small vocab for sim

    def generate_response(self, prompt_tokens, max_length=10):
        """
        Simulates generating a response.
        In a real scenario, this would be autoregressive sampling.
        Here, we just generate a fixed length sequence of random tokens 
        influenced slightly by the 'quality' of the model (which improves over training).
        """
        # Simulate a response length that varies
        resp_len = min(max_length, int(self.rng.exponential(5) + 5))
        
        # Simulated "content" generation
        # As the model trains, it might produce 'better' tokens, 
        # but for this simulation, we just generate random tokens 
        # and rely on the reward function to evaluate the final string.
        
        tokens = []
        for _ in range(resp_len):
            token = self.rng.integers(10, 100)
            tokens.append(token)
        
        return tokens

    def forward(self, tokens):
        """
        Simulates the forward pass of the policy.
        Returns log-probabilities for the tokens.
        """
        # Simple simulation: log-probs are just random but consistent for the same token
        # In a real model, this depends on the previous context.
        # For simulation, we assume a flat distribution with slight noise.
        log_probs = self.rng.normal(0, 1, size=len(tokens))
        return log_probs

class SimpleCriticModel:
    """
    Simulates a Critic Model.
    """
    def __init__(self, hidden_dim=64, seed=43):
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)
        self.weight_matrix = self.rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.value_head = self.rng.normal(0, 0.1, size=(hidden_dim, 1))

    def predict_value(self, tokens, aux_info=None):
        """
        Predicts the value for each token.
        aux_info: e.g., ground truth answer, rubric info. 
        In simulation, aux_info can slightly bias the value prediction.
        """
        base_value = self.rng.normal(0, 0.5, size=len(tokens))
        
        # If aux_info is provided (e.g., correct answer), the critic might be better
        if aux_info is not None:
             # Simulate that aux info helps the critic estimate the final reward better
             # We add a small positive bias if aux_info suggests 'good'
             base_value += 0.1 * np.ones_like(base_value)
             
        # Sigmoid to bound between 0 and 1 (Value Bounding)
        values = 1 / (1 + np.exp(-base_value))
        return values

    def update(self, tokens, targets, learning_rate=0.01):
        """
        Updates the critic weights to match the Monte Carlo targets.
        """
        # Simulated gradient descent step
        # In reality, this would backpropagate through the neural network.
        # Here, we just log that an update happened.
        pass

class BPCOTrainer:
    def __init__(self, tokenizer, policy, critic, config):
        self.tokenizer = tokenizer
        self.policy = policy
        self.critic = critic
        self.config = config
        self.rng = np.random.default_rng(123)

    def calculate_reward(self, response_text, ground_truth):
        """
        Simulates a reward function (e.g., exact match or rubric).
        """
        # Simple exact match simulation
        if ground_truth in response_text:
            return 1.0
        else:
            return 0.0

    def compute_gae(self, rewards, values, gamma, lambda_gae):
        """
        Computes Generalized Advantage Estimation.
        Implements Length-Adaptive GAE by adjusting lambda based on length.
        """
        T = len(rewards)
        advantages = np.zeros(T)
        last_advantage = 0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            
            # Length-Adaptive GAE:
            # If the sequence is long, use a higher lambda (more look-ahead) 
            # or adjust gamma. 
            # Here we simulate adaptive lambda: longer sequences get lambda closer to 1
            current_lambda = lambda_gae * (1.0 + (T - t) / (T + 1))
            current_lambda = min(current_lambda, 1.0) # Cap at 1.0
            
            last_advantage = delta + gamma * current_lambda * last_advantage
            advantages[t] = last_advantage
            
        # Unnormalized Advantage: We do NOT standardize (subtract mean, divide by std)
        # This is a key part of the "Unnormalized Advantage" recipe.
        return advantages

    def train_step(self, prompt, ground_truth, aux_info):
        """
        Executes one step of BPCO training.
        """
        # 1. Sample a SINGLE response from the policy
        prompt_tokens = self.tokenizer.encode(prompt)
        response_tokens = self.policy.generate_response(prompt_tokens, max_length=self.config['max_response_len'])
        
        # 2. Compute Reward
        response_text = ''.join([chr(t + 96) for t in response_tokens]) # Decode for sim
        reward = self.calculate_reward(response_text, ground_truth)
        
        # 3. Critic Prediction
        # The critic predicts values for the response tokens.
        # We provide aux_info to the critic (e.g., the ground truth)
        predicted_values = self.critic.predict_value(response_tokens, aux_info=ground_truth)
        
        # Value Bounding is handled inside predict_value via sigmoid, 
        # but we can explicitly clip if needed
        predicted_values = np.clip(predicted_values, 0, 1)
        
        # 4. Monte Carlo Target
        # The target for the value function at the end is the actual reward.
        # For intermediate steps, we use GAE.
        # Let's assume the reward is only given at the end (sparse reward).
        # For simplicity, let's assume a dense reward signal for GAE calculation 
        # where the final step gets the reward and previous steps get 0 (or discounted).
        # Standard GAE usually uses per-step rewards. 
        # In LLM RL, often the reward is scalar at the end.
        # We can simulate per-step rewards by distributing the final reward 
        # or assuming a small reward for each token if it's "good".
        # Here, let's assume a sparse reward: 0 for all steps except the last one gets `reward`.
        
        step_rewards = np.zeros(len(response_tokens))
        if len(response_tokens) > 0:
            step_rewards[-1] = reward
            
        # 5. Compute Advantages (Length-Adaptive GAE, Unnormalized)
        advantages = self.compute_gae(
            step_rewards, 
            predicted_values, 
            gamma=self.config['gamma'], 
            lambda_gae=self.config['lambda_gae']
        )
        
        # 6. Critic Update (BPCO Core)
        # Target for the last token should be the reward.
        # For simulation, we just say the critic learns from the gap.
        mc_target = reward
        value_loss = np.mean((predicted_values[-1] - mc_target) ** 2)
        self.critic.update(response_tokens, mc_target, self.config['critic_lr'])
        
        # 7. Policy Update (DPPO style - simplified)
        # DPPO decouples the policy update. 
        # We update the policy based on the unnormalized advantages.
        # Simulated policy update: 
        # If advantage is positive, increase prob of these tokens.
        policy_loss = -np.mean(advantages) # Simplified loss
        # In reality: policy_loss = -min(ratio * A, clip(ratio) * A)
        
        # Return metrics for logging
        return {
            "reward": reward,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "resp_len": len(response_tokens)
        }

def main():
    # Configuration
    config = {
        "max_response_len": 20,
        "gamma": 0.99,
        "lambda_gae": 0.95,
        "critic_lr": 0.01,
        "epochs": 5,
        "batch_size": 10
    }

    # Initialize Components
    tokenizer = SimpleTokenizer()
    policy = SimplePolicyModel(hidden_dim=64)
    critic = SimpleCriticModel(hidden_dim=64)
    
    trainer = BPCOTrainer(tokenizer, policy, critic, config)

    # Simulated Dataset
    # (Prompt, Ground Truth Answer, Aux Info for Critic)
    dataset = [
        ("What is 2+2?", "4", "Answer is 4"),
        ("What is 3*3?", "9", "Answer is 9"),
        ("What is 10-2?", "8", "Answer is 8"),
        ("What is 5+5?", "10", "Answer is 10"),
        ("What is 1*1?", "1", "Answer is 1"),
    ]

    print("Starting BPCO Training Simulation...")
    print("-" * 30)
    
    total_reward_history = []
    total_value_loss_history = []

    for epoch in range(config["epochs"]):
        epoch_rewards = []
        epoch_value_losses = []
        
        for i in range(config["batch_size"]):
            # Select a sample
            prompt, gt, aux = dataset[i % len(dataset)]
            
            # Train Step
            metrics = trainer.train_step(prompt, gt, aux)
            
            epoch_rewards.append(metrics["reward"])
            epoch_value_losses.append(metrics["value_loss"])
            
        avg_reward = np.mean(epoch_rewards)
        avg_vloss = np.mean(epoch_value_losses)
        
        total_reward_history.append(avg_reward)
        total_value_loss_history.append(avg_vloss)
        
        print(f"Epoch {epoch+1:2d} | Avg Reward: {avg_reward:.3f} | Critic Value Loss: {avg_vloss:.4f}")

    print("-" * 30)
    print("Training Complete.")
    print(f"Final Avg Reward: {total_reward_history[-1]:.3f}")
    print(f"Final Critic Value Loss: {total_value_loss_history[-1]:.4f}")
    print("Note: In a real scenario, the reward would likely increase as the policy improves.")

if __name__ == "__main__":
    main()
