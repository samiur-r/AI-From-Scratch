# Reinforcement Learning Algorithms Quick Reference

Reinforcement Learning (RL) is a machine learning paradigm where agents learn to make decisions by interacting with an environment to maximize cumulative reward. Unlike supervised learning, RL learns from trial and error through rewards and penalties, making it ideal for sequential decision-making problems.

## What Reinforcement Learning Does

Reinforcement Learning solves sequential decision-making problems where an agent must learn the optimal policy (strategy) to maximize long-term reward. The key components are:

1. **Agent**: The learner or decision maker
2. **Environment**: The world the agent interacts with
3. **State (S)**: Current situation of the agent
4. **Action (A)**: Choices available to the agent
5. **Reward (R)**: Feedback signal from the environment
6. **Policy (π)**: Strategy that maps states to actions
7. **Value Function (V)**: Expected cumulative reward from a state
8. **Q-Function (Q)**: Expected cumulative reward for a state-action pair

Mathematical foundation: The agent learns to maximize the expected cumulative discounted reward: E[∑(γ^t * R_t)] where γ is the discount factor.

## When to Use Reinforcement Learning

### Problem Types
- **Sequential decision making**: Multi-step problems where actions affect future states
- **Game playing**: Chess, Go, video games, board games
- **Control problems**: Robotics, autonomous vehicles, trading systems
- **Optimization**: Resource allocation, scheduling, routing
- **Recommendation systems**: Personalized content with user feedback
- **Dynamic pricing**: Adjusting prices based on market response

### Data Characteristics
- **Interactive environment**: Ability to take actions and receive feedback
- **Sequential nature**: Current decisions affect future options
- **Delayed rewards**: Feedback may come after multiple actions
- **Exploration vs exploitation**: Need to balance trying new actions vs using known good ones
- **Partial observability**: May not see the complete state of the environment

### Business Contexts
- Autonomous systems (self-driving cars, drones, robots)
- Financial trading and portfolio management
- Supply chain and inventory optimization
- Personalized recommendations and advertising
- Game AI and entertainment
- Energy management and smart grids
- Healthcare treatment optimization

### Comparison with Alternatives
- **Use RL when**: Sequential decisions, interactive environment, long-term optimization
- **Use supervised learning when**: Clear input-output mapping, labeled data available
- **Use unsupervised learning when**: Pattern discovery, no clear objective function
- **Use evolutionary algorithms when**: Black-box optimization, no gradient information

## Strengths & Weaknesses

### Strengths
- **Learning from interaction**: No need for labeled training data
- **Optimal decision making**: Can find globally optimal policies
- **Handles sequential dependencies**: Naturally models temporal relationships
- **Adaptability**: Can adjust to changing environments
- **Generalization**: Learned policies can work in similar environments
- **Multi-objective optimization**: Can balance multiple competing objectives
- **Real-time learning**: Can improve performance while deployed

### Weaknesses
- **Sample efficiency**: Often requires many interactions to learn
- **Exploration challenges**: May get stuck in local optima
- **Reward engineering**: Designing good reward functions is difficult
- **Computational complexity**: Can be computationally expensive
- **Stability issues**: Learning can be unstable and sensitive to hyperparameters
- **Safety concerns**: Exploration can lead to dangerous actions
- **Evaluation difficulty**: Hard to evaluate without deployment

## Important Hyperparameters

### Learning Parameters
- **learning_rate (α)**: Step size for updates (0.001-0.1 typical)
- **discount_factor (γ)**: Future reward importance (0.9-0.99 common)
- **exploration_rate (ε)**: Probability of random action in ε-greedy (0.1-0.3)
- **exploration_decay**: Rate of reducing exploration over time
- **batch_size**: Number of experiences per update (32-512)

### Network Architecture (Deep RL)
- **hidden_layers**: Number and size of neural network layers
- **activation_function**: ReLU, tanh, or other activation functions
- **optimizer**: Adam, RMSprop, SGD for neural network training
- **target_update_frequency**: How often to update target networks

### Algorithm Specific
- **replay_buffer_size**: Memory size for experience replay (10K-1M)
- **update_frequency**: How often to perform learning updates
- **polyak_tau**: Soft update parameter for target networks (0.001-0.01)
- **entropy_coefficient**: Exploration bonus in policy gradient methods

### Environment Settings
- **max_episodes**: Maximum training episodes
- **max_steps_per_episode**: Episode length limit
- **reward_scaling**: Scaling factor for rewards
- **state_preprocessing**: Normalization or feature extraction

## Key Assumptions

### Environment Assumptions
- **Markov property**: Current state contains all relevant information
- **Stationary environment**: Environment dynamics don't change during learning
- **Bounded rewards**: Rewards are finite and bounded
- **Episodic or continuing**: Tasks have clear episodes or run indefinitely

### Learning Assumptions
- **Sufficient exploration**: Agent can explore all relevant states
- **Convergence conditions**: Learning rates and exploration schedules allow convergence
- **Function approximation**: Linear or neural network approximation is sufficient
- **Sample generation**: Can generate sufficient training samples

### Mathematical Assumptions
- **Bellman optimality**: Optimal value functions satisfy Bellman equations
- **Policy improvement**: Greedy policies with respect to value functions improve
- **Convergence guarantees**: Under certain conditions, algorithms converge to optimal policies

### Violations and Consequences
- **Non-Markov environments**: May need to include history in state representation
- **Non-stationary environments**: May require adaptive learning rates or forgetting
- **Continuous state/action spaces**: Require function approximation
- **Partial observability**: May need to learn state representations

## Performance Characteristics

### Time Complexity
- **Tabular methods**: O(|S| × |A|) memory, O(1) per update
- **Function approximation**: O(d) where d is parameter dimension
- **Deep RL**: O(n × m) where n is network size, m is batch size
- **Planning methods**: O(|S|² × |A|) for value iteration

### Space Complexity
- **Q-tables**: O(|S| × |A|) for discrete state-action spaces
- **Neural networks**: O(parameters) typically millions for deep networks
- **Experience replay**: O(buffer_size × state_dimension)
- **Model-based**: Additional O(|S| × |A| × |S|) for transition model

### Sample Complexity
- **Tabular Q-learning**: Polynomial in state/action space size
- **Function approximation**: Depends on function class complexity
- **Deep RL**: Often requires millions of environment interactions
- **Model-based**: Generally more sample efficient than model-free

### Convergence Properties
- **Tabular methods**: Guaranteed convergence under certain conditions
- **Function approximation**: May not converge, depends on approximation quality
- **Policy gradient**: Converges to local optima
- **Actor-critic**: Combines benefits but may be less stable

## Evaluation & Comparison

### Performance Metrics
- **Cumulative reward**: Total reward obtained per episode
- **Average reward**: Mean reward over multiple episodes
- **Success rate**: Percentage of episodes achieving the goal
- **Learning curve**: Reward progression over training time
- **Sample efficiency**: Performance vs number of environment interactions

### Evaluation Strategies
- **Training performance**: Monitor reward during training
- **Test episodes**: Evaluate on unseen scenarios without exploration
- **Cross-validation**: Test on different environment configurations
- **Ablation studies**: Remove components to understand their contribution
- **Baseline comparison**: Compare against random, heuristic, or optimal policies

### Stability Metrics
- **Variance**: Consistency of performance across runs
- **Convergence time**: Episodes needed to reach stable performance
- **Final performance**: Ultimate capability after full training
- **Robustness**: Performance under environment variations

### Safety Evaluation
- **Constraint violations**: Number of unsafe actions taken
- **Worst-case performance**: Performance in adversarial scenarios
- **Recovery capability**: Ability to recover from mistakes

## Practical Usage Guidelines

### Implementation Tips
- **Start simple**: Begin with tabular methods for small problems
- **Careful reward design**: Ensure rewards align with desired behavior
- **Hyperparameter tuning**: Use grid search or random search for key parameters
- **Monitoring**: Track learning curves and debug training issues
- **Seed control**: Use fixed seeds for reproducible experiments

### Common Mistakes
- **Reward hacking**: Agent finds unintended ways to maximize reward
- **Insufficient exploration**: Getting stuck in local optima
- **Unstable learning**: Poor hyperparameter choices leading to divergence
- **Overfitting**: Memorizing specific environment instances
- **Ignoring safety**: Allowing dangerous exploration during training

### Debugging Strategies
- **Sanity checks**: Test on simple known environments first
- **Reward analysis**: Monitor reward distribution and trajectory
- **Action analysis**: Check if agent explores action space adequately
- **Value function visualization**: Plot learned value functions
- **Ablation testing**: Remove components to isolate issues

### Production Considerations
- **Safety constraints**: Implement hard constraints for critical applications
- **Online learning**: Capability to continue learning in deployment
- **Computational efficiency**: Optimize for real-time decision making
- **Robustness testing**: Evaluate under various conditions
- **Monitoring and logging**: Track performance in production

## Complete Example: Q-Learning Implementation

Here's a comprehensive implementation of Q-Learning for a grid world environment:

### Step 1: Environment Setup
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# What's happening: Creating a grid world environment for the agent to learn in
# Why this setup: Grid worlds are simple but capture key RL concepts like
# states, actions, rewards, and sequential decision making

class GridWorld:
    """Simple grid world environment for reinforcement learning"""

    def __init__(self, width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or [(2, 2), (3, 2)]

        # Actions: up, down, left, right
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        # Current state
        self.current_state = start

    def reset(self):
        """Reset environment to starting state"""
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        """Take an action and return (next_state, reward, done, info)"""
        if action < 0 or action >= len(self.actions):
            raise ValueError(f"Invalid action: {action}")

        # Calculate next state
        dx, dy = self.actions[action]
        next_x = self.current_state[0] + dx
        next_y = self.current_state[1] + dy

        # Check boundaries
        next_x = max(0, min(self.width - 1, next_x))
        next_y = max(0, min(self.height - 1, next_y))

        next_state = (next_x, next_y)

        # Check obstacles
        if next_state in self.obstacles:
            next_state = self.current_state  # Stay in current state

        # Calculate reward
        if next_state == self.goal:
            reward = 100  # Large positive reward for reaching goal
        elif next_state in self.obstacles:
            reward = -10  # Penalty for hitting obstacle
        else:
            reward = -1   # Small penalty for each step (encourage efficiency)

        # Check if episode is done
        done = (next_state == self.goal)

        self.current_state = next_state

        return next_state, reward, done, {}

    def get_valid_actions(self, state):
        """Get list of valid actions from current state"""
        valid_actions = []
        for i, (dx, dy) in enumerate(self.actions):
            next_x = state[0] + dx
            next_y = state[1] + dy

            # Check if action leads to valid position
            if (0 <= next_x < self.width and
                0 <= next_y < self.height and
                (next_x, next_y) not in self.obstacles):
                valid_actions.append(i)

        return valid_actions

    def render(self, q_table=None, policy=None):
        """Visualize the grid world"""
        grid = np.zeros((self.height, self.width))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1

        # Mark goal
        grid[self.goal[1], self.goal[0]] = 2

        # Mark current position
        grid[self.current_state[1], self.current_state[0]] = 1

        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='viridis')

        # Add grid lines
        for i in range(self.height + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
        for i in range(self.width + 1):
            plt.axvline(i - 0.5, color='black', linewidth=1)

        # Add labels
        for i in range(self.height):
            for j in range(self.width):
                if (j, i) == self.start:
                    plt.text(j, i, 'S', ha='center', va='center', fontsize=16, color='white')
                elif (j, i) == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center', fontsize=16, color='white')
                elif (j, i) in self.obstacles:
                    plt.text(j, i, 'X', ha='center', va='center', fontsize=16, color='white')

        plt.title('Grid World Environment')
        plt.show()

# Create environment
env = GridWorld(width=5, height=5, start=(0, 0), goal=(4, 4))
print("Environment created successfully")
env.render()
```

### Step 2: Q-Learning Algorithm Implementation
```python
# What's happening: Implementing the Q-Learning algorithm
# What the algorithm is learning: The Q-function Q(s,a) which estimates
# the expected future reward for taking action 'a' in state 's'

class QLearningAgent:
    """Q-Learning reinforcement learning agent"""

    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Q-table: maps (state, action) to Q-value
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def choose_action(self, state, valid_actions=None):
        """Choose action using epsilon-greedy policy"""
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)

        # Exploitation: choose best action
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}

        # Choose action with highest Q-value among valid actions
        best_action = max(valid_q_values, key=valid_q_values.get)
        return best_action

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule"""

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]

        if done:
            # No future reward if episode is done
            target_q = reward
        else:
            # Maximum Q-value for next state
            next_max_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * next_max_q

        # Update Q-value
        self.q_table[state][action] = (
            current_q + self.learning_rate * (target_q - current_q)
        )

    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def get_policy(self):
        """Extract policy from Q-table"""
        policy = {}
        for state in self.q_table:
            policy[state] = np.argmax(self.q_table[state])
        return policy

    def get_value_function(self):
        """Extract value function from Q-table"""
        value_function = {}
        for state in self.q_table:
            value_function[state] = np.max(self.q_table[state])
        return value_function

# Initialize agent
agent = QLearningAgent(
    n_actions=4,  # 4 possible actions in grid world
    learning_rate=0.1,
    discount_factor=0.95,
    exploration_rate=1.0,
    exploration_decay=0.995
)

print("Q-Learning agent initialized")
print(f"Initial exploration rate: {agent.exploration_rate}")
```

### Step 3: Training Process
```python
# What's happening: Training the Q-Learning agent through episodes
# What the algorithm is learning: Optimal action values Q*(s,a) through
# trial and error, balancing exploration and exploitation

def train_agent(agent, env, num_episodes=1000, max_steps_per_episode=200):
    """Train the Q-learning agent"""

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):
            # Choose action
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, done)

            # Update state and tracking
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay exploration rate
        agent.decay_exploration()

        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode:4d}: Avg Reward = {avg_reward:6.2f}, "
                  f"Avg Length = {avg_length:6.2f}, Exploration = {agent.exploration_rate:.3f}")

    return episode_rewards, episode_lengths

# Train the agent
print("Starting training...")
rewards, lengths = train_agent(agent, env, num_episodes=1000)

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot rewards
window = 50
rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax1.plot(rewards_smooth)
ax1.set_title('Learning Curve: Rewards per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Reward')
ax1.grid(True)

# Plot episode lengths
lengths_smooth = np.convolve(lengths, np.ones(window)/window, mode='valid')
ax2.plot(lengths_smooth)
ax2.set_title('Learning Curve: Steps per Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Steps')
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Training completed!")
print(f"Final exploration rate: {agent.exploration_rate:.3f}")
print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")
```

### Step 4: Evaluation and Policy Analysis
```python
# What's happening: Evaluating the learned policy and analyzing the results
# How to interpret results: Good policies should reach the goal efficiently
# with high rewards and short episode lengths

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate the trained agent"""

    # Set exploration rate to 0 for evaluation (pure exploitation)
    original_exploration = agent.exploration_rate
    agent.exploration_rate = 0.0

    eval_rewards = []
    eval_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(200):  # Max steps per episode
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                success_count += 1
                break

        eval_rewards.append(total_reward)
        eval_lengths.append(steps)

    # Restore original exploration rate
    agent.exploration_rate = original_exploration

    return {
        'avg_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'avg_length': np.mean(eval_lengths),
        'std_length': np.std(eval_lengths),
        'success_rate': success_count / num_episodes,
        'rewards': eval_rewards,
        'lengths': eval_lengths
    }

# Evaluate the agent
eval_results = evaluate_agent(agent, env, num_episodes=100)

print("Evaluation Results:")
print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
print(f"Success Rate: {eval_results['success_rate']:.2%}")

# Visualize learned policy
def visualize_policy(agent, env):
    """Visualize the learned policy"""
    policy = agent.get_policy()
    value_function = agent.get_value_function()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot value function
    value_grid = np.zeros((env.height, env.width))
    for state, value in value_function.items():
        value_grid[state[1], state[0]] = value

    im1 = ax1.imshow(value_grid, cmap='viridis')
    ax1.set_title('Value Function')
    plt.colorbar(im1, ax=ax1)

    # Add value numbers
    for i in range(env.height):
        for j in range(env.width):
            if (j, i) not in env.obstacles:
                value = value_function.get((j, i), 0)
                ax1.text(j, i, f'{value:.1f}', ha='center', va='center',
                        color='white' if value < np.mean(list(value_function.values())) else 'black')

    # Plot policy arrows
    policy_grid = np.zeros((env.height, env.width))
    for i in range(env.height):
        for j in range(env.width):
            if (j, i) == env.goal:
                ax2.text(j, i, 'G', ha='center', va='center', fontsize=16, color='red')
            elif (j, i) == env.start:
                ax2.text(j, i, 'S', ha='center', va='center', fontsize=16, color='blue')
            elif (j, i) in env.obstacles:
                ax2.text(j, i, 'X', ha='center', va='center', fontsize=16, color='black')
                policy_grid[i, j] = -1
            else:
                action = policy.get((j, i), 0)
                arrow = ['↑', '↓', '←', '→'][action]
                ax2.text(j, i, arrow, ha='center', va='center', fontsize=12)

    ax2.imshow(policy_grid, cmap='gray', alpha=0.3)
    ax2.set_title('Learned Policy')
    ax2.set_xticks(range(env.width))
    ax2.set_yticks(range(env.height))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Visualize the learned policy
visualize_policy(agent, env)

# Test a single episode with visualization
def test_episode_with_visualization(agent, env):
    """Run a single episode and show the path"""
    # Set to pure exploitation
    original_exploration = agent.exploration_rate
    agent.exploration_rate = 0.0

    state = env.reset()
    path = [state]
    actions_taken = []

    for step in range(50):  # Max steps
        valid_actions = env.get_valid_actions(state)
        action = agent.choose_action(state, valid_actions)
        actions_taken.append(env.action_names[action])

        next_state, reward, done, _ = env.step(action)
        path.append(next_state)

        if done:
            break

        state = next_state

    # Restore exploration rate
    agent.exploration_rate = original_exploration

    print(f"Episode completed in {len(path)-1} steps")
    print(f"Path taken: {' -> '.join([str(p) for p in path])}")
    print(f"Actions: {' -> '.join(actions_taken)}")

    return path, actions_taken

# Test a single episode
print("\nTesting single episode:")
path, actions = test_episode_with_visualization(agent, env)
```

### Step 5: Advanced Analysis and Extensions
```python
# What's happening: Advanced analysis of the learning process and extensions
# How to use in practice: These techniques help understand and improve RL performance

def analyze_q_table(agent, env):
    """Analyze the learned Q-table"""

    print("Q-Table Analysis:")
    print("=" * 50)

    # Find states with highest values
    value_function = agent.get_value_function()
    sorted_states = sorted(value_function.items(), key=lambda x: x[1], reverse=True)

    print("Top 5 most valuable states:")
    for i, (state, value) in enumerate(sorted_states[:5]):
        print(f"{i+1}. State {state}: Value = {value:.2f}")

    # Analyze Q-values for specific states
    print(f"\nQ-values for start state {env.start}:")
    q_values = agent.q_table[env.start]
    for action, q_val in enumerate(q_values):
        action_name = env.action_names[action]
        print(f"  {action_name}: {q_val:.2f}")

    # Check convergence by looking at Q-value ranges
    all_q_values = []
    for state in agent.q_table:
        all_q_values.extend(agent.q_table[state])

    print(f"\nQ-value statistics:")
    print(f"  Range: [{np.min(all_q_values):.2f}, {np.max(all_q_values):.2f}]")
    print(f"  Mean: {np.mean(all_q_values):.2f}")
    print(f"  Std: {np.std(all_q_values):.2f}")

# Analyze the learned Q-table
analyze_q_table(agent, env)

# Compare different hyperparameters
def hyperparameter_comparison():
    """Compare different hyperparameter settings"""

    hyperparams = [
        {'learning_rate': 0.05, 'discount_factor': 0.9, 'name': 'Low LR, Low Gamma'},
        {'learning_rate': 0.1, 'discount_factor': 0.95, 'name': 'Medium LR, Medium Gamma'},
        {'learning_rate': 0.2, 'discount_factor': 0.99, 'name': 'High LR, High Gamma'},
    ]

    results = {}

    for params in hyperparams:
        print(f"\nTesting: {params['name']}")

        # Create new agent
        test_agent = QLearningAgent(
            n_actions=4,
            learning_rate=params['learning_rate'],
            discount_factor=params['discount_factor'],
            exploration_rate=1.0,
            exploration_decay=0.995
        )

        # Train briefly
        test_env = GridWorld(width=5, height=5, start=(0, 0), goal=(4, 4))
        train_rewards, _ = train_agent(test_agent, test_env, num_episodes=500)

        # Evaluate
        eval_results = evaluate_agent(test_agent, test_env, num_episodes=50)

        results[params['name']] = {
            'final_reward': np.mean(train_rewards[-50:]),
            'eval_reward': eval_results['avg_reward'],
            'success_rate': eval_results['success_rate']
        }

    print("\nHyperparameter Comparison Results:")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final Training Reward: {result['final_reward']:.2f}")
        print(f"  Evaluation Reward: {result['eval_reward']:.2f}")
        print(f"  Success Rate: {result['success_rate']:.2%}")

# Run hyperparameter comparison (commented out for demo)
# hyperparameter_comparison()

# Function to save and load the agent
def save_agent(agent, filename):
    """Save trained agent to file"""
    import pickle

    agent_data = {
        'q_table': dict(agent.q_table),
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'discount_factor': agent.discount_factor,
            'exploration_rate': agent.exploration_rate,
            'exploration_decay': agent.exploration_decay,
            'min_exploration': agent.min_exploration
        }
    }

    with open(filename, 'wb') as f:
        pickle.dump(agent_data, f)

    print(f"Agent saved to {filename}")

def load_agent(filename, n_actions):
    """Load trained agent from file"""
    import pickle

    with open(filename, 'rb') as f:
        agent_data = pickle.load(f)

    # Create new agent with saved hyperparameters
    agent = QLearningAgent(
        n_actions=n_actions,
        **agent_data['hyperparameters']
    )

    # Load Q-table
    agent.q_table = defaultdict(lambda: np.zeros(n_actions), agent_data['q_table'])

    print(f"Agent loaded from {filename}")
    return agent

# Save the trained agent
save_agent(agent, 'q_learning_agent.pkl')

print("\nQ-Learning implementation complete!")
print("Key concepts demonstrated:")
print("1. Environment interaction and reward structure")
print("2. Q-value updates using Bellman equation")
print("3. Exploration vs exploitation trade-off")
print("4. Policy extraction from Q-values")
print("5. Performance evaluation and analysis")
```

## Key Reinforcement Learning Algorithms

### Value-Based Methods
- **Q-Learning**: Off-policy method learning Q(s,a) values
- **SARSA**: On-policy method updating current policy
- **Double Q-Learning**: Reduces overestimation bias
- **Deep Q-Networks (DQN)**: Neural network Q-function approximation

### Policy-Based Methods
- **REINFORCE**: Basic policy gradient method
- **Actor-Critic**: Combines value and policy learning
- **Proximal Policy Optimization (PPO)**: Stable policy updates
- **Trust Region Policy Optimization (TRPO)**: Constrained policy improvements

### Model-Based Methods
- **Value Iteration**: Dynamic programming for known environments
- **Policy Iteration**: Alternates between policy evaluation and improvement
- **Monte Carlo Tree Search (MCTS)**: Planning with simulation
- **Dyna-Q**: Combines learning and planning

### Advanced Methods
- **Deep Deterministic Policy Gradient (DDPG)**: Continuous action spaces
- **Soft Actor-Critic (SAC)**: Maximum entropy reinforcement learning
- **Rainbow DQN**: Combines multiple DQN improvements
- **Multi-Agent RL**: Learning in environments with multiple agents

## Summary

**Key Takeaways:**
- **Trial and error learning** through interaction with environment
- **Sequential decision making** with delayed rewards
- **Exploration vs exploitation** fundamental trade-off
- **Value functions and policies** core concepts for optimization
- **Sample efficiency** major challenge requiring careful algorithm design
- **Wide applicability** from games to robotics to business optimization

**Quick Decision Guide:**
- Use **Q-Learning** for discrete, small state spaces
- Use **Policy Gradients** for continuous or large action spaces
- Use **Actor-Critic** for balance of stability and efficiency
- Use **Model-Based** methods when environment model is available
- Consider **Deep RL** for complex, high-dimensional problems
- Start with **tabular methods** for learning and prototyping

**Success Factors:**
- Careful reward function design aligned with objectives
- Appropriate exploration strategy for the problem
- Hyperparameter tuning especially learning rates and exploration
- Sufficient training time and sample collection
- Proper evaluation methodology with multiple episodes