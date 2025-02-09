# 10-Armed Bandit Experiment
A classic **multi-armed bandit** problem to explore **reinforcement learning** strategies.

[GitHub Repository](GitHub_Link)

---

## 1. Bandit Testbed
The **10-armed bandit problem** is modeled as follows:
- Each arm has a **true action value** \( q^*(a) \), sampled from \( N(0,1) \).
- Pulling an arm gives a **reward** sampled from \( N(q^*(a),1) \).

### **Code Implementation**
```python
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, num_arms=10):
        # True action values q*(a) sampled from N(0,1)
        self.q_star = np.random.normal(0, 1, num_arms)

    def get_reward(self, action):
        # Reward sampled from N(q*(a),1) given an action
        return np.random.normal(self.q_star[action], 1)

    def optimal_action(self):
        return np.argmax(self.q_star)
```

---

## 2. Agent (Reinforcement Learning Strategy)
We use the **ε-greedy** action selection:
- **With probability \( \epsilon \)**: select a random action (**exploration**).
- **With probability \( 1 - \epsilon \)**: select the action with the highest estimated reward (**exploitation**).

### **Code Implementation**
```python
class Agent:
    def __init__(self, num_arms=10, epsilon=0.1):
        self.epsilon = epsilon
        self.q_estimates = np.zeros(num_arms)  # Initialize Q-values to 0
        self.action_counts = np.zeros(num_arms)  # Track action selection counts

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_estimates))  # Random action (exploration)
        else:
            return np.argmax(self.q_estimates)  # Greedy action (exploitation)

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]  # Incremental sample averaging
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
```

### **Incremental Q-Value Update Formula**
We update the **estimated action-value** \( Q(a) \) using incremental mean:
\[
Q_{n+1}(a) = Q_n(a) + \alpha \left( R_n - Q_n(a) \right)
\]
where:
- \( Q_n(a) \) is the current estimate.
- \( R_n \) is the reward received.
- \( \alpha = \frac{1}{n} \) is the step size.

---

## 3. Running a Single Test and Observing Q-Value Updates
Simulating **1000 steps** for a **single bandit** and **ε-greedy agent**.

### **Code Implementation**
```python
num_steps = 1000
bandit = Bandit()
agent = Agent(epsilon=0.1)

rewards = []
optimal_action_counts = []

for step in range(num_steps):
    action = agent.select_action()
    reward = bandit.get_reward(action)
    agent.update(action, reward)

    rewards.append(reward)
    optimal_action_counts.append(action == bandit.optimal_action())

print("Final Q estimates:", agent.q_estimates)
print("True q* values:", bandit.q_star)

plt.plot(rewards)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()
```
![Reward Trend](1.png)

---

## 4. Running 2000 Experiments and Calculating Average Reward
To evaluate **different exploration strategies**, we compare **ε = {0, 0.01, 0.1}** over **2000 runs**.

### **Code Implementation**
```python
num_experiments = 2000
epsilons = [0, 0.01, 0.1]
num_arms = 10

avg_rewards = {eps: np.zeros(num_steps) for eps in epsilons}
optimal_action_pct = {eps: np.zeros(num_steps) for eps in epsilons}

for experiment in range(num_experiments):
    bandit = Bandit()

    for eps in epsilons:
        agent = Agent(num_arms=num_arms, epsilon=eps)
        optimal_action = bandit.optimal_action()

        for step in range(num_steps):
            action = agent.select_action()
            reward = bandit.get_reward(action)
            agent.update(action, reward)

            avg_rewards[eps][step] += reward
            optimal_action_pct[eps][step] += (action == optimal_action)

# Compute final averages
for eps in epsilons:
    avg_rewards[eps] /= num_experiments
    optimal_action_pct[eps] = (optimal_action_pct[eps] / num_experiments) * 100

print("Finished 2000 experiments!")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for eps in epsilons:
    plt.plot(avg_rewards[eps], label=f'ε={eps}')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Reward vs Steps")

plt.subplot(1, 2, 2)
for eps in epsilons:
    plt.plot(optimal_action_pct[eps], label=f'ε={eps}')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Selection vs Steps")

plt.show()
```

### **Results**
- **Average Reward Over Time**:
  ![Average Reward](2.png)
- **Optimal Action Selection Percentage**:
  ![Optimal Action Percentage](3.png)

---

## 5. Summary and Insights
### **Bandit Testbed**
- Implements a **10-armed bandit** with **true action values** sampled from \( N(0,1) \).

### **Agent (Reinforcement Learning Strategy)**
- Implements **ε-greedy** action selection.
- Uses **incremental Q-value updates**.

### **Single Test Run**
- Demonstrates **Q-value updates** and **reward trends over time**.

### **Multiple Experiments (2000 runs)**
- **Exploration (\( \epsilon \)) affects performance**:
  - **\( \epsilon = 0 \)** (greedy only) learns slowly, gets stuck in **suboptimal actions**.
  - **\( \epsilon = 0.01 \)** balances exploration and exploitation.
  - **\( \epsilon = 0.1 \)** explores more, initially unstable but finds **optimal actions more often**.

---

## 6. Knowledge Review
### **1. Why Use Exploration (\( \epsilon \))?**
- Helps the agent **discover** the best actions.
- Avoids **premature convergence** to a suboptimal solution.

### **2. Why Use Incremental Q-Update?**
- **Computational efficiency**: No need to store all past rewards.
- **Online learning**: Can adapt to **non-stationary** problems.

### **3. What’s Next?**
- **Non-stationary bandits**: Rewards change over time.
- **Optimistic initialization**: Encourages exploration.
- **UCB (Upper Confidence Bound)**: More sophisticated exploration.

