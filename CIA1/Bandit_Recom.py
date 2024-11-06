import numpy as np

class NArmBanditRecommender:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.q_values)

    def update(self, chosen_arm, reward):
        self.action_counts[chosen_arm] += 1
        alpha = 1 / self.action_counts[chosen_arm]
        self.q_values[chosen_arm] += alpha * (reward - self.q_values[chosen_arm])

    def recommend(self):
        chosen_arm = self.select_arm()
        return chosen_arm

n_arms = 5
epsilon = 0.1

recommender = NArmBanditRecommender(n_arms, epsilon)

for _ in range(100):
    chosen_arm = recommender.recommend()
    reward = np.random.binomial(1, 0.5 + chosen_arm * 0.05)
    recommender.update(chosen_arm, reward)

print("Estimated rewards for each arm:", recommender.q_values)
print("Selection counts for each arm:", recommender.action_counts)
