import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from torch.distributions import Categorical
from random import choices
import wandb

# Initialize the tracking of the cartpole learning
#wandb.init(project="cartpole_handmade")

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2) # output : probability of the 2 possibilities for the agent : left or right
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Environment and model parameters
env = gym.make("CartPole-v1")

model = Net()
optimizer = optim.Adam(model.parameters(), lr= 5e-3)
gamma = 0.99

max_eps_steps = env.spec.max_episode_steps
print("Nombre max d'étapes par épisode :", max_eps_steps)
nb_ep = 500
print("Nombre d'épisodes :", nb_ep)

# Progress lists
losses = []
total_rewards = []

for iter in range(nb_ep):
    # Reset the environment at the beginning of a new episode
    observation = env.reset()
    done = False

    # Reset the previous results
    buffer = torch.zeros(max_eps_steps)
    probs = torch.zeros(max_eps_steps)
    step = 0

    while not(done):

        # Get the new probabilities of action
        prob = model(torch.tensor(observation))

        # Sample the action based on the probabilities and store its probability in the buffer
        m = Categorical(prob)
        action = m.sample()
        probs[step] = prob[action]

        # Step the environment with the chosen action
        observation, reward, done, info = env.step(action.item())

        # Update buffer with the new reward
        for i in range(step):
            buffer[i] += reward * gamma ** (step - i)
        
        if step < len(buffer) - 1:
            buffer[step + 1] = reward

        step += 1

        # Render the Cart Pole environment
        #env.render()

    # Add the current reward to the list of rewards
    total_rewards.append(step)

    # Adjust results' size and normalize buffer
    buffer = buffer[:step]
    probs = probs[:step]
    F.normalize(buffer, dim=0)

    # Compute loss
    logs = torch.log10(probs)
    loss = - torch.sum(torch.mul(logs, buffer))
    losses.append(loss)

    # Print progress
    if (iter + 1) % 50 == 0:
        print("Épisode : {} / {}".format(iter + 1, nb_ep))
        print("Reward : ", step)
        print("Loss : ", round(loss.item(), 2))


    # Perform gradient descent to update neural network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log the progress in Weights and Biases
    #wandb.log({'Reward': step, 'Loss': loss.item()})


print("Épisode : {} / {}".format(iter + 1, nb_ep))
print("Loss : ", round(loss.item(), 2))
print("Reward : ", step)

# Plot the evolution of learning : reward and loss
# Reward
plt.plot(total_rewards)
plt.xlabel('Épisode')
plt.ylabel('Reward')
plt.title("Évolution du reward en fonction de l'épisode")
plt.show()

# Loss
total_losses = [loss.item() for loss in losses]
plt.plot(total_losses)
plt.xlabel('Épisode')
plt.ylabel('Loss')
plt.title("Évolution du loss en fonction de l'épisode")
plt.show()

