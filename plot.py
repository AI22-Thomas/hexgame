
#import plt
import matplotlib.pyplot as plt
#import numpy as np
import numpy as np
arrayOfAverages = []

#load averages form averages.txt (I saved arrays in this txt file with /n as a seperator)
with open("models/averages.txt", "r") as f:
    arr = [eval(line) for line in f.readlines()]
arrayOfAverages = arr


averages = [sum(array) / len(array) if len(array) > 0 else 0 for array in arrayOfAverages]

##plot averages as a line graph
#plt.plot(averages)
#plt.ylabel('Average')
#plt.xlabel('Episode')
#plt.show()


# Calculate averages
avg_rewards = [sum(array1)/len(array1) for array1 in arrayOfAverages]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, marker='o')
plt.title('Average reward over time')
plt.xlabel('Model iteration')
plt.ylabel('Average reward')
plt.grid()
plt.show()


data = arrayOfAverages




# For each model, plot their average reward over time
for model_index in range(len(data[-1])):
    model_scores = [gen[model_index] for gen in data if model_index < len(gen)]
    generations = range(model_index + 1, len(model_scores) + model_index + 1)
    plt.plot(generations, model_scores, marker='o', linestyle='-', label=f'Model {model_index+1}')

plt.title('Model average reward over time')
plt.xlabel('Generation')
plt.ylabel('Average reward')
plt.legend()
plt.grid()
plt.show()

#fig, axs = plt.subplots(len(data), 1, figsize=(10, 6), sharex=True, tight_layout=True)

#for i in range(len(data)):
#    rewards = data[i]
#    axs[i].plot(range(1, len(rewards) + 1), rewards, marker='o', color='b')
#    axs[i].set_ylabel(f'Model {i+1}')

#plt.xlabel('Test play')
#plt.suptitle('Rewards for each model iteration')
#plt.show()
#fig, axs = plt.subplots(len(data), 1, figsize=(10, 6), sharex=True, tight_layout=True)

#for i in range(len(data)):
#    rewards = data[i]
#    axs[i].bar(range(1, len(rewards) + 1), rewards, color='b')
#    axs[i].set_ylabel(f'Model {i+1}')

#plt.xlabel('Test play')
#plt.suptitle('Rewards for each model iteration')
#plt.show()


