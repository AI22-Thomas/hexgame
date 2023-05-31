
#import plt
import matplotlib.pyplot as plt
arrayOfAverages = []

#load averages form averages.txt (I saved arrays in this txt file with /n as a seperator)
with open("models/averages.txt", "r") as f:
    arr = [eval(line) for line in f.readlines()]
arrayOfAverages = arr


averages = [sum(array) / len(array) if len(array) > 0 else 0 for array in arrayOfAverages]

#plot averages as a line graph
plt.plot(averages)
plt.ylabel('Average')
plt.xlabel('Episode')
plt.show()


# Assuming each array represents a different time point
time = list(range(1, len(arr) + 1))  # Time points from 1 to 16

# plot for each model
for model_idx in range(len(arr[0])):
    plt.plot(time, [item[model_idx] for item in arr], label=f'Model {model_idx+1}')

# labels and legend
plt.xlabel('Time')
plt.ylabel('Value')
#plt.legend()

# show the plot
plt.show()
#This will create a line plot where x-axis represents time and y-axis represents the value of each model

    


