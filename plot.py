
#import plt
import matplotlib.pyplot as plt
#import numpy as np
import numpy as np
arrayOfAverages = []

#load averages form averages.txt (I saved arrays in this txt file with /n as a seperator)
with open("models/averages.txt", "r") as f:
    arr = [eval(line) for line in f.readlines()]
arrayOfAverages = arr
#print(arr)
#print("Arr Print End;")
averages = [sum(array) / len(array) if len(array) > 0 else 0 for array in arrayOfAverages]

##plot averages as a line graph
#plt.plot(averages)
#plt.ylabel('Average')
#plt.xlabel('Episode')
#plt.show()


# Calculate averages
avg_rewards = [sum(array1)/len(array1) for array1 in arrayOfAverages]
data = arrayOfAverages



def plotInteractive(data):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    lines = []

    # For each model, plot their average reward over time
    for model_index in range(len(data[-1])):
        model_scores = [gen[model_index] for gen in data if model_index < len(gen)]
        generations = range(model_index + 1, len(model_scores) + model_index + 1)
        # Plotting the line with smaller markers
        line, = ax.plot(generations, model_scores, marker='o', linestyle='-', label=f'Model {model_index+1}', markersize=5)
        # Plotting the first point with a larger marker
        ax.plot(generations[0], model_scores[0], marker='o', color='cyan', markersize=10)
        lines.append(line)

    plt.title('Model average reward over time')
    plt.xlabel('Generation')
    plt.ylabel('Average reward')


    leg = plt.legend()
    # Make lines invisible by default
    for line in lines:
        line.set_visible(False)

    # Make the first line visible, as it was the first to be trained
    lines[0].set_visible(True)

    lined = {}  # Will map legend lines to original lines
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # Enable picking on the legend line
        lined[legline] = origline

    # Define what to do on pick event
    def on_pick(event):
        # On the pick event, find the original line corresponding to the
        # legend proxy line, and toggle the visibility.
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        if visible:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.grid()
    plt.show()





def plotBaseAv(avg_rewards):
    # Plot
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, marker='o')
    plt.title('Average reward over time')
    plt.xlabel('Model iteration')
    plt.ylabel('Average reward')
    plt.grid()
    plt.show()






def plotavSmallDots(data):
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

def plotAverageWithBigDots(data):
    # For each model, plot their average reward over time
    for model_index in range(len(data[-1])):
        model_scores = [gen[model_index] for gen in data if model_index < len(gen)]
        generations = range(model_index + 1, len(model_scores) + model_index + 1)
        # Plotting the line with smaller markers
        plt.plot(generations, model_scores, marker='o', linestyle='-', label=f'Model {model_index+1}', markersize=5)
        # Plotting the first point with a larger marker
        plt.plot(generations[0], model_scores[0], marker='o', color='red', markersize=10)

    plt.title('Model average reward over time')
    plt.xlabel('Generation')
    plt.ylabel('Average reward')
    plt.legend()
    plt.grid()
    plt.show()



def plotAvav(averages):
    # create an array of the same length for the x values (time)
    time = np.arange(len(averages))

    # calculate the line of best fit
    coefficients = np.polyfit(time, averages, 1)
    polynomial = np.poly1d(coefficients)

    # create the line of best fit
    line_of_best_fit = polynomial(time)

    # plot your data
    plt.plot(time, averages, 'o')
    plt.plot(time, line_of_best_fit)
    plt.show()


length = 0
prevlength = 0
line = 0
for array in arr:
    line+=1
    length = len(array) - prevlength
    if(length != 1):
        print("Error in Averages at line: ", line)
    prevlength = len(array)



#plotavSmallDots(data)
#plotAverageWithBigDots(data)

#plotInteractive(data)
plotBaseAv(avg_rewards)
plotAvav(avg_rewards[0:])

rewards = avg_rewards
# Compute the cumulative sum and divide by the number of elements so far at each point
avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

# Plotting
plt.plot(avg_rewards)
plt.title('Average Reward Growth Over Time')
plt.xlabel('Time')
plt.ylabel('Average Reward')
plt.show()
