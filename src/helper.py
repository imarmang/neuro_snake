import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# Initialize the figure and axes for live updates
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
fig.tight_layout(pad=3.0)


def plot(scores, mean_scores, survival_times):
    # Clear the previous content from axes
    ax1.clear()
    ax2.clear()

    # Plot scores and mean scores on the first axis
    ax1.plot(scores, label='Score', color='blue')
    ax1.plot(mean_scores, label='Mean Score', color='orange')
    ax1.set_title('Scores & Mean Scores Over Time')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.set_ylim(ymin=0)

    # Plot survival times on the second axis
    ax2.plot(survival_times, label='Survival Time', color='green')
    ax2.set_title('Survival Time Over Time')
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Survival Time (Frames)')
    ax2.legend(loc='upper left')
    ax2.set_ylim(ymin=0)

    # Refresh the display with updated plots
    display.clear_output(wait=True)
    display.display(plt.gcf())

