import matplotlib.pyplot as plt

plt.ion()  # interactive mode ON

# Initialize the figure and axes for live updates
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
fig.tight_layout(pad=3.0)

def plot(scores, mean_scores, survival_times):
    # Clear the previous content from axes
    ax1.clear()
    ax2.clear()

    # Plot scores and mean scores
    ax1.plot(scores, label='Score')
    ax1.plot(mean_scores, label='Mean Score')
    ax1.set_title('Scores & Mean Scores Over Time')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.set_ylim(bottom=0)

    # Plot survival times (frames alive)
    ax2.plot(survival_times, label='Frames Alive')
    ax2.set_title('Frames Alive Over Time')
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Frames')
    ax2.legend(loc='upper left')
    ax2.set_ylim(bottom=0)

    # Redraw the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)  # small pause to allow GUI event loop
