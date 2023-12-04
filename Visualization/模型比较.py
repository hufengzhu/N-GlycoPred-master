import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter


linewidth = 3
alpha = 0.8

# Set global font size
mpl.rcParams['font.size'] = 20
# Set global font size for labels
mpl.rcParams['axes.labelsize'] = 20
# Set global font size for tick labels
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['font.sans-serif'] = ['Arial']  # Specify the default font as SimHei to support Chinese
mpl.rcParams['axes.unicode_minus'] = False  # This is needed to ensure that the minus sign is displayed correctly


labels = np.array(['MCC', 'Acc', 'Sp', 'Sn', 'AUC'])  # 用的时候替换成响应的Metrics
# # hm / one-hot
stats1 = np.array([87.50,93.60,89.81, 97.37,94.81])
stats2 = np.array([88.01,93.88,90.55,97.21,95.06])
stats3 = np.array([38.68,68.96,58.61,79.31,73.75])


# # Function to create a radar chart for a single model
def create_radar_chart(ax, angles, data, label, color):
    # Number of variables
    num_vars = len(data)
    # Compute angle for each bar
    bar_width = 2 * np.pi / num_vars

    # Create bars
    for i in range(num_vars):
        ax.bar(angles[i], data[i], width=bar_width, color=color, alpha=0.3, edgecolor='black')

    # Set the format for the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    # ax.spines['polar'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

    # Annotate each point with the data value
    for i, txt in enumerate(data):
        angle_rad = angles[i]
        if angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        ax.annotate(txt, (angle_rad, data[i]), textcoords="offset points",
                    xytext=(-12, 8), ha='center', color='black', fontsize=15)


# Create figure and subplots
fig, axs = plt.subplots(figsize=(18, 5), nrows=1, ncols=3, subplot_kw=dict(polar=True))
# Calculate angles
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# Extend data to close the loop
stats1 = np.concatenate((stats1, [stats1[0]]))
stats2 = np.concatenate((stats2, [stats2[0]]))
stats3 = np.concatenate((stats3, [stats3[0]]))

# Create each subplot
create_radar_chart(axs[0], angles, stats1, 'Model 1', 'lightblue')
create_radar_chart(axs[1], angles, stats2, 'Model 2', 'green')
create_radar_chart(axs[2], angles, stats3, 'Model 3', 'pink')


# Set the title and display the plot
# fig.suptitle('Comparison of Models', fontweight='bold', fontsize=20, y=1.0)
plt.subplots_adjust()
plt.savefig('hm_onehot.jpg')
plt.show()