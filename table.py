import matplotlib.pyplot as plt
import numpy as np

# Data
data1 = [
    ["Mean", "4.09"],
    ["Median", "4.0"],
    ["Standard Deviation", "1.86"]
]

data2 = [
    ["Mean", "36.99"],
    ["Median", "35.5"],
    ["Standard Deviation", "12.01"]
]

data3 = [
    ["High school degree", "57"],
    ["College degree", "42"],
    ["Master's degree", "5"],
    ["PhD degree", "4"]
]

data4 = [
    ["Woman", "56"],
    ["Man", "49"],
    ["Nonbinary", "2"],
    ["Other...", "1"]
]

# Create figure and axes
fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs = axs.flatten()

# Function to create tables
def draw_table(ax, data, title):
    ax.axis('off')
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=None, cellColours=None)
    ax.set_title(title, fontweight="bold")

    # # Set font size for the table
    # for (i, j), cell in table.get_celld().items():
    #     cell.set_fontsize(10)  # Set consistent font size

    # Adjust column widths
    table.auto_set_column_width([0])  # Automatically set width for the first column
    for i in range(len(data)):  # Loop through each row to set the second column width
        table[i, 1].set_width(0.1)

# Plot the tables
draw_table(axs[0], data1, "Familiarity with Diabetes")
draw_table(axs[1], data2, "Age")
draw_table(axs[2], data3, "Education")
draw_table(axs[3], data4, "Gender")

# Adjust layout and display
plt.tight_layout()
plt.show()