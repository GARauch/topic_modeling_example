# Code written by Greta Frei (mrauch2@bu.edu), contact with any questions
# If adapted, please cite properly

# This code visualizes provides an alternative to the standard tSNE visualizations. 
# Instead of graphing each topic as a dot colored by their highest topic value, 
# each document is graphed as a pie chart with the pie chart representing the various topics that compose a single document.

## Code based on the following https://towardsdatascience.com/visualizing-topic-models-with-scatterpies-and-t-sne-f21f228f7b02
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import threading

# In order to properly adapt this code, set the following variables
model_output_folder = ""    # directory where the generated models are stored
index_column = ""           # the name of the column in the dataset that stores the entry's unique id
columns_to_drop = ""        # list of columns from the dataframe to drop from the output

# set number of topics
NUM_TOPICS = 15

# read in dataframe with topic scores for each article
doc_topic_df = pd.read_csv(f'{model_output_folder}/doc_topic_lda{NUM_TOPICS}.csv', index_col=index_column)
doc_topic_df = doc_topic_df.drop(columns=columns_to_drop)
doc_topic_df = doc_topic_df.astype(float)
doc_topic_df.rename(columns=lambda x: x.split()[-1], inplace=True)

# read in dataframe with tSNE dimension reduction information
df = pd.read_csv(f'{model_output_folder}/tsne_lda{NUM_TOPICS}.csv', index_col='section_id')

# Create a plot
fig, ax = plt.subplots(figsize=(30,12))
plt.title('t-SNE Clustering of LDA with {} Topics Colored by Percentage'.format(NUM_TOPICS))

# Choose a colormap
cmap = plt.colormaps.get_cmap('tab20')
color_map = [cmap(i) for i in range(NUM_TOPICS)]

# Draw a pie chart at a given point on the map
# based on this: https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
def drawPieMarker(id, xs, ys, color_map):
    ratios = doc_topic_df.loc[id]
    significant_topics = ratios[ratios >= 0.2]  # Only color a topic if it received a score of at least 20%
    colors = [color_map[int(idx)] for idx in significant_topics.index]
    sizes = [50] * len(colors)
    
    assert significant_topics.sum() <= 1
    
    # Create the markers for each portion of the piechart iteratively
    markers = []
    previous = 0
    for ratio, color, size in zip(significant_topics, colors, sizes):
        this = 2 * np.pi * ratio + previous
        angles = np.linspace(previous, this, 10)
        x = [0] + np.cos(angles).tolist() + [0]
        y = [0] + np.sin(angles).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        marker_size = np.abs(xy).max() ** 2 * size
        markers.append({'marker': xy, 's': marker_size, 'facecolor': color})
    
    # Draw each marker for the pie chart
    lock.acquire()
    for marker in markers:
        ax.scatter(xs, ys, **marker)
    lock.release()

# Method to call drawPieMarker to thread this program
def draw_pie_marker_threaded(idx, xs, ys, color_map):
    drawPieMarker(idx, xs, ys, color_map)

# Use multithreading to draw the pie markers
threads = []
lock = threading.Lock()
for idx, row in df.iterrows():
    xs = df.loc[idx, 'Dimension 1']
    ys = df.loc[idx, 'Dimension 2']
    thread = threading.Thread(target=draw_pie_marker_threaded, args=(idx, xs, ys, color_map))
    thread.start()
    threads.append(thread)

# Wait for threads to complete
for thread in threads:
    thread.join()

# Complete final graph display
legend_handles = [Patch(color=color_map[i], label='Topic {}'.format(i)) for i in range(NUM_TOPICS)]
plt.legend(handles=legend_handles, title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(right=0.79)  # Increase right margin to make space for the legend
plt.savefig('t-SNE Clustering of LDA with {} Topics Colored by Percentage.png'.format(NUM_TOPICS))
plt.show()