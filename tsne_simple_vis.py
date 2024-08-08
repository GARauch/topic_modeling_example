# Code written by Greta Frei (mrauch2@bu.edu), contact with any questions
# If adapted, please cite properly

## Code based on the following https://towardsdatascience.com/visualizing-topic-models-with-scatterpies-and-t-sne-f21f228f7b02
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

# In order to properly adapt this code, set the following variables
model_output_folder = ""    # directory where the generated models are stored
data_set_location = ""      # csv file with the dataset for analysis
NUM_TOPICS = 15

df = pd.read_csv(f'{model_output_folder}/tsne_lda{NUM_TOPICS}.csv', index_col='id')
df['Top_Topic'] = df["Top_Topic"]

# Plot the scatter plot using Seaborn
plt.figure(figsize=(30, 12))
sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df, palette='tab20', hue='Top_Topic') # You can specify your own palette
plt.title('t-SNE Clustering of LDA with {} Topics'.format(NUM_TOPICS))
plt.savefig('t-SNE Clustering of LDA with {} Topics.png'.format(NUM_TOPICS))
plt.show()