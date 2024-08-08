# Code written by Greta Frei (mrauch2@bu.edu), contact with any questions
# If adapted, please cite properly

# This file runs the t-SNE algorithm on the topic model in order to visualize the output in 2 dimensions

## Code based on the following https://towardsdatascience.com/visualizing-topic-models-with-scatterpies-and-t-sne-f21f228f7b02
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

# In order to properly adapt this code, set the following variables
model_output_folder = ""    # directory where the generated models are stored
data_set_location = ""      # csv file with the dataset for analysis
data_column = ""            # the name of the column in the dataset json with the preprocesed raw text
index_column = ""           # the name of the column in the dataset that stores the entry's unique id
columns_to_drop = ""        # list of columns from the dataframe to drop from the output

NUM_TOPICS = 15             # number of topics

# Load in data corpus
articles_df = pd.read_csv(data_set_location, index_col=index_column).dropna(subset=[data_column])
articles = articles_df[data_column].tolist()
tokenized_articles = [article.split() for article in articles]
dictionary = Dictionary(tokenized_articles)
corpus = [dictionary.doc2bow(article) for article in tokenized_articles]

# Load in the sheet with topic scores for each article
doc_topic_df = pd.read_csv(f'{model_output_folder}/doc_topic_lda{NUM_TOPICS}.csv', index_col=index_column)
doc_topic_df = doc_topic_df.drop(columns=columns_to_drop) 
doc_topic_df = doc_topic_df.astype(float)

print(doc_topic_df)

# Keep the well separated points (this makes the visualization cleaner)
arr = doc_topic_df[np.amax(doc_topic_df, axis=1) > 0.15]

# Extract the dominant topic number in each doc (used for coloring)
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
# Note that the hyper-parameters (init, perplexity, learning_rate) were tuned by hand
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='random', perplexity=200, learning_rate=500) 
tsne_lda = tsne_model.fit_transform(arr)

# Output dataframe with the results of the tSNE reduction
df = pd.DataFrame(tsne_lda, columns=['Dimension 1', 'Dimension 2'])
df['Top_Topic'] = topic_num
df['section_id'] = arr.index
df.index.name = 'id'
df.to_csv(f'tsne_lda{NUM_TOPICS}.csv')