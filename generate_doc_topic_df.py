# Code written by Greta Frei (mrauch2@bu.edu), contact with any questions
# If adapted, please cite properly
# Code based on the following https://towardsdatascience.com/visualizing-topic-models-with-scatterpies-and-t-sne-f21f228f7b02
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

# This script computes a CSV file with the topic scores for each article in the corpus.
# It is used in the visualization scripts to speed up the visualizing process

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

# Set the number of topics you want
# In order to properly adapt this code, set the following variables
model_output_folder = ""    # directory where the generated models are stored
data_set_location = ""      # csv file with the dataset for analysis
data_column = ""            # the name of the column in the dataset json with the preprocesed raw text
index_column = ""           # the name of the column in the dataset that stores the entry's unique id
min_topic = 2               # first number of topics to try
NUM_TOPICS = 30             # This number determines which model file is read in 

# load in model
lda_model = LdaModel.load(f'{model_output_folder}/lda{NUM_TOPICS}.model')
articles_df = pd.read_csv(data_set_location, index_col=index_column).dropna(subset=[data_column])
articles = articles_df[data_column].tolist()
tokenized_articles = [article.split() for article in articles]
dictionary = Dictionary(tokenized_articles)
corpus = [dictionary.doc2bow(article) for article in tokenized_articles]

# Generate column names (topic 0, topic 1, topic 2...)
column_names = ['topic {}'.format(i) for i in range(NUM_TOPICS)]
doc_topic_df = pd.DataFrame(columns=column_names)
doc_topic_df['article_contents'] = ''

# Iterate over all documents to get their scores for each topic
for x, doc in enumerate(corpus):
    score_list = lda_model.get_document_topics(doc)
    doc_id = articles_df.index[x]
    doc_topic_df.loc[doc_id, 'article_contents'] = articles_df.loc[doc_id, data_column]
    for topic_num, score in score_list:
        doc_topic_df.loc[doc_id, f'topic {topic_num}'] = score

# Fill all null columns with zeroes
doc_topic_df.fillna(0, inplace=True)
doc_topic_df.index.name = index_column

# get publication year
doc_topic_df['pub_date'] = articles_df['date']

# get top topic
max_topic_col = doc_topic_df[['topic {}'.format(i) for i in range(NUM_TOPICS)]].idxmax(axis=1)
doc_topic_df['max_topic'] = max_topic_col.str.split().str[-1].astype(int)

# save the output
print(doc_topic_df)
doc_topic_df.to_csv(f'{model_output_folder}/doc_topic_lda{NUM_TOPICS}.csv')