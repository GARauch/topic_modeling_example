# Code written by Greta Frei (mrauch2@bu.edu), contact with any questions
# If adapted, please cite properly

# This code creates an LDA model for a range of topic numbers and determines the best model.
# All model outputs are saved for future use.

import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary 
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# In order to properly adapt this code, set the following variables
model_output_folder = ""    # directory where the generated models are stored
data_set_location = ""      # csv file with the dataset for analysis
data_column = ""            # the name of the column in the dataset json with the preprocesed raw text
index_column = ""           # the name of the column in the dataset that stores the entry's unique id
min_topic = 2               # first number of topics to try
max_topic = 40              # last number of topics to try
step = 1                    # step to make between topics

# This method computes the LDA model for a given number of topics
# It writes an html file visualizing the model and saves the gensim model files
# It then computes the model's coherence score using the u_mass metric. This score is returned
def compute_coherence_for_topic(num_topics, corpus, dictionary, texts):
    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, 
                             passes=50, workers=8)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, f'{model_output_folder}/lda_visualization_ncwc_{num_topics}_topics.html')
    lda_model.save(f'{model_output_folder}/lda{num_topics}.model')
    coherence_model = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass', 
                                     dictionary=dictionary, texts=texts)
    coherence_score = coherence_model.get_coherence()
    
    print(f'Topics: {num_topics} Coherence: {coherence_score}')
    return coherence_score

# Loading in the data set
articles_df = pd.read_csv(data_set_location, index_col=index_column).dropna(subset=[data_column])
articles = articles_df[data_column].tolist()
tokenized_articles = [article.split() for article in articles]
dictionary = Dictionary(tokenized_articles)
corpus = [dictionary.doc2bow(article) for article in tokenized_articles]
coherence_values = []

# Create topic models for 2 through 40 topics
for num_topics in range(min_topic, max_topic, step):
    coherence_val = compute_coherence_for_topic(num_topics, corpus, dictionary, tokenized_articles)
    coherence_values.append(coherence_val)

# Create a plot of the corresponding coherence scores
x = range(min_topic, max_topic, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()