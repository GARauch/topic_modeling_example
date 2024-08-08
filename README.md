# topic_modeling_example
This repository provides starter code for doing topic analysis. It is based on a project I conducted in the spring of 2024 at Boston College.

Below is a description of the files in this repository.

* *clean_data.ipynb*
This Jupyter notebook presents the code I used to clean my data.

* *compute_lda_models.py*
This script computes an LDA topic model for a range of topics, and calculates the coherence score for each of these models. All model and visualization files are saved.

* *generate_doc_to_topic.py*
This script computes a CSV file with the topic scores for each article in the corpus.

* *generate_tsne_df.py*
This script runs the tSNE dimensionality reduction on the topic model data and produces a CSV with each dimensions for each topic. This reduces the data down to two axises so it can be easily graphed.

* *tsne_simple_vis.py*
This script visualizes the t-SNE reduction on a plot and saves the plot in the models folder. 

* tsne_optimized_piechart_vis.py
This visualization provides an alternative to the standard tSNE visualizations. Instead of graphing each topic as a dot colored by their highest topic value, each document is graphed as a pie chart with the pie chart representing the various topics that compose a single document.



