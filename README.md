# Suicide Notes — Inferring suicide diagnoses from medical notes

------------------------

Credit for all code outside the project directory goes to https://github.com/bsinghpratap/ScAN/tree/main

Instructions for creating the necessary files for training new models
* Load the notes data into csv files
  * Download a copy of the MIMIC NOTEEVENTS.csv file from physio.net
  * Place the csv file in the get_data/resources directory
  * Load the relevant notes by running the get_data/get_HADM_files.ipynb jupyter notebook
  * Save the sentences of notes into files by running the project/get_data.ipynd jupyter notebook
* Download the embedding vectors
  * Download the 400K, 200 dimensional GloVe word vectors from https://nlp.stanford.edu/projects/glove/
  * Place the embeddings in the project/datasets folder
