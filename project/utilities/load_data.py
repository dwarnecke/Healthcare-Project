import json
import nltk
import os
import pandas as pd
import re
import torch
import transformers
nltk.download('all')


def find_annotations(notes, annotations):
    """
    Find the set of annotations within a set of admission notes.
    :param notes: The notes to find the annotations
    :param annotations: The details for each annotation
    :return: The annotation texts found in the notes
    """

    annotation_dataset = []

    for annotation in annotations:

        # Determine the annotation text and code
        annotation_idxs = annotation['annotation']
        text = notes[int(annotation_idxs[0]):int(annotation_idxs[1])]
        if 'suicide_attempt' in annotation.keys(): code = annotation['category']
        elif 'suicide_ideation' in annotation.keys(): code = 'R45.851'
        else: code = 'N/A'
        if code == 'N/A': continue

        # Append the text and code to the findings list
        annotation_data = {'text': text, 'code': code}
        annotation_dataset.append(annotation_data)

    return annotation_dataset


def label_sentences(notes, annotations):
    """
    Label the notes sentences on whether they are annotated or not.
    :param notes: The notes to label
    :param annotations: The annotation text and labels in the notes
    :return: The notes sentences and their suicide related labels
    """

    sentences_dataset = []

    # Include every sentence in the notes in the dataset
    sentences = nltk.sent_tokenize(notes)
    for sentence in sentences:

        # Label the sentence assuming is not annotated
        sentence_data = {'text': sentence, 'suicidal': 0, 'code': None}

        # Correct the sentence details if it is annotated
        for annotation in annotations:
            if annotation['text'] in sentence:
                sentence_data = {'text': sentence, 'suicidal': 1, 'code': annotation['code']}
                break

        sentences_dataset.append(sentence_data)

    return sentences_dataset


def load_data(file_path):
    """
    Load notes sentences and their suicide annotation details.
    :param file_path: The path of the json file with admission notes and annotations to load
    :return: Sentences and their suicidality details
    """

    dataset = []

    # Save all admission notes listed in the file
    annotations = json.load(open(file_path))
    for admission in annotations.keys():

        # Load the notes for that hospital admission
        with open(f'get_data/corpus/{admission}') as file:
            admission_notes = file.read()

        # Label every sentence as annotated or not
        admission_annotations = annotations[admission].values()
        admission_annotations = find_annotations(admission_notes, admission_annotations)
        admission_sentences = label_sentences(admission_notes, admission_annotations)

        # Append the admission sentences to the dataset
        dataset.extend(admission_sentences)

    dataset = pd.DataFrame(dataset)

    return dataset