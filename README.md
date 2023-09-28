## Overview

This repository contains code for a machine learning model that classifies news articles as either real or fake based on their content. The model is trained on a dataset consisting of news article titles and text, with corresponding labels indicating whether the news is real (1) or fake (0).

## Dataset

The dataset used for this project, referred to as **(WELFake)**, is a compilation of news articles from various sources, including Kaggle, McIntire, Reuters, and BuzzFeed Political. It comprises 72,134 news articles, with 35,028 labeled as real news and 37,106 labeled as fake news.

**Citation**: If you use this dataset in your work, please cite the following paper:

- Author(s): P. K. Verma, P. Agrawal, I. Amorim, and R. Prodan
- Title: WELFake: Word Embedding Over Linguistic Features for Fake News Detection
- Published in: IEEE Transactions on Computational Social Systems
- Volume: 8
- Number: 4
- Pages: 881-893
- Published Date: August 2021
- DOI: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519)

## Purpose

This project is created for my personal educational purposes.

## Contents

- `generate_model.py`: Python script for training and saving the fake news classification model.
- `main.py`: Python script for user interaction and prediction using the trained model.
- `WELFake_Dataset.csv`: The dataset containing news articles, titles, text, and labels.

## Usage

1. Run `generate_model.py` to train the fake news classification model. The model and TF-IDF vectorizer will be saved as pickle files.

2. Use `main.py` to interact with the model. Enter a news title and text, and the model will predict whether the news is likely fake or real.
