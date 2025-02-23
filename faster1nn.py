# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:35:02 2024

@author: Jin Li
"""
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the 20newsgroups dataset
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# Initialize CountVectorizer with a suitable max_features
vectorizer = CountVectorizer(max_features=129796)
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target
X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

# Dimensionality reduction with TruncatedSVD

scaler = StandardScaler(with_mean=False)  
svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
pipeline = make_pipeline(scaler, svd)

X_train_reduced = pipeline.fit_transform(X_train)
X_test_reduced = pipeline.transform(X_test)

# Split a small portion of the reduced test set for time estimation
portion_reduced = 0.01
indices_reduced = np.random.choice(X_test_reduced.shape[0], int(X_test_reduced.shape[0] * portion_reduced), replace=False)
X_test_reduced_sample = X_test_reduced[indices_reduced]
y_test_reduced_sample = y_test[indices_reduced]

# Re-run k-NN classification with the reduced data
knn_clf_reduced = KNeighborsClassifier(n_neighbors=1)
start_time_reduced = timer()
knn_clf_reduced.fit(X_train_reduced, y_train)
predictions_reduced = knn_clf_reduced.predict(X_test_reduced_sample)
end_time_reduced = timer()
accuracy_reduced = accuracy_score(y_test_reduced_sample, predictions_reduced)

# Estimate the total computation time for the full reduced test set
estimated_time_full_reduced = (end_time_reduced - start_time_reduced) / portion_reduced
print(f"Estimated total computation time for the full reduced test set: {estimated_time_full_reduced:.2f} seconds")
print(f"Classification accuracy on the reduced test sample: {accuracy_reduced:.4f}")
