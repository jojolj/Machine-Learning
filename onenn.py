# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:42:43 2024

@author: Jin Li
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier


#%%
# Question a
# load the 20newsgroups dataset
from sklearn.datasets import fetch_20newsgroups
# "pretty-print" data structures in a way that's easier to read than just using the print function
from pprint import pprint
# size_mb function calculate the size of the documents in megabytes. 
# It takes docs (a list of strings) as an input and returns their combined size in megabytes. 
# It does this by encoding each string in UTF-8, finding the number of bytes for each string, 
# summing these up, and then dividing by 1 million to convert from bytes to megabytes.
def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

pprint(data_train.target_names)

print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')
#%%
# Data Preprocessing: 
# 1. lower all the words
# 2. Stemming
# 3. remove stop words
# 4. tokenization
# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Retrieve the English stop words list from NLTK and convert it to a set for faster access
stop_words = set(stopwords.words('english'))

def custom_tokenize(text):
    # Lowercase the entire text
    text = text.lower()
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stop words and stem the remaining words, allowing apostrophes in words
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and re.match("^[a-zA-Z']+$", token)]
    return filtered_tokens

# Initialize CountVectorizer with the custom tokenizer
vectorizer = CountVectorizer(tokenizer=custom_tokenize, stop_words=None)  # Explicitly disabling internal stop words

# Example usage with your data
# transforms the training data into a matrix of token counts.
X_train = vectorizer.fit_transform(data_train.data)
print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}')
#%%
X_test = vectorizer.transform(data_test.data)
y_train, y_test = data_train.target, data_test.target
print(X_train.shape)
print(y_train.shape)
#%%
# Question b & c & d
# Initialize the DummyClassifier to use the most_frequent strategy
clf = DummyClassifier(strategy="most_frequent")
clf.fit(X_train, y_train)

# Start timer before classification
start_time = timer()

# Classify all test samples
predictions = clf.predict(X_test)

# End timer after classification
end_time = timer()


# Question d

# Compute the elapsed time
elapsed_time = end_time - start_time
print(f"Classification took {elapsed_time:.4f} seconds.")

# Compute and print the accuracy 
accuracy = accuracy_score(y_test, predictions)
print(f"Classification accuracy: {accuracy:.3f}")

## Result:
# Classification took 0.0004 seconds.
# Classification accuracy: 0.053
#%% Question e&f 
def simple_nearest_neighbor(X_train, y_train, X_test):
    """
    Simple nearest neighbor classifier.
    Parameters:
    - X_train: Training feature matrix.
    - y_train: Training labels.
    - X_test: Test feature matrix.
    Returns:
    - predictions: Predicted labels for the test data.
    """
    # Compute Euclidean distances between each test sample and all training samples
    distances = cdist(X_test, X_train, metric='euclidean')
    
    # Find the index of the nearest training sample for each test sample
    nearest_neighbors_indices = np.argmin(distances, axis=1)
    
    # Use the labels of the nearest neighbors as predictions
    predictions = y_train[nearest_neighbors_indices]
    
    return predictions

# Select 0.01 of the test set
portion = 0.01
sample_indices = np.random.choice(X_test.shape[0], size=int(X_test.shape[0] * portion), replace=False)
X_test_sample = X_test[sample_indices]
y_test_sample = y_test[sample_indices]

# Time the classification of the test sample
start_time = timer()
predictions_sample = simple_nearest_neighbor(X_train.toarray(), y_train, X_test_sample.toarray())
end_time = timer()

# Calculate accuracy on the test sample
accuracy_sample = accuracy_score(y_test_sample, predictions_sample)

# Print results
print(f"Number of test samples used: {X_test_sample.shape[0]}")
print(f"Estimated total computation time for the full test set: {(end_time - start_time) / portion:.2f} seconds")
print(f"Classification accuracy on the test sample: {accuracy_sample:.4f}")

# Result:
# Number of test samples used: 75
# Estimated total computation time for the test set: 7347.47 seconds
# Classification accuracy on the test sample: 0.4000
#%% Question g & h


# Initialize the KNeighborsClassifier for 1 nearest neighbor
knn_clf = KNeighborsClassifier(n_neighbors=1)

# Fit the classifier on the training data
knn_clf.fit(X_train, y_train)

# Time the classification of the entire test set
start_time = timer()
predictions = knn_clf.predict(X_test)
end_time = timer()

# Calculate accuracy on the entire test set
accuracy = accuracy_score(y_test, predictions)

# Print results from h
print(f"Computation time for classifying the test set: {end_time - start_time:.2f} seconds")
print(f"Classification accuracy: {accuracy:.4f}")

#result:
# Computation time for classifying the test set: 8.25 seconds
# Classification accuracy: 0.4241
#%%
# Question d
# Compute the elapsed time
elapsed_time = end_time - start_time
print(f"Classification took {elapsed_time:.4f} seconds.")
# Compute and print the accuracy 
accuracy = accuracy_score(y_test, predictions)
print(f"Classification accuracy: {accuracy:.3f}")

# Classification took 8.2520 seconds.
# Classification accuracy: 0.424

# Question f

print(f"Number of test samples used: {X_test_sample.shape[0]}")
print(f"Estimated total computation time for the full test set: {(end_time - start_time) / portion:.2f} seconds")
print(f"Classification accuracy on the test sample: {accuracy_sample:.4f}")


# Number of test samples used: 75
# Estimated total computation time for the test set: 825.20 seconds
# Classification accuracy on the test sample: 0.4000
#%%
from sklearn.decomposition import TruncatedSVD

# Fit and transform the training data to 25 dimensions using TruncatedSVD
svd = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
X_train_reduced = svd.fit_transform(X_train)

# Transform the testing data to 25 dimensions using the same transformation
X_test_reduced = svd.transform(X_test)

# For estimation of computation time on dimensionally reduced data
# Set up the train and test loaders using the reduced data
tensor_x_train_reduced = torch.Tensor(X_train_reduced)
tensor_x_test_reduced = torch.Tensor(X_test_reduced)

train_dataset_reduced = TensorDataset(tensor_x_train_reduced, tensor_y_train)
test_dataset_reduced = TensorDataset(tensor_x_test_reduced, tensor_y_test)

train_loader_reduced = DataLoader(train_dataset_reduced, batch_size=32, shuffle=True)
test_loader_reduced = DataLoader(test_dataset_reduced, batch_size=32, shuffle=False)

# ...

# Re-run classification on reduced data
# Initialize the KNeighborsClassifier for 1 nearest neighbor
knn_clf_reduced = KNeighborsClassifier(n_neighbors=1)

# Fit the classifier on the reduced training data
knn_clf_reduced.fit(X_train_reduced, y_train)

# Select 1% of the reduced test set for time estimation
portion_reduced = 0.01
sample_indices_reduced = np.random.choice(X_test_reduced.shape[0], size=int(X_test_reduced.shape[0] * portion_reduced), replace=False)
X_test_sample_reduced = X_test_reduced[sample_indices_reduced]
y_test_sample_reduced = y_test[sample_indices_reduced]

# Time the classification of the reduced test sample
start_time_reduced = timer()
predictions_sample_reduced = knn_clf_reduced.predict(X_test_sample_reduced)
end_time_reduced = timer()

# Calculate accuracy on the reduced test sample
accuracy_sample_reduced = accuracy_score(y_test_sample_reduced, predictions_sample_reduced)

# Print results for reduced data
print(f"Number of reduced test samples used: {X_test_sample_reduced.shape[0]}")
print(f"Estimated total computation time for the full reduced test set: {(end_time_reduced - start_time_reduced) / portion_reduced:.2f} seconds")
print(f"Classification accuracy on the reduced test sample: {accuracy_sample_reduced:.4f}")

