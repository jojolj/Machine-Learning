from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
import numpy as np
from sklearn.model_selection import train_test_split

# Load the 20newsgroups dataset
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# Initialize CountVectorizer to remove English stopwords
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target
X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

# Classification with Random Forest on the full-dimensional data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = timer()
rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)
end_time = timer()
accuracy = accuracy_score(y_test, predictions)

# Compute the classification metrics and display results
print(f"Total computation time: {end_time - start_time:.2f} seconds")
print(f"Classification accuracy on the full test set: {accuracy:.4f}")
