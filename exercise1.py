{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71025bae",
   "metadata": {},
   "outputs": [],
   "source": [
    "Question a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7c723a6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['alt.atheism',\n",
      " 'comp.graphics',\n",
      " 'comp.os.ms-windows.misc',\n",
      " 'comp.sys.ibm.pc.hardware',\n",
      " 'comp.sys.mac.hardware',\n",
      " 'comp.windows.x',\n",
      " 'misc.forsale',\n",
      " 'rec.autos',\n",
      " 'rec.motorcycles',\n",
      " 'rec.sport.baseball',\n",
      " 'rec.sport.hockey',\n",
      " 'sci.crypt',\n",
      " 'sci.electronics',\n",
      " 'sci.med',\n",
      " 'sci.space',\n",
      " 'soc.religion.christian',\n",
      " 'talk.politics.guns',\n",
      " 'talk.politics.mideast',\n",
      " 'talk.politics.misc',\n",
      " 'talk.religion.misc']\n",
      "Total of 11314 posts in the dataset and the total size is 22.05MB\n"
     ]
    }
   ],
   "source": [
    "# load the 20newsgroups dataset\n",
    "from sklearn.datasets import fetch_20newsgroups\n",
    "# \"pretty-print\" data structures in a way that's easier to read than just using the print function\n",
    "from pprint import pprint\n",
    "# size_mb function calculate the size of the documents in megabytes. \n",
    "# It takes docs (a list of strings) as an input and returns their combined size in megabytes. \n",
    "# It does this by encoding each string in UTF-8, finding the number of bytes for each string, \n",
    "# summing these up, and then dividing by 1 million to convert from bytes to megabytes.\n",
    "def size_mb(docs):\n",
    "    return sum(len(s.encode(\"utf-8\")) for s in docs) / 1e6\n",
    "\n",
    "data_train = fetch_20newsgroups(subset='train')\n",
    "data_test = fetch_20newsgroups(subset='test')\n",
    "\n",
    "pprint(data_train.target_names)\n",
    "\n",
    "print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "45173bc6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class id: 7 and class name rec.autos\n",
      "From: lerxst@wam.umd.edu (where's my thing)\n",
      "Subject: WHAT car is this!?\n",
      "Nntp-Posting-Host: rac3.wam.umd.edu\n",
      "Organization: University of Maryland, College Park\n",
      "Lines: 15\n",
      "\n",
      " I was wondering if anyone out there could enlighten me on this car I saw\n",
      "the other day. It was a 2-door sports car, looked to be from the late 60s/\n",
      "early 70s. It was called a Bricklin. The doors were really small. In addition,\n",
      "the front bumper was separate from the rest of the body. This is \n",
      "all I know. If anyone can tellme a model name, engine specs, years\n",
      "of production, where this car is made, history, or whatever info you\n",
      "have on this funky looking car, please e-mail.\n",
      "\n",
      "Thanks,\n",
      "- IL\n",
      "   ---- brought to you by your neighborhood Lerxst ----\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "Class id: 4 and class name comp.sys.mac.hardware\n",
      "From: guykuo@carson.u.washington.edu (Guy Kuo)\n",
      "Subject: SI Clock Poll - Final Call\n",
      "Summary: Final call for SI clock reports\n",
      "Keywords: SI,acceleration,clock,upgrade\n",
      "Article-I.D.: shelley.1qvfo9INNc3s\n",
      "Organization: University of Washington\n",
      "Lines: 11\n",
      "NNTP-Posting-Host: carson.u.washington.edu\n",
      "\n",
      "A fair number of brave souls who upgraded their SI clock oscillator have\n",
      "shared their experiences for this poll. Please send a brief message detailing\n",
      "your experiences with the procedure. Top speed attained, CPU rated speed,\n",
      "add on cards and adapters, heat sinks, hour of usage per day, floppy disk\n",
      "functionality with 800 and 1.4 m floppies are especially requested.\n",
      "\n",
      "I will be summarizing in the next two days, so please add to the network\n",
      "knowledge base if you have done the clock upgrade and haven't answered this\n",
      "poll. Thanks.\n",
      "\n",
      "Guy Kuo <guykuo@u.washington.edu>\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# first document of the dataset\n",
    "sample_id = 0\n",
    "# gets the category (as a numerical id) of the document that corresponds to sample_id.\n",
    "sample_target = data_train.target[sample_id]\n",
    "# displays the class id and the class name for this document. \n",
    "print(f'Class id: {sample_target} and class name {data_train.target_names[sample_target]}')\n",
    "# This prints the full text of the document with the id specified by sample_id.\n",
    "print(data_train.data[sample_id])\n",
    "\n",
    "sample_id = 1\n",
    "sample_target = data_train.target[sample_id]\n",
    "print(f'Class id: {sample_target} and class name {data_train.target_names[sample_target]}')\n",
    "print(data_train.data[sample_id])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ef68e9c7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import PorterStemmer\n",
    "from nltk.tokenize import word_tokenize\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "import re\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a1c9b12e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\justd\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\justd\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0ad5d16d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size of the vocabulary is 57060\n"
     ]
    }
   ],
   "source": [
    "# Data Preprocessing: \n",
    "# 1. lower all the words\n",
    "# 2. Stemming\n",
    "# 3. remove stop words\n",
    "# 4. tokenization\n",
    "# Initialize the Porter stemmer\n",
    "stemmer = PorterStemmer()\n",
    "\n",
    "# Retrieve the English stop words list from NLTK and convert it to a set for faster access\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def custom_tokenize(text):\n",
    "    # Lowercase the entire text\n",
    "    text = text.lower()\n",
    "    # Tokenize the text into words\n",
    "    tokens = word_tokenize(text)\n",
    "    # Remove stop words and stem the remaining words, allowing apostrophes in words\n",
    "    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and re.match(\"^[a-zA-Z']+$\", token)]\n",
    "    return filtered_tokens\n",
    "\n",
    "# Initialize CountVectorizer with the custom tokenizer\n",
    "vectorizer = CountVectorizer(tokenizer=custom_tokenize, stop_words=None)  # Explicitly disabling internal stop words\n",
    "\n",
    "# Example usage with your data\n",
    "# transforms the training data into a matrix of token counts.\n",
    "X_train = vectorizer.fit_transform(data_train.data)\n",
    "print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "30dbf8a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(11314, 57060)\n",
      "(11314,)\n"
     ]
    }
   ],
   "source": [
    "X_test = vectorizer.transform(data_test.data)\n",
    "y_train, y_test = data_train.target, data_test.target\n",
    "print(X_train.shape)\n",
    "print(y_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "106218f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.dummy import DummyClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d04c3225",
   "metadata": {},
   "outputs": [],
   "source": [
    "Question b & c & d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "48f76209",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification took 0.0004 seconds.\n",
      "Classification accuracy: 0.053\n"
     ]
    }
   ],
   "source": [
    "# Question b & c\n",
    "\n",
    "from timeit import default_timer as timer\n",
    "# Initialize the DummyClassifier to use the most_frequent strategy\n",
    "clf = DummyClassifier(strategy=\"most_frequent\")\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Start timer before classification\n",
    "start_time = timer()\n",
    "\n",
    "# Classify all test samples\n",
    "predictions = clf.predict(X_test)\n",
    "\n",
    "# End timer after classification\n",
    "end_time = timer()\n",
    "\n",
    "\n",
    "# Question d\n",
    "\n",
    "# Compute the elapsed time\n",
    "elapsed_time = end_time - start_time\n",
    "print(f\"Classification took {elapsed_time:.4f} seconds.\")\n",
    "\n",
    "# Compute and print the accuracy \n",
    "accuracy = accuracy_score(y_test, predictions)\n",
    "print(f\"Classification accuracy: {accuracy:.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0665ae7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from scipy.spatial.distance import cdist\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "798c58b3",
   "metadata": {},
   "source": [
    "Question e&f "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a25cc547",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of test samples used: 75\n",
      "Estimated total computation time for the full test set: 7347.47 seconds\n",
      "Classification accuracy on the test sample: 0.4000\n"
     ]
    }
   ],
   "source": [
    "def simple_nearest_neighbor(X_train, y_train, X_test):\n",
    "    \"\"\"\n",
    "    Simple nearest neighbor classifier.\n",
    "    Parameters:\n",
    "    - X_train: Training feature matrix.\n",
    "    - y_train: Training labels.\n",
    "    - X_test: Test feature matrix.\n",
    "    Returns:\n",
    "    - predictions: Predicted labels for the test data.\n",
    "    \"\"\"\n",
    "    # Compute Euclidean distances between each test sample and all training samples\n",
    "    distances = cdist(X_test, X_train, metric='euclidean')\n",
    "    \n",
    "    # Find the index of the nearest training sample for each test sample\n",
    "    nearest_neighbors_indices = np.argmin(distances, axis=1)\n",
    "    \n",
    "    # Use the labels of the nearest neighbors as predictions\n",
    "    predictions = y_train[nearest_neighbors_indices]\n",
    "    \n",
    "    return predictions\n",
    "\n",
    "# Select 0.01 of the test set\n",
    "portion = 0.01\n",
    "sample_indices = np.random.choice(X_test.shape[0], size=int(X_test.shape[0] * portion), replace=False)\n",
    "X_test_sample = X_test[sample_indices]\n",
    "y_test_sample = y_test[sample_indices]\n",
    "\n",
    "# Time the classification of the test sample\n",
    "start_time = timer()\n",
    "predictions_sample = simple_nearest_neighbor(X_train.toarray(), y_train, X_test_sample.toarray())\n",
    "end_time = timer()\n",
    "\n",
    "# Calculate accuracy on the test sample\n",
    "accuracy_sample = accuracy_score(y_test_sample, predictions_sample)\n",
    "\n",
    "# Print results\n",
    "print(f\"Number of test samples used: {X_test_sample.shape[0]}\")\n",
    "print(f\"Estimated total computation time for the full test set: {(end_time - start_time) / portion:.2f} seconds\")\n",
    "print(f\"Classification accuracy on the test sample: {accuracy_sample:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c69431a",
   "metadata": {},
   "outputs": [],
   "source": [
    "Question g & h"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b98dcfc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computation time for classifying the test set: 8.25 seconds\n",
      "Classification accuracy: 0.4241\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Initialize the KNeighborsClassifier for 1 nearest neighbor\n",
    "knn_clf = KNeighborsClassifier(n_neighbors=1)\n",
    "\n",
    "# Fit the classifier on the training data\n",
    "knn_clf.fit(X_train, y_train)\n",
    "\n",
    "# Time the classification of the entire test set\n",
    "start_time = timer()\n",
    "predictions = knn_clf.predict(X_test)\n",
    "end_time = timer()\n",
    "\n",
    "# Calculate accuracy on the entire test set\n",
    "accuracy = accuracy_score(y_test, predictions)\n",
    "\n",
    "# Print results from h\n",
    "print(f\"Computation time for classifying the test set: {end_time - start_time:.2f} seconds\")\n",
    "print(f\"Classification accuracy: {accuracy:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f4a56249",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification took 8.2520 seconds.\n",
      "Classification accuracy: 0.424\n",
      "Number of test samples used: 75\n",
      "Estimated total computation time for the full test set: 825.20 seconds\n",
      "Classification accuracy on the test sample: 0.4000\n"
     ]
    }
   ],
   "source": [
    "# Question d\n",
    "# Compute the elapsed time\n",
    "elapsed_time = end_time - start_time\n",
    "print(f\"Classification took {elapsed_time:.4f} seconds.\")\n",
    "# Compute and print the accuracy \n",
    "accuracy = accuracy_score(y_test, predictions)\n",
    "print(f\"Classification accuracy: {accuracy:.3f}\")\n",
    "\n",
    "# Question f\n",
    "\n",
    "print(f\"Number of test samples used: {X_test_sample.shape[0]}\")\n",
    "print(f\"Estimated total computation time for the full test set: {(end_time - start_time) / portion:.2f} seconds\")\n",
    "print(f\"Classification accuracy on the test sample: {accuracy_sample:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cdbd90b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
