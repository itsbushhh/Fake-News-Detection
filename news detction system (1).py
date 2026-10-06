import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOAD DATA
fake = pd.read_csv("fake_news_1000.csv")
real = pd.read_csv("real_news_1000.csv")

# COMBINE DATA
data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

print("Total rows:", len(data))
print(data['label'].value_counts())

print("Dataset Loaded")
print(data.head())

# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

# SPLIT DATA
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# LOGISTIC REGRESSION
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))

# RANDOM FOREST
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)

rf_pred = rf_model.predict(X_test_vec)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

# ============================================
# CLASS-WISE ACCURACY (FAKE vs REAL)
# ============================================

cm = confusion_matrix(y_test, lr_pred)

print("\nConfusion Matrix:\n", cm)

# Extract values
TN, FP, FN, TP = cm.ravel()

# Accuracy for each class
fake_accuracy = TN / (TN + FP)
real_accuracy = TP / (TP + FN)

print("\nFake News Accuracy:", fake_accuracy)
print("Real News Accuracy:", real_accuracy)

# ============================================
# GRAPH: FAKE vs REAL ACCURACY
# ============================================

labels = ['Fake News', 'Real News']
values = [fake_accuracy, real_accuracy]

colors = ['red', 'green']

plt.figure()
plt.bar(labels, values, color=colors)
plt.title("Accuracy: Fake vs Real News")
plt.ylabel("Accuracy")
plt.show()

# ============================================
# CLASS DISTRIBUTION GRAPH
# ============================================

counts = data['label'].value_counts()

labels = ['Fake', 'Real']
values = [counts[0], counts[1]]

plt.figure()
plt.bar(labels, values, color=['red', 'green'])
plt.title("Dataset Distribution (Fake vs Real)")
plt.ylabel("Number of News")
plt.show()

# ============================================
# PIE CHART
# ============================================

plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
plt.title("Fake vs Real News Percentage")
plt.show()

# ============================================
# ERROR ANALYSIS
# ============================================

errors = (y_test != lr_pred)

plt.figure()
plt.bar(['Correct', 'Incorrect'], 
        [sum(~errors), sum(errors)])
plt.title("Correct vs गलत Predictions")
plt.show()

# ============================================
# CONFUSION MATRIX (FAKE vs REAL)
# ============================================

cm = confusion_matrix(y_test, lr_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])

plt.title("Confusion Matrix (Fake vs Real News)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# PREDICTION FUNCTION
def predict_news(text):
    text = clean_text(text)
    vect = vectorizer.transform([text])
    
    lr = lr_model.predict(vect)[0]
    rf = rf_model.predict(vect)[0]
    
    print("\nPrediction:")
    print("Logistic Regression:", "Real" if lr==1 else "Fake")
    print("Random Forest:", "Real" if rf==1 else "Fake")

# TEST INPUT
predict_news("Government announces new policy for education")