import streamlit as st
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATA
# -----------------------------
fake = pd.read_csv("fake_news_1000.csv")
real = pd.read_csv("real_news_1000.csv")

data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

# -----------------------------
# MODEL (TRAIN-TEST SPLIT)
# -----------------------------
X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
X_vec_train = vectorizer.fit_transform(X_train)
X_vec_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_vec_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📰 AI-Based Fake News Detection Dashboard")
st.subheader("📰 Automated News Analysis (No Manual Input)")

# -----------------------------
# SESSION STATE
# -----------------------------
if "combined_data" not in st.session_state:
    st.session_state.combined_data = pd.DataFrame()

if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = data.copy()

# -----------------------------
# RANDOM NEWS
# -----------------------------
if st.button("Check Random News"):
    news = random.choice(list(data['text']))
    vect = vectorizer.transform([news])
    probs = model.predict_proba(vect)[0]
    pred = 1 if probs[1] > 0.45 else 0

    st.write("📰 News:", news)
    st.write(f"🔴 Fake: {probs[0]*100:.2f}% | 🟢 Real: {probs[1]*100:.2f}%")

    if pred == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")

    st.session_state.filtered_data = data[data['text'] == news]

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df['text'] = df['text'].apply(lambda x: clean_text("news report " + x))
        vect = vectorizer.transform(df['text'])
        probs = model.predict_proba(vect)
        preds = [1 if p[1] > 0.45 else 0 for p in probs]
        df['label'] = preds

        st.session_state.combined_data = pd.concat(
            [st.session_state.combined_data, df],
            ignore_index=True
        )

# -----------------------------
# RESET BUTTON
# -----------------------------
if st.button("Reset Uploaded Data"):
    st.session_state.combined_data = pd.DataFrame()
    st.session_state.filtered_data = data.copy()
    st.session_state.fake_count = 5
    st.session_state.real_count = 5
    st.success("Data cleared successfully!")

# -----------------------------
# SELECTION OF FAKE & REAL
# -----------------------------
if not st.session_state.combined_data.empty:
    st.write("📂 Combined Uploaded Data")
    fake_data = st.session_state.combined_data[st.session_state.combined_data['label']==0]
    real_data = st.session_state.combined_data[st.session_state.combined_data['label']==1]

    # ----- Input number of Fake News -----
    fake_input = st.text_input(
        f"Enter number of Fake News (0-{len(fake_data)}):",
        value=str(min(5, len(fake_data))),
        key="fake_count_input"
    )
    try:
        fake_n = int(fake_input)
        if fake_n < 0: fake_n = 0
        elif fake_n > len(fake_data): fake_n = len(fake_data)
    except:
        fake_n = min(5, len(fake_data))
    selected_fake = fake_data.head(fake_n)

    # ----- Input number of Real News -----
    real_input = st.text_input(
        f"Enter number of Real News (0-{len(real_data)}):",
        value=str(min(5, len(real_data))),
        key="real_count_input"
    )
    try:
        real_n = int(real_input)
        if real_n < 0: real_n = 0
        elif real_n > len(real_data): real_n = len(real_data)
    except:
        real_n = min(5, len(real_data))
    selected_real = real_data.head(real_n)

    st.session_state.filtered_data = pd.concat([selected_fake, selected_real], ignore_index=True)
    st.write(f"📊 Showing {fake_n} Fake and {real_n} Real News")
    st.write(st.session_state.filtered_data)

# -----------------------------
# DASHBOARD (ALL ORIGINAL GRAPHS)
# -----------------------------
st.subheader("📊 AI Analysis Dashboard")
filtered_data = st.session_state.filtered_data

# PIE CHART
counts = filtered_data['label'].value_counts()
labels = ["Fake" if i==0 else "Real" for i in counts.index]
colors = ["red" if i==0 else "green" for i in counts.index]

fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
ax1.set_title("Fake vs Real Distribution")
st.pyplot(fig1)

# BAR CHART
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(labels, counts.values, color=colors)
ax_bar.set_title("Fake vs Real Count")
st.pyplot(fig_bar)

# LINE CHART
trend = filtered_data['label'].rolling(5).mean()
st.line_chart(trend)

# WORD COUNT
filtered_data['word_count'] = filtered_data['text'].apply(lambda x: len(x.split()))

# SCATTER
fig2, ax2 = plt.subplots()
color_map = ['red' if i==0 else 'green' for i in filtered_data['label']]
ax2.scatter(filtered_data['word_count'], filtered_data['label'],
            c=color_map, alpha=0.6)
ax2.set_title("Word Count vs News Type")
st.pyplot(fig2)

# CLUSTERED BAR
bins = [0,10,20,30,50,100]
filtered_data['range'] = pd.cut(filtered_data['word_count'], bins)
grouped = filtered_data.groupby(['range','label']).size().unstack(fill_value=0)
grouped.columns = ["Fake" if col==0 else "Real" for col in grouped.columns]

fig3, ax3 = plt.subplots()
grouped.plot(kind='bar', ax=ax3, color=['red','green'][:len(grouped.columns)])
ax3.set_title("Clustered Bar Chart")
st.pyplot(fig3)

# -----------------------------
# MODEL PERFORMANCE (TRAIN-TEST DATA)
# -----------------------------
st.subheader("📈 Model Performance (Test Data)")

# CONFUSION MATRIX
y_pred = model.predict(X_vec_test)
cm = confusion_matrix(y_test, y_pred)

fig_perf1, ax_perf1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])
ax_perf1.set_title("Confusion Matrix (Test Data)")
st.pyplot(fig_perf1)

# ACCURACY GRAPH
if cm.size == 4:
    TN, FP, FN, TP = cm.ravel()
    fake_acc = TN/(TN+FP) if (TN+FP)!=0 else 0
    real_acc = TP/(TP+FN) if (TP+FN)!=0 else 0

    fig_perf2, ax_perf2 = plt.subplots()
    ax_perf2.bar(['Fake','Real'], [fake_acc, real_acc], color=['red','green'])
    ax_perf2.set_title("Accuracy (Test Data)")
    st.pyplot(fig_perf2)

# ROC CURVE
y_prob = model.predict_proba(X_vec_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig_perf3, ax_perf3 = plt.subplots()
ax_perf3.plot(fpr, tpr, label="AUC = " + str(round(roc_auc,2)))
ax_perf3.plot([0,1],[0,1],'--')
ax_perf3.set_title("ROC Curve (Test Data)")
ax_perf3.legend()
st.pyplot(fig_perf3)