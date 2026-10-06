import pandas as pd
import random

# -------- REAL NEWS SAMPLE --------
real_samples = [
    "Government announces new education policy",
    "Scientists discover breakthrough in cancer research",
    "Stock market reaches record high",
    "New technology improves electric vehicle efficiency",
    "International summit focuses on climate change",
    "University develops low cost water purification system",
    "New railway project inaugurated by government",
    "Economic growth rate increases this quarter",
    "Researchers develop new vaccine for virus",
    "Space agency launches satellite successfully"
]

# -------- FAKE NEWS SAMPLE --------
fake_samples = [
    "Aliens landed in India yesterday",
    "Man becomes invisible using secret formula",
    "Time travel machine discovered by teenager",
    "Secret society controls entire world economy",
    "Magical cure heals all diseases instantly",
    "Humans found living on Mars secretly",
    "Government hiding existence of dragons",
    "Person claims to live without food for years",
    "Scientists create immortality pill",
    "Portal to another world found under ocean"
]

# Generate 1000 rows each
real_data = {
    "text": [random.choice(real_samples) for _ in range(1000)],
    "label": [1]*1000
}

fake_data = {
    "text": [random.choice(fake_samples) for _ in range(1000)],
    "label": [0]*1000
}

# Convert to DataFrame
real_df = pd.DataFrame(real_data)
fake_df = pd.DataFrame(fake_data)

# Save files
real_df.to_csv("real_news_1000.csv", index=False)
fake_df.to_csv("fake_news_1000.csv", index=False)

print("✅ 1000-row datasets created successfully!")