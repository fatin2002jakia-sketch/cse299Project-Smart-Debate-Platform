import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("kialo_large_dataset.csv")

print("Original rows:", len(df))




import pandas as pd

df = pd.read_csv("kialo_large_dataset.csv")

print(df.columns)
print(df.head())





# Keep needed columns
df = df[["topic", "side", "content"]].dropna()

# Remove unknown labels
df["side"] = df["side"].astype(str).str.lower().str.strip()
df = df[df["side"].isin(["pro", "con"])]

print("After removing unknown:", len(df))

# Combine topic + content
df["text"] = df["topic"] + " " + df["content"]

# Remove duplicates
df = df.drop_duplicates(subset=["text", "side"])

print("After removing duplicates:", len(df))
print("\nClass distribution:\n", df["side"].value_counts())

# Split data
X = df["text"]
y = df["side"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
with open("stance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as stance_model.pkl")
