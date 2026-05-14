import pandas as pd

# Load your dataset
df = pd.read_csv("kialo_large_dataset.csv", encoding='utf-8', engine='python')

print("Before cleaning:", df.shape)

# Remove unknown sides
df = df[df['side'] != 'unknown']

# Remove empty arguments
df = df.dropna(subset=['content'])

# Remove very short arguments (optional but good)
df = df[df['content'].str.len() > 20]

print("After cleaning:", df.shape)

# Save new clean file
df.to_csv("clean_kialo.csv", index=False)

print("✅ Clean dataset created: clean_kialo.csv")