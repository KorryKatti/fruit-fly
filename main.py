import pandas as pd

# Load the feather file (Apache Arrow format, fast)
df = pd.read_feather('connectome-weights-male-cns-v1.0-minconf-0.5.feather')

# Inspect
print(df.head())
print(df.columns)
print(df.shape)
