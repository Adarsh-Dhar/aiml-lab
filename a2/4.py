import pandas as pd

# Creating the DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [24, 30, 29, 35],
    "City": ["New York", "Chicago", "San Francisco", "New York"]
}

df = pd.DataFrame(data)

# Filtering rows where age is greater than 28
filtered_df = df[df["Age"] > 28]

print(filtered_df)
