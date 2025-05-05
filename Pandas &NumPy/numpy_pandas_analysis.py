import numpy as np
import pandas as pd


df = pd.read_csv('OpinRank.csv')

ratings = df['overall_rating'].to_numpy()

# Find the mean rating using NumPy
print(np.mean(ratings))

# Find the median rating using NumPy
print(np.median(ratings))

# Find the standard deviation of ratings using NumPy
print(np.std(ratings))

# Find the variance of ratings using NumPy
print(np.var(ratings))

# Find the maximum and minimum rating using NumPy functions
print(np.max(ratings), np.min(ratings))

# Count the number of reviews with ratings greater than 4 using NumPy
print(np.sum(ratings > 4))

# Extract the array of all ratings as a NumPy array
print(ratings)

# Find the sum of all ratings using NumPy
print(np.sum(ratings))

# Use NumPy to normalize (min-max scale) the ratings between 0 and 1
print((ratings - np.min(ratings)) / (np.max(ratings) - np.min(ratings)))

# Use NumPy to find unique rating values and their counts
print(np.unique(ratings, return_counts=True))

# Find the total number of reviews using Pandas
print(len(df))

# Display the first 5 rows of the dataset using Pandas
print(df.head())

# Find the number of missing values in each column using Pandas
print(df.isnull().sum())

# Fill missing ratings with the mean rating using Pandas
print(df['overall_rating'].fillna(df['overall_rating'].mean()))

# Find the most common reviewer (who gave maximum reviews) using Pandas
print(df['docid'].mode().iloc[0], df['docid'].value_counts().iloc[0])

# Group the dataset by product (Car/Hotel) and find average rating per product
print(df.groupby('docid').agg({'overall_rating': 'mean'}))

# Create a new column "Review_Length" showing number of words in Review_Text
print(df['docid'].str.len())

# Find the review with the maximum number of words in Review_Text using Pandas
print(df.loc[df['docid'].str.len().idxmax()])

# Sort all reviews by rating in descending order using Pandas
print(df.sort_values('overall_rating', ascending=False))

# Create a pivot table showing average rating per reviewer using Pandas
print(pd.pivot_table(df, values='overall_rating', index='docid', aggfunc='mean'))
