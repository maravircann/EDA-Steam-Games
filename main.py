import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sns.set(style='darkgrid')
df = pd.read_csv('steam.csv')

# data visualization
print('dataset size: ', df.shape)
print(df.head())
print(df.isnull().sum())

# data preparation
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

print(df[['release_date', 'release_year']].head())

print('Missing values after data conversion:', df.isnull().sum())

# visualizations and analysis
plt.figure(figsize=(10, 5))
sns.histplot(df[df['price'] < 60]['price'], bins=60, color='skyblue')
plt.title('Distribution of Game Prices (under $60)')
plt.xlabel('Price ($)')
plt.ylabel('Number of Games')
#plt.show()

plt.figure(figsize=(12, 6))
df['release_year'].value_counts().sort_index().plot(kind='line', marker='o')
plt.title('Number of Games Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.grid(True)
#plt.show()

df['genres'] = df['genres'].str.split(';')
all_genres = df.explode('genres')
top_genres = all_genres['genres'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, dodge=False, palette='viridis', legend=False)
plt.title('Top 10 Most Common Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='positive_ratings', y='negative_ratings', alpha=0.5)
plt.title('Positive vs. Negative Ratings')
plt.xlabel('Positive Ratings')
plt.ylabel('Negative Ratings')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
#plt.show()

# estimated player count
df['owners_min'] = df['owners'].str.split('-').str[0].astype(int)
top_owned = df.sort_values(by='owners_min', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='owners_min', y='name', data=top_owned, hue='name', dodge=False, palette='mako', legend=False)
plt.title('Top 10 Most Owned Games')
plt.xlabel('Estimated Owners (min)')
plt.ylabel('Game')
plt.tight_layout()
#plt.show()

# correlation matrix
plt.figure(figsize=(8, 6))
numeric_cols = ['positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'price', 'owners_min']
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Selected Numeric Features)')
plt.tight_layout()
#plt.show()

# top value for money games on steam
df = df[df['average_playtime'] > 0]
df['value_score'] = (df['positive_ratings'] * df['average_playtime']) / (df['price'] + 1)
top_value = df.sort_values(by='value_score', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='value_score', y='name', data=top_value, hue='name', dodge=False, palette='light:#5A9', legend=False)
plt.title('Top 10 Games by Value-for-Money Score')
plt.xlabel('Value Score')
plt.ylabel('Game')
plt.tight_layout()
#plt.show()

# worst value for money games
worst_value = df.sort_values(by='value_score', ascending=True).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='value_score', y='name', data=worst_value, hue='name', dodge=False, palette='light:#D55', legend=False)
plt.title('Bottom 10 Games by Value-for-Money Score')
plt.xlabel('Value Score')
plt.ylabel('Game')
plt.tight_layout()
#plt.show()


# combine pca, fa and cluster analysis for Tipuri de jocuri Steam pe baza comportamentului utilizatorilor și a recenziilor


selected_cols=['positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'price', 'owners_min']
df_selected=df[selected_cols].copy()
df_selected=df_selected[df_selected['average_playtime']>0]


scaler=StandardScaler()
scaled_data=scaler.fit_transform(df_selected[selected_cols])

#pricipal component analysis
#reduce dimensionality and help identify the main directions of variability in the data and visualize patterns
#game groupings based on user ratings, playtime, price, etc

pca=PCA(n_components=2)
pca_result=pca.fit_transform(scaled_data)
pca_df=pd.DataFrame(data=pca_result, columns=['pc1', 'pc2'])

# components and loading matrix
features = ['positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'price', 'owners_min']
loadings = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
print(loadings.T)

# based on the results of the loadings matrix, the principal component is associated with ratings and ownership metrics.
# this means that it can be interpreted as a measure of game popularity.
# the second pricipal component is influenced by average and median playtime, reflecting player engagement.
# the game price contributes very little to either component, suggesting it does not significantly explain variance in this dataset.


plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='pc1', y='pc2', alpha=0.5)
plt.title('PCA – Game Distribution (2D Projection)')
plt.xlabel('Game Popularity')
plt.ylabel('Player Engagement')
plt.tight_layout()
plt.show()

# the PCA distribution shows that most games on Steam are neither highly popular nor highly engaging.
# however, a few clear outliers stand out — these are games that perform exceptionally well in both popularity and player engagement, such as highly-rated titles with large player bases and long average playtimes.
# this supports the idea that while most games struggle to gain traction, top-performing games tend to succeed on multiple fronts.


# k clustering to reduce the complexity of the pca data and automaticaly group games based on the 2 principal components

