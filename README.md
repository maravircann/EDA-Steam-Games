# Steam Games – Exploratory Data Analysis (EDA)

This project presents an exploratory data analysis on a dataset of Steam games, using Python libraries such as pandas, matplotlib, and seaborn.  

## Dataset

- Source: [Steam Store Games Dataset on Kaggle](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
- Includes: game name, genres, price, release date, user ratings, platform availability

## Technologies Used

- Python 3.10+
- pandas
- matplotlib
- seaborn
- Jupyter Notebook
- PyCharm IDE

## Key Analyses

- Distribution of game prices
- Top 10 most common genres
- Trend of game releases by year
- Correlation between price and user ratings
- Platform-specific availability insights

## Notes

- The dataset was cleaned (null values, incorrect formats).
- All graphs include labeled axes and titles.
- The notebook ends with a summary of findings.


## Advanced Analysis

To deepen the insights, I applied unsupervised machine learning techniques to segment the games based on numeric features such as ratings, playtime, price, and estimated player base.

## Principal Component Analysis (PCA) was used to reduce dimensionality and uncover latent features:
- PC1: Game Popularity (influenced by ratings and player base)
- PC2: Player Engagement (driven by average and median playtime)

The result revealed trade-offs between popularity and engagement.

## KMeans Clustering was then performed on the PCA-reduced data, resulting in 3 distinct game segments:

- ## Cluster 0 – Standard Games
  Moderate popularity and price. Suitable for seasonal promotion or bundles.

- ## Cluster 1 – Popular & Cheap
  Extremely popular, low-priced titles with massive playtime. Ideal for freemium models and community-driven marketing.

- ## Cluster 2 – Premium Niche 
  Higher-priced games with smaller but loyal player bases. Best positioned as premium experiences or narrative-rich titles.

Each cluster's behavioral profile helped define tailored marketing and pricing strategies.
