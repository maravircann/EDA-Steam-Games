## Steam Games – Exploratory Data Analysis (EDA) & Machine Learning
This project offers a comprehensive exploratory data analysis (EDA) and predictive modeling on a dataset of Steam games. The goal is to uncover trends, segment game behavior, and build a model to predict player engagement using real-world features like ratings, price, and player base.

## Dataset
Source: Steam Store Games Dataset on Kaggle

Features: Game name, genres, release date, user ratings, price, playtime, platform availability, ownership estimates

## Technologies Used
Python 3.10+

pandas

matplotlib

seaborn

scikit-learn

Jupyter Notebook & PyCharm IDE

## Key EDA Analyses
Distribution of game prices

Top 10 most common genres

Trend of game releases by year

Correlation between user ratings and price

Player rating analysis (positive vs. negative)

Estimated owners per game

Value-for-money ranking based on ratings and playtime

 The dataset was cleaned to handle missing values and formatting issues.
 All plots include clear labels and titles.
 Insights were drawn directly from visualizations.

## Advanced Unsupervised Analysis
To go beyond descriptive statistics, I applied unsupervised machine learning to identify hidden structures in the data.

## Principal Component Analysis (PCA)
Reduced dimensionality of numeric features

Helped uncover two latent components:

PC1 – Game Popularity (ratings + player base)

PC2 – Player Engagement (average & median playtime)

Showed that most games tend to perform well on one dimension (either popular or engaging), but rarely both

## KMeans Clustering (on PCA-transformed data)
Revealed 3 behavioral clusters:

Cluster	Profile	Description
0	Standard Games	Moderate popularity and pricing. Candidates for seasonal promotions.
1	Popular & Cheap	Massively owned, low-priced titles with high engagement. Perfect for freemium models.
2	Premium Niche	High-priced games with smaller, loyal audiences. Ideal for premium positioning.

 These clusters offer actionable insights for targeted marketing, bundling strategies, and product positioning.

## Predicting High Player Engagement (Supervised Learning)
To simulate a real-world business case, I developed a binary classification model to predict whether a game is likely to generate high player engagement.

 Problem Definition
Target: High engagement = average_playtime > 1000 minutes
Features: positive_ratings, negative_ratings, price, owners_min
Model: Logistic Regression (with class_weight='balanced' due to class imbalance)

## ML Pipeline
Data cleaning and feature engineering
Standardization of features
Train-test split (80/20)
Model training and evaluation (confusion matrix, classification report, accuracy)

 ##Results
The balanced logistic regression model successfully identified patterns contributing to high engagement.

positive_ratings and negative_ratings were the strongest predictors, followed by price and owners_min.

## Key Takeaways
EDA uncovers trends in pricing, genres, and player behavior

PCA + KMeans identify natural game groupings with strategic business value

Logistic Regression enables prediction of user engagement from real-world signals
