# after using PCA and KMeans to explore the natural structure of the data,
# it should predict whether a game has high player engagement (defined as average_playtime > 1000 min).
# this will be framed as a binary classification task using features such as:
# positive/negative ratings, price, and estimated number of players.
# the goal is to understand which features contribute most to engagement
# and to simulate how such a prediction could assist in game development or marketing.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv('steam.csv')
df['owners_min']=df['owners'].str.split('-').str[0].astype(int)
df = df[df['average_playtime'] > 0]

# binary target 1 if average time > 1000, else 0
df['high_engagement']=df['average_playtime'].apply(lambda x: 1 if x > 1000 else 0)

# input variables and target
features=['positive_ratings', 'negative_ratings', 'price', 'owners_min']
x=df[features]
y=df['high_engagement']

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model=LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print('classification report:')
print(classification_report(y_test, y_pred))

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))

print('accuracy score:')
print(accuracy_score(y_test, y_pred))



coeffs = pd.Series(model.coef_[0], index=features)
print(coeffs.sort_values(ascending=False))


coeffs.sort_values().plot(kind='barh', title='Feature Influence on High Engagement')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.show()

# the plot shows how each feature influences the likelihood that a game has high engagement
# a higher coefficient indicates a stronger contribution to predicting a high engagement label (1)
# games with a high number of negative ratings still tend to have high engagement (large coefficient for 'negative_ratings')
# 'positive_ratings' is also a strong predictor of player engagement
# 'price' and 'owners_min' have a weaker, but still positive, impact.
# so both types of ratings are the most influential factors in predicting high player engagement on Steam.