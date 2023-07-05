#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('CAR DETAILS.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df.info()


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[10]:


###Performing EDA on the given data 


# In[11]:


print(df[["year", "selling_price", "km_driven"]].describe())


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


# Bar plot of "owner" column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="owner")
plt.title("Distribution of Owners")
plt.xlabel("Owner Type")
plt.ylabel("Count")
plt.show()


# In[14]:


# Histogram of "km_driven" column
plt.figure(figsize=(8, 6))
sns.histplot(df["km_driven"], bins=20, kde=True)
plt.title("Distribution of Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.ylabel("Count")
plt.show()


# In[15]:


# Histogram of "year" column
plt.figure(figsize=(8, 6))
sns.histplot(df["year"], bins=10, kde=True)
plt.title("Distribution of Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


# In[16]:


# Histogram of "selling_price" column
plt.figure(figsize=(8, 6))
sns.histplot(df["selling_price"], bins=20, kde=True)
plt.title("Distribution of Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.show()


# In[17]:


# Bar plot of "fuel" column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="fuel")
plt.title("Distribution of Fuel Types")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.show()


# In[18]:


# Bar plot of "seller_type" column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="seller_type")
plt.title("Distribution of Seller Types")
plt.xlabel("Seller Type")
plt.ylabel("Count")
plt.show()


# In[19]:


# Bar plot of "transmission" column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="transmission")
plt.title("Distribution of Transmission Types")
plt.xlabel("Transmission Type")
plt.ylabel("Count")
plt.show()


# In[20]:


# Create a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="year", y="selling_price")
plt.title("Correlation between Selling Price and Year")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.show()


# In[21]:


plt.scatter(df["km_driven"], df["selling_price"])
plt.title("Selling Price vs. Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.show()


# In[22]:


# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(df["transmission"],df["selling_price"])
plt.title("Selling Price by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Average Selling Price")
plt.show()


# In[23]:


# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(df["fuel"], df["selling_price"])
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price")
plt.show()


# In[24]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[25]:


categorical_columns = ["name","fuel", "seller_type", "transmission", "owner"]
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


# In[26]:


print(df.head())


# In[27]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[28]:


# Separate the features (X) and target variable (y)
X = df.drop("selling_price", axis=1)
y = df["selling_price"]


# In[29]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


reg = LinearRegression()


# In[31]:


reg.fit(X_train, y_train)


# In[32]:


y_pred = reg.predict(X_test)


# In[33]:


print("MSE => ", mean_squared_error(y_test, y_pred))
print("R2 Score => ",r2_score(y_test, y_pred))


# In[34]:


las = Lasso()


# In[35]:


las.fit(X_train,y_train)


# In[36]:


y_pred = las.predict(X_test)
print("MSE => ", mean_squared_error(y_test, y_pred))
print("R2 Score => ",r2_score(y_test, y_pred))


# In[37]:


rid = Ridge()


# In[38]:


rid.fit(X_train,y_train)


# In[39]:


y_pred = rid.predict(X_test)
print("MSE => ", mean_squared_error(y_test, y_pred))
print("R2 Score => ",r2_score(y_test, y_pred))


# In[40]:


data = [['Linear', 184332080354.485,0.39596976258883987], ['Lasso',184331821263.43762,0.39597061159367153], ['Ridge', 184274152942.32608, 0.39615958255100636]]
summary = pd.DataFrame(data, columns=['Model','MSE', 'R2 Score'])
summary


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier


# In[42]:


def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, cm, report


# In[43]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[44]:


logreg_accuracy, logreg_cm, logreg_report = evaluate_model(logreg, X_test, y_test)


# In[45]:


print("Logistic Regression")
print("Accuracy:", logreg_accuracy)
print("Confusion Matrix:")
print(logreg_cm)
print("Classification Report:")
print(logreg_report)
print("\n")


# In[46]:


dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)


# In[47]:


dt_accuracy, dt_cm, dt_report = evaluate_model(dt_classifier, X_test, y_test)


# In[48]:


print("Decision Tree Classifier")
print("Accuracy:", dt_accuracy)
print("Confusion Matrix:")
print(dt_cm)
print("Classification Report:")
print(dt_report)
print("\n")


# In[49]:


rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)


# In[50]:


rf_accuracy, rf_cm, rf_report = evaluate_model(rf_classifier, X_test, y_test)


# In[51]:


print("Random Forest Classifier")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:")
print(rf_cm)
print("Classification Report:")
print(rf_report)
print("\n")


# In[52]:


knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)


# In[53]:


knn_accuracy, knn_cm, knn_report = evaluate_model(knn_classifier, X_test, y_test)


# In[54]:


print("K-Nearest Neighbors (KNN) Classifier")
print("Accuracy:", knn_accuracy)
print("Confusion Matrix:")
print(knn_cm)
print("Classification Report:")
print(knn_report)
print("\n")


# In[55]:


bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)


# In[56]:


y_pred = bagging.predict(X_test)


# In[57]:


mse = mean_squared_error(y_test, y_pred)

print("Bagging Regression")
print("Mean Squared Error:", mse)


# In[58]:


bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)


# In[59]:


y_pred = bagging.predict(X_test)


# In[60]:


accuracy = accuracy_score(y_test, y_pred)

print("Bagging Classification")
print("Accuracy:", accuracy)


# In[77]:


adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
adaboost.fit(X_train, y_train)


# In[78]:


y_pred = adaboost.predict(X_test)


# In[79]:


mse = mean_squared_error(y_test, y_pred)

print("AdaBoost Regression")
print("Mean Squared Error:", mse)


# In[80]:


adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
adaboost.fit(X_train, y_train)


# In[81]:


y_pred = adaboost.predict(X_test)


# In[82]:


print("AdaBoost Classification")
print("Accuracy:", accuracy)


# In[67]:


import pickle


# In[88]:


pickle.dump(rf_classifier,open('rf_classifier.pkl','wb'))


# In[89]:


pickle.dump(dt_classifier, open('dt_classifier.pkl', 'wb'))


# In[93]:


new_dataset = df.sample(n=20, random_state=42)


# In[94]:


new_dataset.head()


# In[95]:


loaded_model = pickle.load(open('dt_classifier.pkl', 'rb'))


# In[96]:


X_new = new_dataset.drop("selling_price", axis=1)


# In[98]:


y_pred = loaded_model.predict(X_new)


# In[99]:


y_actual = new_dataset["selling_price"]


# In[100]:


print("Predicted Values:\n", y_pred)
print("Actual Values:\n", y_actual)

