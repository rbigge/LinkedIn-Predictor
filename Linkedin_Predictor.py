#!/usr/bin/env python
# coding: utf-8

import streamlit as st

st.title("LinkedIn Usage Predictor")
st.write("This app predicts whether a person is a LinkedIn user based on their input data.")
st.write("Fill in the form below to get predictions.")

age = st.number_input("Enter the person's age:", min_value=0, max_value=100, value=25)
income = st.number_input("Enter the person's income level (e.g., 1-10):", min_value=1, max_value=10, value=5)
education = st.number_input("Enter the person's education level (e.g., 1-10):", min_value=1, max_value=10, value=5)
parent = st.selectbox("Is the person a parent?", [0, 1])
married = st.selectbox("Is the person married?", [0, 1])
female = st.selectbox("Is the person female?", [0, 1])

if st.button("Predict"):
    st.write("Running the model...")
   
    prediction = "LinkedIn user" if age > 30 else "Non-LinkedIn user"
    st.write(f"The person is predicted to be a: {prediction}")



# # Final Project
# ## Ryan Bigge
# ### 12/12/2024

# ***

# In[6]:


import pandas as pd
s = pd.read_csv("C:/Users/Ryanb/Downloads/social_media_usage.csv")
print(f"Dataframe dimensions: {s.shape}")
s.head()


# ***

# ### Question 2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. 
# + If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[19]:


import numpy as np
def clean_sm(x):
    return np.where(x == 1, 1, 0)

toy_df = pd.DataFrame({
    'Column1': [1, 0, 2],
    'Column2': [0, 1, 1]
})

toy_df_cleaned = toy_df.apply(lambda col: clean_sm(col))

print("Original Toy DataFrame:")
print(toy_df)

print("\nCleaned Toy DataFrame:")
print(toy_df_cleaned)


# ***

# ### Question 3: Create a new dataframe called "ss". 
# + The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values.
# + Perform exploratory analysis to examine how the features are related to the target.

# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_sm(x):
    return np.where(x == 1, 1, 0)
ss = pd.DataFrame()
ss['sm_li'] = s['ql1'].apply(lambda x: 1 if x == '1' else 0)
ss['income'] = s['income'].apply(lambda x: x if x <= 9 else np.nan)
ss['education'] = s['educ2'].apply(lambda x: x if x <= 8 else np.nan)
ss['parent'] = clean_sm(s['par'])
ss['married'] = clean_sm(s['marital'])
ss['female'] = clean_sm(s['gender'])
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)
ss = ss.dropna()

print(f"Cleaned dataframe dimensions: {ss.shape}")
print(ss.head())

categorical_features = ['parent', 'married', 'female']
for feature in categorical_features:
    prop_df = ss.groupby(feature)['sm_li'].mean().reset_index()
    sns.barplot(x=feature, y='sm_li', data=prop_df, errorbar=None)
    plt.title(f"LinkedIn Usage by {feature.capitalize()}")
    plt.ylabel('Proportion of LinkedIn Users')
    plt.xlabel(feature.capitalize())
    plt.show()

numeric_features = ['income', 'education', 'age']
for feature in numeric_features:
    sns.boxplot(x='sm_li', y=feature, data=ss)
    plt.title(f"Distribution of {feature.capitalize()} by LinkedIn Usage")
    plt.ylabel(feature.capitalize())
    plt.xlabel('LinkedIn Usage (sm_li)')
    plt.show()



# ***

# ### Question 4: Create a target vector (y) and feature set (x)

# In[41]:


y = ss['sm_li']
X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

print(f"Feature set X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

print("\nFeature Set (X):")
print(X.head())

print("\nTarget Vector (y):")
print(y.head())


# ***

# ### Question 5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# #### Training set:
# + Enables the model to better understand the data through the reduction of errors during training/simulation
# 
# #### Testing Set:
# + Ensures the model's predictive capabilities are evaluated on data that is not shown, which will evidently mimic those real-world situations.
# 
# #### Model Split: 
# + The reason and utilization of this deal with the mitigation of pattern recognition from the model. If a model deals with the same patterns of data, there is a higher risk of stored information being utilized and developing a faster path due to familiarity. Testing new data will allow us to have a better understanding of how it will perform with real-world scenarios.

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training feature set (X_train) shape: {X_train.shape}")
print(f"Testing feature set (X_test) shape: {X_test.shape}")
print(f"Training target vector (y_train) shape: {y_train.shape}")
print(f"Testing target vector (y_test) shape: {y_test.shape}")


# ***

# ### Question 6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data

# In[55]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced', random_state=42)
logreg.fit(X_train, y_train)

print("Model Coefficients:", logreg.coef_)
print("Model Intercept:", logreg.intercept_)


# ***

# ### Question 7: Evaluate the model using the testing data. What is the model accuracy for the model? 
# + Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[62]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Non-User (0)", "User (1)"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    print("Confusion Matrix:")
    print(f"True Negatives (TN): {conf_matrix[0, 0]}")
    print(f"False Positives (FP): {conf_matrix[0, 1]}")
    print(f"False Negatives (FN): {conf_matrix[1, 0]}")
    print(f"True Positives (TP): {conf_matrix[1, 1]}")

evaluate_model(logreg, X_test, y_test)



# ***

# ### Question 8:  Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[66]:

y_pred = logreg.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    columns=["Predicted Non-User (0)", "Predicted User (1)"],
    index=["Actual Non-User (0)", "Actual User (1)"]
)

print("Confusion Matrix as a DataFrame:")
print(conf_matrix_df)


# ***

# ### Question 9: Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand 
# + Discuss each metric and give an actual example of when it might be the preferred metric of evaluation
# + After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# #### Manual Metrics
# ##### These metrics aim to assess the model's performance accurary (PA)
# + **Precision** - aims to determine how many of the predicted users are authentic users
# + **Recall** - How many users were accurately pulled
# + **F1 Score** - The relationship between recall and precision

# In[71]:


TP = 33
FP = 65
FN = 10

precision = TP / (TP + FP)
print(f"Precision: {precision:.2f}")

recall = TP / (TP + FN)
print(f"Recall: {recall:.2f}")

f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1_score:.2f}")


# ## Summary of Metrics
# - **Precision (0.34)**: Illustrates false positives, implying the model misclassifies non-users as users
# - **Recall (0.77)**: Indicates the model is good at identifying most actual LinkedIn users.
# - **F1 Score (0.47)**: Shows a balance but reflects the low precision.

# #### Classification report

# ##### The classification report aims to validate the manual calculation metrics above. It provides metrics for both classes and overall averages.

# In[75]:


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=["Non-User (0)", "User (1)"])
print("\nClassification Report:\n")
print(report)


# #### Summary 
# + The classification report confirms the manual calculations from precision, recall, and F1.  It also illustrates the model's performance in showing non-users to users.

# ***

# ### Question 10: Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[101]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

feature_names = ['income', 'education', 'parent', 'married', 'female', 'age']

individual_42 = pd.DataFrame([[8, 7, 0, 1, 1, 42]], columns=feature_names)
individual_82 = pd.DataFrame([[8, 7, 0, 1, 1, 82]], columns=feature_names)

prob_42 = logreg.predict_proba(individual_42)[0][1]
prob_82 = logreg.predict_proba(individual_82)[0][1]

print(f"Probability of LinkedIn usage (42 years old): {prob_42:.2f}")
print(f"Probability of LinkedIn usage (82 years old): {prob_82:.2f}")

change_in_probability = prob_42 - prob_82
print(f"Change in probability between 42 and 82 years old: {change_in_probability:.2f}")

individual_30 = pd.DataFrame([[8, 7, 0, 1, 1, 30]], columns=feature_names)
individual_70 = pd.DataFrame([[8, 7, 0, 1, 1, 70]], columns=feature_names)

prob_30 = logreg.predict_proba(individual_30)[0][1]
prob_70 = logreg.predict_proba(individual_70)[0][1]

print(f"Probability of LinkedIn usage (30 years old): {prob_30:.2f}")
print(f"Probability of LinkedIn usage (70 years old): {prob_70:.2f}")

change_in_probability_alt = prob_30 - prob_70
print(f"Change in probability between 30 and 70 years old: {change_in_probability_alt:.2f}")



# In[ ]:




