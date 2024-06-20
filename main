import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, mean_squared_error, roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data_set.csv'
data = pd.read_csv(file_path)

# Preprocess the data
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['education'] = le.fit_transform(data['education'])
data['profession'] = le.fit_transform(data['profession'])
data['workDuration'] = data['workDuration'].str.extract('(\d+)').astype(int)
data['typeSocial'] = le.fit_transform(data['typeSocial'])
data['useSocial'] = le.fit_transform(data['useSocial'])
data['productivity'] = le.fit_transform(data['productivity'])  # Target variable

# Define features and target
X = data.drop(columns=['id', 'productivity'])
y = data['productivity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate Pseudo R-squared (McFadden's R²)
ll_null = log_loss(y_test, [y_train.mean()] * len(y_test))  # Log-likelihood of null model
ll_model = log_loss(y_test, y_pred_proba)  # Log-likelihood of the model
pseudo_r2 = 1 - (ll_model / ll_null)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Pseudo R-squared: {pseudo_r2}')
print(f'RMSE: {rmse}')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
