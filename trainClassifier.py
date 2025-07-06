# Modified trainClassifier.py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=15)
model.fit(x_train, y_train)

# Predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for ASL Gesture Recognition')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
