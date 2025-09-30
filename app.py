# 1. Import all libraries together
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load data and view shape, head, info
df = pd.read_csv("train.csv")

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo about dataset:\n")
print(df.info())
print("\nClass distribution:\n", df['class'].value_counts())

# 3. Encode the categorical columns
categorical_cols = ['protocol_type', 'service', 'flag', 'class']
encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 4. Train-test split
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Define both models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 6. Fit the data and get accuracy

# Random Forest
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", rf_acc)
print("Random Forest Report:")
print(classification_report(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))

# Gradient Boosting
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_preds)
print("Gradient Boosting Accuracy:", gb_acc)
print("Gradient Boosting Report:")
print(classification_report(y_test, gb_preds))
print(confusion_matrix(y_test, gb_preds))