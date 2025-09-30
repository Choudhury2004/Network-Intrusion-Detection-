# 1. Import all libraries together
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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

# 4. Features and target
X = df.drop('class', axis=1)
y = df['class']

# 5. Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 6. K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, name):
    fold = 1
    acc_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_scores.append(acc)
        
        print(f"\n{name} - Fold {fold} Accuracy: {acc}")
        print(f"{name} - Fold {fold} Classification Report:")
        print(classification_report(y_test, preds))
        print(f"{name} - Fold {fold} Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        fold += 1
    
    print(f"\n{name} Average Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")

# 7. Run evaluations
evaluate_model(rf_model, X, y, "Random Forest")
evaluate_model(gb_model, X, y, "Gradient Boosting")
