import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

base_dir = 'output_lexical_metrics'
mmse_path = 'mmse_input.csv'

# mostly pre-processing pipeline to replicate with different labs' datasets

print(f"Checking data directory: {base_dir}")
for subgroup in ['cc', 'cd']:
    path = os.path.join(base_dir, subgroup)
    print(f" - Subfolder '{path}': {'FOUND' if os.path.isdir(path) else 'MISSING'}")

print("\nLoading lexical metrics CSVs...")

# load lexical metrics (1 row per csv) and label from folder name
filenames, labels, rows = [], [], []
for label in ['cc', 'cd']:
    folder = os.path.join(base_dir, label)
    files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    print(f" * {len(files)} in '{label}'")
    for fname in files:
        full_path = os.path.join(folder, fname)
        rows.append(pd.read_csv(full_path).iloc[0])
        labels.append(label)
        filenames.append(os.path.splitext(fname)[0])

lex_df = pd.DataFrame(rows).assign(label=labels, filename=filenames)
print(f"Total lexical samples: {len(lex_df)}")

print(f"\nLoading MMSE scores from: {mmse_path}")
mmse_df = pd.read_csv(mmse_path)
print("Original MMSE columns:", mmse_df.columns.tolist())

if 'filename' not in mmse_df.columns:
    print("MMSE file must have a 'filename' column.")
if 'mmse_score' not in mmse_df.columns:
    print("MMSE file must have a 'mmse_score' column.")

mmse_df['filename'] = mmse_df['filename'].astype(str).apply(lambda x: os.path.splitext(x)[0])

print("MMSE data after processing filenames:")
print(mmse_df.head())

# combine lexical metrics files w/ mmse data
merged = lex_df.merge(mmse_df[['filename', 'mmse_score']], on='filename', how='left')
print(f"\nAfter merge, data shape: {merged.shape}")
print(f"MMSE missing for {merged['mmse_score'].isna().sum()} samples")

# build X/y
y = merged['label'].map({'cc': 0, 'cd': 1})
X = merged.drop(columns=['label', 'filename'])

# some of the patients did not have all the values for every featrure, and in that case we sub'd w the average
print("\nImputing missing values via column means...")
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
print("Missing after imputation:", X.isna().sum().sum())

#training the model
print("\nSplitting data (80% train / 20% test)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Standardizing features…")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nInitializing and training MLPClassifier…")
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    verbose=True,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# evaluate on the test set
print("\nEvaluating on test set…")
y_pred = mlp.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nFinal training loss: {mlp.loss_curve_[-1]:.4f}")
