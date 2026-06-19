X = df[feature_cols].values
y = df[target_col].astype(str).values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", le.classes_)
# reshape for sequence models
X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
