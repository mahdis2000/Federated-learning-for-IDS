NUM_CLIENTS = 8
ALPHA = 0.5   # non-iid level

num_classes = len(np.unique(y_encoded))

client_indices = [[] for _ in range(NUM_CLIENTS)]

# ---- Dirichlet distribution ----
for class_id in range(num_classes):

    idx_class = np.where(y_encoded == class_id)[0]
    np.random.shuffle(idx_class)

    proportions = np.random.dirichlet(
        alpha=[ALPHA]*NUM_CLIENTS
    )

    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]

    split_indices = np.split(idx_class, proportions)

    for client_id in range(NUM_CLIENTS):
        client_indices[client_id].extend(split_indices[client_id])

# ---- Build clients ----
non_iid_clients = {}

for i in range(NUM_CLIENTS):

    indices = client_indices[i]

    X_c = X_scaled[indices]
    y_c = y_encoded[indices]

    non_iid_clients[f'client_{i+1}'] = (X_c, y_c)

    print(f"client_{i+1} samples:", len(indices))

print("\n✅ Non-IID Clients Created Successfully")
