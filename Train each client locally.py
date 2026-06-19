client_results = {}

for cname, (Xc, yc) in non_iid_clients.items():

    print(f"\nTraining {cname} using {client_models[cname]}")

    model = create_model(client_models[cname])

    start = time.time()

    history = model.fit(
        Xc, yc,
        epochs=5,
        batch_size=64,
        verbose=0
    )

    train_time = time.time() - start

    # evaluation
    y_pred = np.argmax(model.predict(Xc, verbose=0), axis=1)

    acc = accuracy_score(yc, y_pred)
    precision = precision_score(yc, y_pred, average='weighted')
    recall = recall_score(yc, y_pred, average='weighted')
    f1 = f1_score(yc, y_pred, average='weighted')

    client_results[cname] = {
        "model": client_models[cname],
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time": train_time
    }
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
