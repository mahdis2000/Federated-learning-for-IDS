MODEL_NAME = "MLP"   # change anytime

ROUNDS = 5
LOCAL_EPOCHS = 2

global_model = build_model(
    MODEL_NAME,
    input_shape=X_train.shape[1:],
    num_classes=num_classes
)

global_accuracy_history = []

for r in range(ROUNDS):

    print(f"\n========== ROUND {r+1} ==========")

    local_weights = []

    for client,(Xc,yc) in clients_data.items():

        print(client,"training...")

        local_model = build_model(
            MODEL_NAME,
            X_train.shape[1:],
            num_classes
        )

        local_model.set_weights(global_model.get_weights())

        local_model.fit(
            Xc, yc,
            epochs=LOCAL_EPOCHS,
            batch_size=64,
            verbose=0
        )

        local_weights.append(local_model.get_weights())

    # aggregation
    new_weights = federated_average(local_weights)
    global_model.set_weights(new_weights)

    loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
    global_accuracy_history.append(acc)

    print("Global Accuracy:", acc)
