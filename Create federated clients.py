NUM_CLIENTS = 8
def create_clients(X, y, num_clients):

    data = list(zip(X, y))
    np.random.shuffle(data)

    shard_size = len(data)//num_clients
    clients = {}
    for i in range(num_clients):

        shard = data[i*shard_size:(i+1)*shard_size]

        X_c = np.array([x for x,_ in shard])
        y_c = np.array([y for _,y in shard])

        clients[f'client_{i+1}'] = (X_c, y_c)

    return clients

clients_data = create_clients(X_train, y_train, NUM_CLIENTS)
print("Clients Created:", len(clients_data))
