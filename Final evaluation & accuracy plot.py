y_pred = np.argmax(global_model.predict(X_test), axis=1)
final_acc = accuracy_score(y_test, y_pred)
print("\nFINAL GLOBAL ACCURACY:", final_acc)

# ============================================================
# Accuracy Plot
# ============================================================

import matplotlib.pyplot as plt
plt.plot(global_accuracy_history, marker='o')
plt.title("Global Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
