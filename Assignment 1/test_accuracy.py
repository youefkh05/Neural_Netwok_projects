from check_accuracy import check_accuracy
import numpy as np

# Generate random predictions (0–9)
y_random = np.random.randint(0, 10, size=10000)

# Run accuracy
acc, correct, total = check_accuracy(y_random)

print("Accuracy:", acc)
print("Correct:", correct)
print("Total:", total)