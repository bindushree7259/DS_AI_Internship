from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Actual values
y_actual = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

# Predicted values
y_predicted = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]

# -------------------------------
# 1. Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_actual, y_predicted)

print("Confusion Matrix:")
print(cm)

# Extract values from confusion matrix
TN, FP, FN, TP = cm.ravel()

print("\nTrue Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print("True Positive (TP):", TP)


# -------------------------------
# 2. Accuracy
# -------------------------------

accuracy = accuracy_score(y_actual, y_predicted)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# -------------------------------
# 3. Precision
# -------------------------------

precision = precision_score(y_actual, y_predicted)

print("\nPrecision:", precision)
print("Precision Percentage:", precision * 100, "%")


# -------------------------------
# 4. Recall
# -------------------------------

recall = recall_score(y_actual, y_predicted)

print("\nRecall:", recall)
print("Recall Percentage:", recall * 100, "%")


# -------------------------------
# 5. Insights
# -------------------------------

print("\n========== INSIGHTS ==========")

if accuracy >= 0.80:
    print("✓ Accuracy is good. The model correctly predicts most values.")
else:
    print("✗ Accuracy is low. The model needs improvement.")

if precision >= 0.80:
    print("✓ Precision is good. Most positive predictions are actually positive.")
else:
    print("✗ Precision is low. The model produces more false positive predictions.")

if recall >= 0.80:
    print("✓ Recall is good. The model identifies most of the actual positive cases.")
else:
    print("✗ Recall is low. The model is missing some actual positive cases.")

print("\nOverall Insight:")
print("The model has 80% accuracy, 80% precision, and 80% recall.")
print("This means the model performs reasonably well in identifying positive and negative cases.")