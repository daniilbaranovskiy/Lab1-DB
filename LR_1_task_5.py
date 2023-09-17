import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, \
    roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5

df['predicted_RF'] = (df['model_RF'] >= thresh).astype(int)
df['predicted_LR'] = (df['model_LR'] >= thresh).astype(int)

df.head()

confusion_matrix_RF = confusion_matrix(df['actual_label'], df['predicted_RF'])
confusion_matrix_LR = confusion_matrix(df['actual_label'], df['predicted_LR'])


def find_TP(y_true, y_pred):
    # Підраховує кількість True Positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # Підраховує кількість False Negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # Підраховує кількість False Positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # Підраховує кількість True Negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


def find_conf_matrix_values(y_true, y_pred):
    # Обчислює TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def baranovskiy_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


# Перевірка роботи моєї функції baranovskiy_confusion_matrix
print("Confusion matrix for RF:")
print(baranovskiy_confusion_matrix(df['actual_label'].values, df['predicted_RF'].values))

print("Confusion matrix for LR:")
print(baranovskiy_confusion_matrix(df['actual_label'].values, df['predicted_LR'].values))

# Порівняння результатів з вбудованою функцією confusion_matrix
print("Confusion matrix for RF (sklearn):")
print(confusion_matrix_RF)

print("Confusion matrix for LR (sklearn):")
print(confusion_matrix_LR)

# Використання accuracy_score
accuracy_RF = accuracy_score(df['actual_label'], df['predicted_RF'])
accuracy_LR = accuracy_score(df['actual_label'], df['predicted_LR'])

print("Accuracy for RF:", accuracy_RF)
print("Accuracy for LR:", accuracy_LR)


def baranovskiy_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


# Перевірка роботи моєї функції baranovskiy_accuracy_score
assert baranovskiy_accuracy_score(df['actual_label'].values, df['predicted_RF'].values) == accuracy_score(
    df['actual_label'].values, df['predicted_RF'].values), 'my_accuracy_score failed on RF'
assert baranovskiy_accuracy_score(df['actual_label'].values, df['predicted_LR'].values) == accuracy_score(
    df['actual_label'].values, df['predicted_LR'].values), 'my_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (baranovskiy_accuracy_score(df['actual_label'].values, df['predicted_RF'].values)))
print('Accuracy LR: %.3f' % (baranovskiy_accuracy_score(df['actual_label'].values, df['predicted_LR'].values)))

recall_RF = recall_score(df['actual_label'], df['predicted_RF'])
recall_LR = recall_score(df['actual_label'], df['predicted_LR'])

print("Recall for RF:", recall_RF)
print("Recall for LR:", recall_LR)


def baranovskiy_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    recall = TP / (TP + FN)
    return recall


# Перевірка роботи моєї функції baranovskiy_recall_score
assert baranovskiy_recall_score(df['actual_label'].values, df['predicted_RF'].values) == recall_score(
    df['actual_label'].values, df['predicted_RF'].values), 'my_recall_score failed on RF'
assert baranovskiy_recall_score(df['actual_label'].values, df['predicted_LR'].values) == recall_score(
    df['actual_label'].values, df['predicted_LR'].values), 'my_recall_score failed on LR'

print('Recall RF: %.3f' % (baranovskiy_recall_score(df['actual_label'].values, df['predicted_RF'].values)))
print('Recall LR: %.3f' % (baranovskiy_recall_score(df['actual_label'].values, df['predicted_LR'].values)))

precision_RF = precision_score(df['actual_label'], df['predicted_RF'])
precision_LR = precision_score(df['actual_label'], df['predicted_LR'])

print("Precision for RF:", precision_RF)
print("Precision for LR:", precision_LR)


def baranovskiy_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    precision = TP / (TP + FP)
    return precision


# Перевірка роботи моєї функції baranovskiy_precision_score
assert baranovskiy_precision_score(df['actual_label'].values, df['predicted_RF'].values) == precision_score(
    df['actual_label'].values, df['predicted_RF'].values), 'my_precision_score failed on RF'
assert baranovskiy_precision_score(df['actual_label'].values, df['predicted_LR'].values) == precision_score(
    df['actual_label'].values, df['predicted_LR'].values), 'my_precision_score failed on LR'

print('Precision RF: %.3f' % (baranovskiy_precision_score(df['actual_label'].values, df['predicted_RF'].values)))
print('Precision LR: %.3f' % (baranovskiy_precision_score(df['actual_label'].values, df['predicted_LR'].values)))

f1_RF = f1_score(df['actual_label'], df['predicted_RF'])
f1_LR = f1_score(df['actual_label'], df['predicted_LR'])

print("F1 Score for RF:", f1_RF)
print("F1 Score for LR:", f1_LR)


def baranovskiy_f1_score(y_true, y_pred):
    precision = baranovskiy_precision_score(y_true, y_pred)
    recall = baranovskiy_recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# Перевірка роботи моєї функції baranovskiy_f1_score
assert baranovskiy_f1_score(df['actual_label'].values, df['predicted_RF'].values) == f1_score(df['actual_label'].values,
                                                                                              df[
                                                                                                  'predicted_RF'].values), 'my_f1_score failed on RF'
assert baranovskiy_f1_score(df['actual_label'].values, df['predicted_LR'].values) == f1_score(df['actual_label'].values,
                                                                                              df[
                                                                                                  'predicted_LR'].values), 'my_f1_score failed on LR'

print('F1 RF: %.3f' % (baranovskiy_f1_score(df['actual_label'].values, df['predicted_RF'].values)))
print('F1 LR: %.3f' % (baranovskiy_f1_score(df['actual_label'].values, df['predicted_LR'].values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (baranovskiy_accuracy_score(df['actual_label'].values, df['predicted_RF'].values)))
print('Recall RF: %.3f' % (baranovskiy_recall_score(df['actual_label'].values, df['predicted_RF'].values)))
print('Precision RF: %.3f' % (baranovskiy_precision_score(df['actual_label'].values, df['predicted_RF'].values)))
print('F1 RF: %.3f' % (baranovskiy_f1_score(df['actual_label'].values, df['predicted_RF'].values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (
    baranovskiy_accuracy_score(df['actual_label'].values, (df['model_RF'] >= 0.25).astype('int').values)))
print(
    'Recall RF: %.3f' % (
        baranovskiy_recall_score(df['actual_label'].values, (df['model_RF'] >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (
    baranovskiy_precision_score(df['actual_label'].values, (df['model_RF'] >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (baranovskiy_f1_score(df['actual_label'].values, (df['model_RF'] >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df['actual_label'].values, df['model_RF'].values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df['actual_label'].values, df['model_LR'].values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df['actual_label'].values, df['model_RF'].values)
auc_LR = roc_auc_score(df['actual_label'].values, df['model_LR'].values)

print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
