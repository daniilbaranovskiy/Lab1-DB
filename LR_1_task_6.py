import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Завантаження даних з файлу
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X = data[:, :-1]  # Ознаки
y = data[:, -1]   # Мітки класів

# Розділення даних на навчальний і тестовий набори (80% - навчання, 20% - тестування)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація та навчання моделі машини опорних векторів (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Прогнозування класів на тестовому наборі за допомогою моделі SVM
svm_predictions = svm_model.predict(X_test)

# Оцінка якості моделі SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_classification_report = classification_report(y_test, svm_predictions)

# Ініціалізація та навчання Gaussian Naive Bayes класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозування класів на тестовому наборі за допомогою Gaussian Naive Bayes класифікатора
nb_predictions = nb_model.predict(X_test)

# Оцінка якості Gaussian Naive Bayes класифікатора
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_classification_report = classification_report(y_test, nb_predictions)

# Виведення результатів для обох моделей
print("Модель машини опорних векторів (SVM) - Оцінка якості:")
print(f"Accuracy: {svm_accuracy}")
print("Звіт про класифікацію:")
print(svm_classification_report)

print("\nGaussian Naive Bayes - Оцінка якості:")
print(f"Accuracy: {nb_accuracy}")
print("Звіт про класифікацію:")
print(nb_classification_report)

# Порівняння моделей та висновки
if svm_accuracy > nb_accuracy:
    print("\nМодель машини опорних векторів (SVM) краще за точністю.")
elif svm_accuracy < nb_accuracy:
    print("\nGaussian Naive Bayes краще за точністю.")
else:
    print("\nОбидві моделі мають однакову точність.")
