import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Генерация простого датасета
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Создание пайплайна
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Масштабирование признаков
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=30, max_depth=15))
])

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Сохранение модели
with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Модель тренирована и сохранена.")