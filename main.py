
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os


df = pd.read_csv("creditcard.csv")
print("Размер датасета:", df.shape)
print("\nПервые строки:")
print(df.head())

# Проверка пропусков
print("\nПропущенные значения:")
print(df.isnull().sum().sum())

# Распределение классов
print("\nРаспределение классов:")
print(df["Class"].value_counts())

print("\nДоля классов:")
print(df["Class"].value_counts(normalize=True) * 100)

# Разделение признаков и целевой переменной
X = df.drop("Class", axis=1)
y = df["Class"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=df.Class,
    random_state=42
)

print("\nРазмер тренировочной выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

print("\nКлассы в тренировочной выборке:")
print(y_train.value_counts())

print("\nКлассы в тестовой выборке:")
print(y_test.value_counts())


# Папка для сохранения моделей
os.makedirs("saved_models", exist_ok=True)

model_files = {
    "Bagging": "saved_models/bagging.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "LightGBM": "saved_models/lightgbm.pkl",
    "CatBoost": "saved_models/catboost.pkl"
}


# Проверка наличия сохраненных моделей
if all(os.path.exists(path) for path in model_files.values()):

    print("\nЗагрузка сохраненных моделей...")

    best_models = {}

    for name, path in model_files.items():
        best_models[name] = joblib.load(path)
        print(f"{name} успешно загружена!")

else:

    print("\nСохраненные модели не найдены. Начинаем обучение...")


    # 4 модели
    models = {
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=100,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
            allow_writing_files=False
        )
    }

    # Обучение и оценка моделей
    results = {}

    for name, model in models.items():
        print(f"\nОбучение модели: {name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results[name] = score

        print(f"Macro F1: {score:.5f}")

    # Сравнение моделей
    print("\nРезультаты моделей:")

    for name, score in sorted(
        results.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name}: {score:.5f}")


    # Пространства гиперпараметров

    # Bagging
    bagging_params = {
        "n_estimators": [50, 100, 150],
        "max_samples": [0.7, 0.9, 1.0],
        "max_features": [0.7, 1.0],
        "estimator__max_depth": [5, 10, 15, None],
        "estimator__min_samples_leaf": [1, 2, 5]
    }

    # Random Forest
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"]
    }

    # LightGBM
    lgbm_params = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 50],
        "max_depth": [-1, 5, 10, 15],
        "min_child_samples": [10, 20, 40],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "subsample_freq": [1],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0, 1, 5, 10]
    }

    # CatBoost
    catboost_params = {
        "iterations": [200, 300, 500],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
        "random_strength": [0.1, 0.5, 1, 2]
    }


    # Модели для оптимизации
    search_models = {
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                random_state=42,
                n_jobs=-1
            ),
            bagging_params
        ),

        "Random Forest": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            rf_params
        ),

        "LightGBM": (
            LGBMClassifier(
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ),
            lgbm_params
        ),

        "CatBoost": (
            CatBoostClassifier(
                auto_class_weights="Balanced",
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
                thread_count=1
            ),
            catboost_params
        )
    }


    # Кросс-валидация
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    best_models = {}
    best_scores = {}

    for name, (model, params) in search_models.items():

        print(f"\nОптимизация: {name}")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=5,
            scoring="f1_macro",
            cv=cv,
            random_state=42,
            n_jobs=1 if name == "CatBoost" else -1,
            verbose=1
        )

        search.fit(X_train, y_train)

        best_models[name] = search.best_estimator_
        best_scores[name] = search.best_score_

        print("Лучший Macro F1:", search.best_score_)
        print("Лучшие параметры:")
        print(search.best_params_)

    print("\nРезультаты оптимизации:")

    for name, score in sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name}: {score:.5f}")


    # Сохранение обученных моделей
    print("\nСохранение моделей...")

    for name, model in best_models.items():
        joblib.dump(model, model_files[name])
        print(f"{name} сохранена!")


# Оценка оптимизированных моделей на тестовой выборке
test_results = {}
test_predictions = {}

for name, model in best_models.items():

    predictions = model.predict(X_test)

    score = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    test_results[name] = score
    test_predictions[name] = predictions

    print(f"\n{name}")
    print(f"Test Macro F1: {score:.5f}")


# Выбираем модель с максимальным результатом
best_name = max(test_results, key=test_results.get)
best_score = test_results[best_name]
best_model = best_models[best_name]
best_prediction = test_predictions[best_name]

print("\nЛучшая модель:", best_name)
print("Лучший Test Macro F1:", best_score)


# Отчет о классификации
print("\nОтчет о классификации:")
print(classification_report(
    y_test,
    best_prediction,
    target_names=["Легальная", "Мошенническая"]
))


# Построение матрицы ошибок
cm = confusion_matrix(
    y_test,
    best_prediction,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Легальная", "Мошенническая"],
    yticklabels=["Легальная", "Мошенническая"]
)

plt.title(f"Матрица ошибок: {best_name}")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
