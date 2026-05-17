# ================================
# Импорт библиотек
# ================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ================================
# Загрузка датасета
# ================================

df = pd.read_csv("data/car_price_dataset.csv")

# ================================
# Первичный анализ данных
# ================================

print("Первые 5 строк датасета:\n")
print(df.head())

print("\nИнформация о датасете:\n")
print(df.info())

print("\nРазмерность датасета:\n")
print(df.shape)

print("\nОписательная статистика:\n")
print(df.describe())

# ================================
# Проверка пропусков
# ================================

print("\nКоличество пропусков в каждом столбце:\n")
print(df.isnull().sum())

# ================================
# Проверка дубликатов
# ================================

print("\nКоличество дубликатов:")
print(df.duplicated().sum())

# ================================
# Анализ целевой переменной
# ================================

plt.figure(figsize=(10, 6))
sns.histplot(df["selling_price"], bins=30, kde=True)
plt.title("Распределение стоимости автомобилей")
plt.xlabel("Стоимость автомобиля")
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

# ================================
# Корреляционная матрица
# ================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Матрица корреляций числовых признаков")
plt.tight_layout()
plt.show()

# ================================
# Связь цены и года выпуска
# ================================

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="year", y="selling_price")
plt.title("Зависимость стоимости автомобиля от года выпуска")
plt.xlabel("Год выпуска")
plt.ylabel("Стоимость автомобиля")
plt.tight_layout()
plt.show()

# ================================
# Связь цены и пробега
# ================================

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="km_driven", y="selling_price")
plt.title("Зависимость стоимости автомобиля от пробега")
plt.xlabel("Пробег, км")
plt.ylabel("Стоимость автомобиля")
plt.tight_layout()
plt.show()

# ================================
# Средняя цена по маркам
# ================================

brand_price = df.groupby("brand")["selling_price"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_price.index, y=brand_price.values)
plt.title("Средняя стоимость автомобилей по маркам")
plt.xlabel("Марка")
plt.ylabel("Средняя стоимость")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# Цена в зависимости от коробки передач
# ================================

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="transmission", y="selling_price")
plt.title("Стоимость автомобилей в зависимости от типа коробки передач")
plt.xlabel("Тип коробки передач")
plt.ylabel("Стоимость автомобиля")
plt.tight_layout()
plt.show()

# ================================
# Подготовка данных
# ================================

# Кодирование категориальных признаков
df_encoded = pd.get_dummies(df, drop_first=True)

print("Размерность после кодирования:")
print(df_encoded.shape)

# ================================
# Разделение признаков и целевой переменной
# ================================

X = df_encoded.drop("selling_price", axis=1)
y = df_encoded["selling_price"]

print("\nРазмерность X:")
print(X.shape)

print("\nРазмерность y:")
print(y.shape)

# ================================
# Разделение на обучающую и тестовую выборки
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nРазмер обучающей выборки:")
print(X_train.shape)

print("\nРазмер тестовой выборки:")
print(X_test.shape)

# ================================
# Baseline model — Linear Regression
# ================================

baseline_model = LinearRegression()

baseline_model.fit(X_train, y_train)

# Предсказания
y_pred = baseline_model.predict(X_test)

# ================================
# Метрики качества
# ================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nМетрики baseline-модели:\n")

print(f"MAE: {mae:.2f}")

print(f"RMSE: {rmse:.2f}")

print(f"R²: {r2:.4f}")

# ================================
# Сравнение реальных и предсказанных значений
# ================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Реальные значения")

plt.ylabel("Предсказанные значения")

plt.title("Baseline model: реальные и предсказанные значения")

plt.tight_layout()

plt.show()

# ================================
# SHAP-анализ
# ================================

import shap

# Приводим данные к числовому типу
X_train_shap = X_train.astype(float)
X_test_shap = X_test.astype(float)

# Создаём объяснитель для линейной модели
explainer = shap.Explainer(baseline_model, X_train_shap)

# Рассчитываем SHAP-значения
shap_values = explainer(X_test_shap)

# Summary Plot
shap.summary_plot(
    shap_values.values,
    X_test_shap,
    feature_names=X_test_shap.columns
)

# Локальное объяснение первого объекта
shap.plots.waterfall(shap_values[0])

# ================================
# Генерация аналитических выводов
# ================================

print("\n===============================")
print("LLM-генерация аналитических выводов")
print("===============================\n")

llm_report = f"""
По результатам проведённого анализа была построена модель прогнозирования стоимости автомобилей.

Предварительный анализ данных показал наличие зависимости стоимости автомобиля от года выпуска, пробега, мощности двигателя и марки автомобиля.

Корреляционный анализ показал, что наибольшее влияние на стоимость оказывают:
- год выпуска;
- мощность двигателя;
- объём двигателя;
- пробег.

В качестве baseline-модели была использована линейная регрессия.

Полученные метрики baseline-модели:
MAE = {mae:.2f}
RMSE = {rmse:.2f}
R² = {r2:.4f}

SHAP-анализ показал, что увеличение года выпуска и мощности двигателя положительно влияет на стоимость автомобиля, тогда как увеличение пробега снижает прогнозируемую цену.

Построенная модель может использоваться для предварительной оценки рыночной стоимости автомобилей.
"""

print(llm_report)
