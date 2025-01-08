import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Abrir el archivo CSV
df = pd.read_csv('data.csv')

# Imprimir los principales datos y títulos de columnas
print(df.head())
print(df.columns)

# Limpiar los datos (eliminar filas con valores nulos)
df = df.dropna()

# Separar en datos de prueba y de predicción
X = df[['Socioeconomic Score', 'Study Hours', 'Sleep Hours', 'Attendance (%)']]
y = df['Grades']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión lineal
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Realizar predicciones con regresión lineal
y_pred_linear = linear_model.predict(X_test_scaled)

# Evaluar el modelo de regresión lineal
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R²: {r2_linear}')
print(f'Linear Regression - Mean Absolute Error: {mae_linear}')

# Crear el modelo de Random Forest Regressor con ajuste de hiperparámetros
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Mejor modelo de Random Forest Regressor
best_rf_model = grid_search.best_estimator_

# Realizar predicciones con el mejor modelo de Random Forest Regressor
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Evaluar el modelo de Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'Random Forest Regressor - Mean Squared Error: {mse_rf}')
print(f'Random Forest Regressor - R²: {r2_rf}')
print(f'Random Forest Regressor - Mean Absolute Error: {mae_rf}')

# Generar un gráfico de los resultados de regresión lineal
plt.scatter(y_test, y_pred_linear, color='red', label='Predicted (Linear Regression)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades (Linear Regression)')
plt.legend()
plt.show()

# Generar un gráfico de los resultados de Random Forest Regressor
plt.scatter(y_test, y_pred_rf, color='green', label='Predicted (Random Forest)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades (Random Forest)')
plt.legend()
plt.show()


