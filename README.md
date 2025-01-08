Predicción de Calificaciones con Regresión Lineal y Random Forest
Este proyecto utiliza técnicas de regresión lineal y Random Forest para predecir las calificaciones de estudiantes basándose en varios factores como el puntaje socioeconómico, las horas de estudio, las horas de sueño y la asistencia a clases. Se utiliza un conjunto de datos de estudiantes para entrenar y evaluar los modelos.

Descripción del Proyecto
El código realiza los siguientes pasos:

Carga y Exploración de Datos:

Se carga un archivo CSV con datos sobre estudiantes.
Se imprimen las primeras filas y los nombres de las columnas para explorar el contenido del archivo.
Limpieza de Datos:

Se eliminan las filas con valores nulos para asegurar que los datos estén completos.
Preprocesamiento:

Se separan las características (X) y la variable objetivo (y).
Se divide el conjunto de datos en entrenamiento y prueba (80%/20%).
Las características se normalizan utilizando el escalador StandardScaler para mejorar el rendimiento de los modelos.
Entrenamiento y Evaluación de Modelos:

Se crea un modelo de Regresión Lineal y se evalúa utilizando el error cuadrático medio (MSE), el coeficiente de determinación (R²) y el error absoluto medio (MAE).
Se crea un modelo de Random Forest Regressor, y se optimizan sus hiperparámetros mediante una búsqueda en cuadrícula (GridSearchCV). Luego, se evalúa el modelo optimizado con las mismas métricas.
Visualización de Resultados:

Se generan gráficos de dispersión para comparar las calificaciones reales con las predicciones de cada modelo (Regresión Lineal y Random Forest).
Requisitos
Asegúrate de tener instaladas las siguientes bibliotecas en tu entorno Python:

pandas
matplotlib
scikit-learn
Puedes instalarlas usando el siguiente comando:

bash
Copiar código
pip install pandas matplotlib scikit-learn
Uso
Coloca el archivo data.csv en el mismo directorio que el script, o actualiza la ruta del archivo en el código.
Ejecuta el script en tu entorno de Python.
Los resultados de las métricas de evaluación se imprimirán en la consola.
Los gráficos comparativos entre las calificaciones reales y las predicciones de ambos modelos (Regresión Lineal y Random Forest) se generarán y mostrarán.
Resultados Esperados
El código imprimirá los valores de Mean Squared Error (MSE), R² y Mean Absolute Error (MAE) para ambos modelos.
Los gráficos generados mostrarán la comparación visual de las predicciones frente a los valores reales de calificaciones.
