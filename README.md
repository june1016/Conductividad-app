# Modelado de Conductividad Eléctrica en Soluciones Salinas

## 📋 Descripción del Proyecto

Este proyecto implementa una aplicación web para modelar la relación entre la concentración de sal y la conductividad eléctrica del agua utilizando diversos algoritmos de machine learning. La aplicación permite cargar datos experimentales, seleccionar algoritmos de modelado, visualizar resultados y generar reportes en PDF.

## 🚀 Tecnologías Utilizadas

### Frontend:
- **Dash**: Framework web para aplicaciones analíticas
- **Dash Bootstrap Components**: Componentes de Bootstrap para Dash
- **Plotly**: Librería para visualizaciones interactivas
- **HTML/CSS/JavaScript**: Para la interfaz de usuario

### Backend:
- **Python 3.8+**: Lenguaje de programación principal
- **Scikit-learn**: Librería de machine learning
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Cálculos numéricos

### Procesamiento de Datos:
- **Regresión Lineal**: Mínimos cuadrados y método de varianza
- **Regresión Polinomial**: Grados 2-5
- **Descenso de Gradiente**: Implementación iterativa
- **Redes Neuronales**: Neurona simple con función sigmoide

### Generación de Reportes:
- **WeasyPrint**: Generación de PDFs desde HTML
- **Plotly.io**: Conversión de gráficos a imágenes

## 🏗️ Estructura del Código

```
conductividad-app/
│
├── app.py                 # Aplicación principal Dash
├── requirements.txt       # Dependencias del proyecto
├── Procfile              # Configuración para despliegue
├── assets/               # Recursos estáticos
│   └── style.css         # Estilos personalizados
│
├── datos_ejemplo/        # Datos de ejemplo
│   └── datos_conductividad.csv
│
└── README.md             # Documentación
```

## ⚙️ Funcionalidades Implementadas

### 1. Carga de Datos
- **Upload de archivos CSV**: Interfaz drag-and-drop
- **Datos de ejemplo**: Conjunto predefinido de mediciones
- **Validación**: Verificación de formato y estructura de datos

### 2. Selección de Algoritmos
- **Regresión Lineal**: Mínimos cuadrados y método de varianza
- **Regresión Polinomial**: Grados 2-5 configurables
- **Descenso de Gradiente**: Con parámetros personalizables
- **Neurona Artificial**: Con función de activación sigmoide

### 3. Visualización de Resultados
- **Gráficos interactivos**: Datos reales vs. modelo ajustado
- **Ecuaciones**: Formato matemático legible
- **Métricas de error**: MSE y RMSE calculados automáticamente

### 4. Generación de Reportes
- **Exportación a PDF**: Incluye gráficos y resultados
- **Formato profesional**: Diseño adecuado para documentación científica
- **Descarga automática**: Disponible desde la interfaz

## 🎯 Uso de la Aplicación

### Instalación:
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/conductividad-app.git

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución:
```bash
python app.py
```
La aplicación estará disponible en: http://127.0.0.1:8050/

### Despliegue:
La aplicación está configurada para despliegue en:
- **Heroku**: Configurado con Procfile
- **Railway**: Compatible con entornos cloud
- **Render**: Configuración incluida

## 📊 Resultados y Modelado

### Algoritmos Implementados:
1. **Regresión Lineal**: Modelo lineal básico
2. **Regresión Polinomial**: Captura relaciones no lineales
3. **Descenso de Gradiente**: Optimización iterativa
4. **Red Neuronal**: Modelado de relaciones complejas

### Métricas de Evaluación:
- **MSE (Error Cuadrático Medio)**: Medida de error absoluto
- **RMSE (Raíz del Error Cuadrático Medio)**: Error en unidades originales
- **Ecuaciones**: Formato matemático para cada modelo

## 🔍 Ejemplo de Uso

1. **Cargar datos**: Subir archivo CSV o usar datos de ejemplo
2. **Seleccionar algoritmo**: Elegir entre las opciones disponibles
3. **Configurar parámetros**: Ajustar según el algoritmo seleccionado
4. **Ejecutar modelo**: Procesar datos y generar resultados
5. **Exportar resultados**: Descargar reporte PDF con todos los datos

## 📋 Estructura de Datos

### Formato CSV esperado:
```csv
Cucharaditas,Conductancia
0,0.00360
1,0.00708
2,0.01304
...
```

### Datos de ejemplo incluidos:
Mediciones realistas de conductividad para concentraciones de 0 a 10 cucharaditas de sal, con valores físicamente posibles y consistentes.

## 🚀 Despliegue en Producción

### Heroku:
```bash
# Crear aplicación
heroku create tu-aplicacion-conductividad

# Desplegar
git push heroku main
```

### Railway:
- Conectar repositorio GitHub
- Despliegue automático al hacer push

## 📝 Próximas Mejoras

- [] Implementación de validación cruzada
- [ ] Comparativa de múltiples modelos simultáneos
- [ ] Optimización de hiperparámetros automática
- [ ] Soporte para más algoritmos de machine learning
- [ ] Exportación de modelos entrenados

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor, leer las guías de contribución antes de enviar un pull request.

## 📞 Soporte

Para soporte o preguntas, contactar a [june16.dev@outlook.com] o abrir un issue en el repositorio GitHub.

---

**Nota**: Esta aplicación fue desarrollada como parte de un proyecto académico de modelado y simulación de sistemas. Los resultados deben validarse experimentalmente para aplicaciones reales.
