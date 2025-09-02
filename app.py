# app.py
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
import io
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import weasyprint
from datetime import datetime

# Datos de ejemplo actualizados
sample_data = [
    {'Cucharaditas': 0, 'Conductancia': 0.00360},
    {'Cucharaditas': 1, 'Conductancia': 0.00708},
    {'Cucharaditas': 2, 'Conductancia': 0.01304},
    {'Cucharaditas': 3, 'Conductancia': 0.03614},
    {'Cucharaditas': 4, 'Conductancia': 0.05882},
    {'Cucharaditas': 5, 'Conductancia': 0.08110},
    {'Cucharaditas': 6, 'Conductancia': 0.09372},
    {'Cucharaditas': 7, 'Conductancia': 0.10989},
    {'Cucharaditas': 8, 'Conductancia': 0.12658},
    {'Cucharaditas': 9, 'Conductancia': 0.14225},
    {'Cucharaditas': 10, 'Conductancia': 0.16051}
]

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
    suppress_callback_exceptions=True
)
app.title = "Modelado de Conductividad Eléctrica"
server = app.server

# Layout de la aplicación
app.layout = dbc.Container([
    # Header
    dbc.Navbar(
        dbc.Container([
            html.Div([
                html.I(className="fas fa-chart-line me-2"),
                html.Span("Modelado de Conductividad Eléctrica", className="navbar-brand mb-0 h1")
            ], className="d-flex align-items-center"),
            dbc.Button("Ayuda", id="help-button", color="light", className="ms-auto")
        ]),
        color="primary",
        dark=True,
        sticky="top"
    ),
    
    # Modal de ayuda
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Ayuda")),
        dbc.ModalBody([
            html.H5("Carga de datos"),
            html.P("Puedes subir un archivo CSV con dos columnas: 'Cucharaditas' y 'Conductancia', o utilizar nuestros datos de ejemplo."),
            html.H5("Algoritmos disponibles"),
            html.Ul([
                html.Li("Regresión Lineal (Mínimos Cuadrados)"),
                html.Li("Regresión Polinomial (hasta grado 5)"),
                html.Li("Descenso de Gradiente"),
                html.Li("Neurona Artificial con Sigmoide")
            ]),
            html.H5("Resultados"),
            html.P("Visualiza el modelo ajustado, la ecuación resultante y las métricas de error (MSE y RMSE).")
        ]),
        dbc.ModalFooter(dbc.Button("Cerrar", id="close-help", className="ms-auto"))
    ], id="help-modal", is_open=False),
    
    # Componentes Store para almacenar parámetros
    dcc.Store(id='polynomial-degree-store', data=2),
    dcc.Store(id='learning-rate-store', data=0.01),
    dcc.Store(id='iterations-store', data=1000),
    dcc.Store(id='last-figure-data', data=json.dumps({})),  # Store para guardar datos del gráfico
    
    # Contenido principal
    dbc.Row([
        # Panel izquierdo - Inputs
        dbc.Col([
            # Card de carga de datos
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-database me-2"),
                    "Carga de Datos"
                ]),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt me-2"),
                            'Arrastra y suelta tu archivo CSV aquí o haz clic'
                        ]),
                        style={
                            'width': '100%', 'height': '100px', 'lineHeight': '100px',
                            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'marginBottom': '10px', 'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    dbc.Checklist(
                        options=[{"label": "Usar datos de ejemplo", "value": True}],
                        value=[True],
                        id="use-sample-data",
                        switch=True,
                    ),
                    html.Div(id='output-data-upload'),
                    html.Div(id='data-table-container', style={'marginTop': '20px'})
                ])
            ], className="mb-4"),
            
            # Card de selección de algoritmo
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-calculator me-2"),
                    "Selección de Algoritmo"
                ]),
                dbc.CardBody([
                    dbc.Select(
                        id="algorithm-select",
                        options=[
                            {"label": "Regresión Lineal (Mínimos Cuadrados)", "value": "linear"},
                            {"label": "Regresión Polinomial", "value": "polynomial"},
                            {"label": "Descenso de Gradiente", "value": "gradient"},
                            {"label": "Neurona Artificial con Sigmoide", "value": "neuron"}
                        ],
                        value="linear"
                    ),
                    html.Div(id="algorithm-params", style={'marginTop': '15px'})
                ])
            ], className="mb-4"),
            
            # Botón de ejecución
            dbc.Button(
                "Ejecutar Modelo", 
                id="execute-button", 
                color="primary", 
                className="w-100 mb-4",
                disabled=False
            )
        ], md=4),
        
        # Panel derecho - Resultados
        dbc.Col([
            # Card de visualización de resultados
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Visualización de Resultados"
                ]),
                dbc.CardBody([
                    dcc.Graph(id="results-graph", style={'display': 'none'}),
                    html.Div(
                        id="no-data-message",
                        children=[
                            html.Div([
                                html.I(className="fas fa-chart-bar fa-3x text-muted mb-3"),
                                html.P("Ejecuta un modelo para ver los resultados", className="text-muted")
                            ], className="text-center p-5")
                        ]
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Ecuación del Modelo"),
                                dbc.CardBody([
                                    html.Pre("y = mx + b", id="equation-result", className="text-primary")
                                ])
                            ])
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Métricas de Error"),
                                dbc.CardBody([
                                    html.Div([
                                        html.Span("MSE: ", className="text-muted"),
                                        html.Span("0.000000", id="mse-result", className="text-success fw-bold")
                                    ]),
                                    html.Div([
                                        html.Span("RMSE: ", className="text-muted"),
                                        html.Span("0.000000", id="rmse-result", className="text-success fw-bold")
                                    ])
                                ])
                            ])
                        ], md=6)
                    ], className="mt-3")
                ])
            ], className="mb-4"),
            
            # Botón de descarga de PDF
            dbc.Button(
                "Descargar Reporte PDF", 
                id="download-pdf-button", 
                color="outline-primary", 
                className="w-100"
            ),
            dcc.Download(id="download-pdf")
        ], md=8)
    ], className="mt-4")
], fluid=True)

# Callback para mostrar/ocultar el modal de ayuda
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks"), Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")]
)
def toggle_help_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback para actualizar la UI de parámetros
@app.callback(
    Output("algorithm-params", "children"),
    Input("algorithm-select", "value"),
    [State("polynomial-degree-store", "data"),
     State("learning-rate-store", "data"),
     State("iterations-store", "data")]
)
def update_algorithm_ui(algorithm, degree, lr, iters):
    if algorithm == "polynomial":
        return dbc.Select(
            id="polynomial-degree",
            options=[{"label": f"Grado {i}", "value": i} for i in range(2, 6)],
            value=degree
        )
    elif algorithm in ["gradient", "neuron"]:
        return dbc.Row([
            dbc.Col([
                dbc.Label("Tasa de aprendizaje"),
                dbc.Input(id="learning-rate", type="number", value=lr, step=0.001, min=0.001, max=1)
            ]),
            dbc.Col([
                dbc.Label("Iteraciones"),
                dbc.Input(id="iterations", type="number", value=iters, step=100, min=100, max=10000)
            ])
        ])
    else:
        return html.P("No se requieren parámetros adicionales para este algoritmo.", className="text-muted")

# Callbacks para actualizar los Stores desde los componentes de parámetros
@app.callback(
    Output("polynomial-degree-store", "data"),
    Input("polynomial-degree", "value"),
    prevent_initial_call=True
)
def update_poly_degree(value):
    return value

@app.callback(
    Output("learning-rate-store", "data"),
    Input("learning-rate", "value"),
    prevent_initial_call=True
)
def update_lr(value):
    return value

@app.callback(
    Output("iterations-store", "data"),
    Input("iterations", "value"),
    prevent_initial_call=True
)
def update_iters(value):
    return value

# Callback para procesar la carga de datos
@app.callback(
    [Output("output-data-upload", "children"),
     Output("data-table-container", "children"),
     Output("execute-button", "disabled")],
    [Input("upload-data", "contents"),
     Input("use-sample-data", "value")],
    [State("upload-data", "filename")]
)
def update_output(contents, use_sample, filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "use-sample-data" and use_sample:
        df = pd.DataFrame(sample_data)
        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")
        return "Usando datos de ejemplo", table, False
    
    elif trigger_id == "upload-data" and contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            if "Cucharaditas" not in df.columns or "Conductancia" not in df.columns:
                return "Error: El CSV debe contener columnas 'Cucharaditas' y 'Conductancia'", None, True
            table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")
            return f"Archivo cargado: {filename}", table, False
        except Exception as e:
            return f"Error al procesar el archivo: {str(e)}", None, True
    
    return "", None, True

# Callback para ejecutar el modelo y mostrar resultados
@app.callback(
    [Output("results-graph", "figure"),
     Output("results-graph", "style"),
     Output("no-data-message", "style"),
     Output("equation-result", "children"),
     Output("mse-result", "children"),
     Output("rmse-result", "children"),
     Output("last-figure-data", "data")],  # Agregar salida para almacenar datos del gráfico
    [Input("execute-button", "n_clicks")],
    [State("use-sample-data", "value"),
     State("algorithm-select", "value"),
     State("upload-data", "contents"),
     State("polynomial-degree-store", "data"),
     State("learning-rate-store", "data"),
     State("iterations-store", "data")],
    prevent_initial_call=True
)
def execute_model(n_clicks, use_sample, algorithm, contents, degree, lr, iterations):
    if n_clicks is None:
        return go.Figure(), {"display": "none"}, {"display": "block"}, "y = mx + b", "0.000000", "0.000000", json.dumps({})
    
    # Obtener los datos
    if use_sample:
        df = pd.DataFrame(sample_data)
    elif contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    else:
        return go.Figure(), {"display": "none"}, {"display": "block"}, "y = mx + b", "0.000000", "0.000000", json.dumps({})
    
    X = df["Cucharaditas"].values.reshape(-1, 1)
    y = df["Conductancia"].values
    
    # Entrenar el modelo seleccionado
    if algorithm == "linear":
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        equation = f"y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}"
        poly_features = None
        scaler = None
    
    elif algorithm == "polynomial":
        # Convertir degree a int si es string
        degree_int = int(degree) if isinstance(degree, (str, float)) else degree
        poly_features = PolynomialFeatures(degree=degree_int)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Construir la ecuación polinomial
        coefs = model.coef_
        intercept = model.intercept_
        equation = f"y = {intercept:.6f}"
        for i in range(1, degree_int + 1):
            equation += f" + {coefs[i]:.6f}x^{i}" if i > 1 else f" + {coefs[i]:.6f}x"
        scaler = None
    
    elif algorithm == "gradient":
        # Implementación simplificada de descenso de gradiente
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        equation = f"y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}"
        poly_features = None
        scaler = None
    
    elif algorithm == "neuron":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = MLPRegressor(
            hidden_layer_sizes=(1,),
            activation='logistic',
            solver='lbfgs',
            max_iter=iterations,
            random_state=42
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        equation = "Neurona con función sigmoide (parámetros internos)"
        poly_features = None
    
    # Calcular métricas de error
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Crear gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Cucharaditas"], y=df["Conductancia"],
        mode='markers', name='Datos reales',
        marker=dict(size=10, color='blue')
    ))
    
    # Crear puntos para la línea del modelo
    X_range = np.linspace(df["Cucharaditas"].min(), df["Cucharaditas"].max(), 100).reshape(-1, 1)
    if algorithm == "linear":
        y_range = model.predict(X_range)
    elif algorithm == "polynomial":
        X_range_poly = poly_features.transform(X_range)
        y_range = model.predict(X_range_poly)
    elif algorithm == "gradient":
        y_range = model.predict(X_range)
    elif algorithm == "neuron":
        X_range_scaled = scaler.transform(X_range)
        y_range = model.predict(X_range_scaled)
    
    fig.add_trace(go.Scatter(
        x=X_range.flatten(), y=y_range,
        mode='lines', name='Modelo ajustado',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Conductancia en función de la concentración de sal",
        xaxis_title="Cucharaditas de sal",
        yaxis_title="Conductancia (S)",
        showlegend=True
    )
    
    # Extraer datos para almacenar en el store (simplificados)
    chart_data = {
        'x_real': df["Cucharaditas"].tolist(),
        'y_real': df["Conductancia"].tolist(),
        'x_model': X_range.flatten().tolist(),
        'y_model': y_range.tolist(),
        'title': "Conductancia en función de la concentración de sal",
        'xaxis_title': "Cucharaditas de sal",
        'yaxis_title': "Conductancia (S)"
    }
    
    return fig, {"display": "block"}, {"display": "none"}, equation, f"{mse:.6f}", f"{rmse:.6f}", json.dumps(chart_data)

# Función para crear una representación textual del gráfico en el PDF
def create_chart_text_representation(chart_data_json):
    try:
        chart_data = json.loads(chart_data_json)
        if not chart_data or chart_data == {}:
            return "No hay datos de gráfico disponibles"
        
        # Crear una tabla de datos para el PDF
        text_repr = "<h3>Datos del Modelo</h3>\n"
        text_repr += "<p><strong>Datos Reales:</strong></p>\n"
        text_repr += "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"
        text_repr += "<tr><th>Cucharaditas</th><th>Conductancia</th></tr>\n"
        
        x_real = chart_data.get('x_real', [])
        y_real = chart_data.get('y_real', [])
        
        for i in range(min(len(x_real), len(y_real), 10)):  # Limitar a 10 puntos para no sobrecargar
            text_repr += f"<tr><td>{x_real[i]}</td><td>{y_real[i]:.6f}</td></tr>\n"
        
        if len(x_real) > 10:
            text_repr += "<tr><td colspan='2'>... (mostrando 10 de {} puntos)</td></tr>\n".format(len(x_real))
        
        text_repr += "</table>\n"
        
        text_repr += "<p><strong>Predicciones del Modelo:</strong></p>\n"
        text_repr += "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"
        text_repr += "<tr><th>Cucharaditas</th><th>Conductancia Predicha</th></tr>\n"
        
        x_model = chart_data.get('x_model', [])
        y_model = chart_data.get('y_model', [])
        
        # Mostrar puntos clave del modelo
        step = max(1, len(x_model) // 10)  # Mostrar aproximadamente 10 puntos
        for i in range(0, len(x_model), step):
            text_repr += f"<tr><td>{x_model[i]:.2f}</td><td>{y_model[i]:.6f}</td></tr>\n"
        
        text_repr += "</table>\n"
        
        return text_repr
    except Exception as e:
        print(f"Error al crear representación textual del gráfico: {e}")
        return "<p>No hay datos de gráfico disponibles</p>"

# Callback para generar y descargar el PDF
@app.callback(
    Output("download-pdf", "data"),
    Input("download-pdf-button", "n_clicks"),
    [State("last-figure-data", "data"),
     State("equation-result", "children"),
     State("mse-result", "children"),
     State("rmse-result", "children"),
     State("algorithm-select", "value")],
    prevent_initial_call=True
)
def generate_pdf(n_clicks, chart_data_json, equation, mse, rmse, algorithm):
    if n_clicks is None:
        return None
    
    # Crear representación textual del gráfico
    chart_representation = create_chart_text_representation(chart_data_json)
    
    # Crear contenido HTML para el PDF
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #0c4a6e; }}
            h2 {{ color: #1e6091; margin-top: 30px; }}
            h3 {{ color: #2d70b3; }}
            .equation {{ font-family: monospace; font-size: 18px; color: #0ea5e9; background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            .metrics {{ margin: 20px 0; }}
            .metric {{ margin: 10px 0; font-weight: bold; }}
            .metric-value {{ color: #0ea5e9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #e9ecef; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Modelado de Conductividad</h1>
        <p><strong>Fecha:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Algoritmo utilizado:</strong> {algorithm}</p>
        
        <div class="metrics">
            <h2>Resultados del Modelo</h2>
            <p><strong>Ecuación:</strong></p>
            <div class="equation">{equation}</div>
            <div class="metric">MSE: <span class="metric-value">{mse}</span></div>
            <div class="metric">RMSE: <span class="metric-value">{rmse}</span></div>
        </div>
        
        <div class="chart-data">
            <h2>Datos del Gráfico</h2>
            {chart_representation}
        </div>
        
        <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #0c4a6e;">
            <h3>Nota</h3>
            <p>Este reporte contiene los datos y resultados del modelo de conductividad. Para visualizar el gráfico completo, 
            por favor utilice la interfaz web de la aplicación.</p>
        </div>
    </body>
    </html>
    """
    
    # Convertir HTML a PDF
    try:
        # Crear PDF con WeasyPrint
        pdf = weasyprint.HTML(string=html_content).write_pdf()
        return dcc.send_bytes(pdf, "reporte_conductividad.pdf")
    except Exception as e:
        print(f"Error al generar PDF: {e}")
        # Fallback: devolver PDF con mensaje de error
        error_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>Error al generar el reporte PDF</h1>
            <p>Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Detalles del error: {str(e)}</p>
        </body>
        </html>
        """
        pdf = weasyprint.HTML(string=error_html).write_pdf()
        return dcc.send_bytes(pdf, "error_reporte.pdf")

if __name__ == "__main__":
    app.run(debug=True)