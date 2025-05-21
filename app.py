from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the model and preprocessor with custom objects
model = tf.keras.models.load_model('best_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
preprocessor = joblib.load('preprocessor.joblib')

# Define options for dropdowns (based on dataset)
ciudades = [
    'Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura', 'Cusco', 'Iquitos',
    'Huancayo', 'Pucallpa', 'Tacna', 'Ayacucho', 'Chimbote', 'Ica', 'Juliaca', 'Tarapoto'
]
categorias = [
    'Documentos', 'Ropa', 'Electrónicos', 'Alimentos', 'Muebles',
    'Libros', 'Medicamentos', 'Repuestos', 'Herramientas', 'Otros'
]
tipos_servicio = ['Estándar', 'Express', 'Económico']
meses = list(range(1, 13))
dias_semana = [
    'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'
]

# HTML template as a string
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shipping Cost Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin: 10px 0 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            text-align: center;
            font-size: 1.2em;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Shipping Cost Predictor</h1>
    <form id="predictionForm">
        <label for="peso">Peso (Kg):</label>
        <input type="number" id="peso" name="peso" step="0.1" required>

        <label for="ciudad_origen">Ciudad de Origen:</label>
        <select id="ciudad_origen" name="ciudad_origen" required>
            {% for ciudad in ciudades %}
                <option value="{{ ciudad }}">{{ ciudad }}</option>
            {% endfor %}
        </select>

        <label for="ciudad_destino">Ciudad de Destino:</label>
        <select id="ciudad_destino" name="ciudad_destino" required>
            {% for ciudad in ciudades %}
                <option value="{{ ciudad }}">{{ ciudad }}</option>
            {% endfor %}
        </select>

        <label for="categoria">Categoría:</label>
        <select id="categoria" name="categoria" required>
            {% for categoria in categorias %}
                <option value="{{ categoria }}">{{ categoria }}</option>
            {% endfor %}
        </select>

        <label for="tipo_servicio">Tipo de Servicio:</label>
        <select id="tipo_servicio" name="tipo_servicio" required>
            {% for tipo in tipos_servicio %}
                <option value="{{ tipo }}">{{ tipo }}</option>
            {% endfor %}
        </select>

        <label for="mes">Mes:</label>
        <select id="mes" name="mes" required>
            {% for mes in meses %}
                <option value="{{ mes }}">{{ mes }}</option>
            {% endfor %}
        </select>

        <label for="dia_semana">Día de la Semana:</label>
        <select id="dia_semana" name="dia_semana" required>
            {% for dia in dias_semana %}
                <option value="{{ dia }}">{{ dia }}</option>
            {% endfor %}
        </select>

        <label for="fragil">Frágil:</label>
        <select id="fragil" name="fragil" required>
            <option value="True">Sí</option>
            <option value="False">No</option>
        </select>

        <button type="submit">Predecir Costo</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.error) {
                    document.getElementById('result').innerText = `Error: ${data.error}`;
                } else {
                    document.getElementById('result').innerText = `Costo Estimado: S/ ${data.prediction}`;
                }
            } catch (error) {
                document.getElementById('result').innerText = `Error: ${error.message}`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    from jinja2 import Template
    return Template(INDEX_HTML).render(
        ciudades=ciudades,
        categorias=categorias,
        tipos_servicio=tipos_servicio,
        meses=meses,
        dias_semana=dias_semana
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        peso = float(data['peso'])
        ciudad_origen = data['ciudad_origen']
        ciudad_destino = data['ciudad_destino']
        categoria = data['categoria']
        tipo_servicio = data['tipo_servicio']
        mes = int(data['mes'])
        dia_semana = dias_semana.index(data['dia_semana'])  # Convert to 0-6
        fragil = data['fragil'] == 'True'

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Peso_Kg': [peso],
            'Ciudad_Origen': [ciudad_origen],
            'Ciudad_Destino': [ciudad_destino],
            'Categoria': [categoria],
            'Tipo_Servicio': [tipo_servicio],
            'Mes': [mes],
            'DiaSemana': [dia_semana],
            'Fragil': [fragil]
        })

        # Preprocess input
        input_processed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_processed)[0][0]

        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)