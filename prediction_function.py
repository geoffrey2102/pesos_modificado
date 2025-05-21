
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

def make_prediction(model_dir, peso, ciudad_origen, ciudad_destino, categoria, tipo_servicio, mes=1, diasemana=0, fragil=False):
    '''
    Función para hacer predicciones con el modelo guardado.
    
    Args:
        model_dir: Directorio donde están los archivos del modelo
        peso: Peso del paquete en kg
        ciudad_origen: Ciudad de origen
        ciudad_destino: Ciudad de destino
        categoria: Categoría del producto
        tipo_servicio: Tipo de servicio (Estándar, Express, Económico)
        mes: Mes del envío (1-12)
        diasemana: Día de la semana (0=Lunes, 6=Domingo)
        fragil: Indica si el paquete es frágil (True/False)
        
    Returns:
        float: Precio predicho en soles
    '''
    model = load_model(os.path.join(model_dir, 'best_model.h5'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    input_data = pd.DataFrame({
        'Peso_Kg': [peso],
        'Ciudad_Origen': [ciudad_origen],
        'Ciudad_Destino': [ciudad_destino],
        'Categoria': [categoria],
        'Tipo_Servicio': [tipo_servicio],
        'Mes': [mes],
        'DiaSemana': [diasemana],
        'Fragil': [fragil]
    })
    input_processed = preprocessor.transform(input_data)
    precio = model.predict(input_processed)[0][0]
    return precio

if __name__ == '__main__':
    precio = make_prediction(
        model_dir='modelos_resultados',
        peso=5.0,
        ciudad_origen='Lima',
        ciudad_destino='Arequipa',
        categoria='Electrónicos',
        tipo_servicio='Estándar',
        mes=1,
        diasemana=0,
        fragil=True
    )
    print(f"Precio predicho: {precio:.2f} soles")
