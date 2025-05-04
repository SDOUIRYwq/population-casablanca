
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ---------------------------
# Données historiques
# ---------------------------
data = {
    'Année': [2004, 2014, 2024],
    'Population': [2949805, 3359818, 3019818]
}
df = pd.DataFrame(data)

# Prétraitement
scaler = MinMaxScaler(feature_range=(0, 1))
population_scaled = scaler.fit_transform(df['Population'].values.reshape(-1, 1))

# Séquences pour LSTM
X = []
y = []
for i in range(len(population_scaled) - 1):
    X.append(population_scaled[i])
    y.append(population_scaled[i + 1])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], 1, 1))

# Modèle LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=300, verbose=0)

# Interface Streamlit
st.title("📊 Prédiction de la population de Casablanca")

année_input = st.number_input("Choisissez l'année à prédire (>=2025)", min_value=2025, max_value=2050, step=1)

if st.button("Prédire"):
    # Prédiction en chaîne jusqu'à l'année demandée
    années_connues = df['Année'].tolist()
    populations_connues = df['Population'].tolist()

    dernier_point = population_scaled[-1].reshape((1, 1, 1))
    années_future = années_connues.copy()
    populations_future = populations_connues.copy()

    for année in range(2025, année_input + 1):
        prediction_scaled = model.predict(dernier_point)
        prediction_population = scaler.inverse_transform(prediction_scaled)[0][0]
        années_future.append(année)
        populations_future.append(prediction_population)
        dernier_point = prediction_scaled.reshape((1, 1, 1))

    st.success(f"👉 Population prédite pour {année_input} : {int(populations_future[-1]):,} habitants")

    # Affichage graphique
    fig, ax = plt.subplots()
    ax.plot(années_connues, populations_connues, marker='o', label='Données historiques')
    ax.plot(années_future, populations_future, marker='x', linestyle='--', color='red', label='Prédictions')
    ax.set_xlabel('Année')
    ax.set_ylabel('Population')
    ax.set_title('Prédiction de la population de Casablanca')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
