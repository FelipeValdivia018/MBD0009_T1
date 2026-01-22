# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/data_inferencia_causal.csv')


#####################################
##### Estimación CATE s-learner #####
#####################################

# Copia de seguridad
data = df.copy()

# Variables
X = data[["edad", "ingreso"]]
W = data["W"]
Y = data["Y"]

# X extendido con el tratamiento
X_s = X.copy()
X_s["W"] = W

s_learner = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42
)

s_learner.fit(X_s, Y)

# Copias contrafactuales
X_treated = X.copy()
X_treated["W"] = 1

X_control = X.copy()
X_control["W"] = 0

# Predicciones
Y_hat_1 = s_learner.predict(X_treated)
Y_hat_0 = s_learner.predict(X_control)

data["cate_s_learner"] = Y_hat_1 - Y_hat_0




# Segmentación de edad
data["segmento_edad"] = pd.cut(
    data["edad"],
    bins=[0, 30, 45, 60, 75],
    labels=["Joven", "Adulto", "Adulto Mayor", "Senior"]
)

#Segmntación ingreso
data["segmento_ingreso"] = pd.qcut(
    data["ingreso"],
    q=3,
    labels=["Bajo", "Medio", "Alto"]
)

#Resumen CATE_s
cate_segmentos = (
    data
    .groupby(["segmento_edad", "segmento_ingreso"])
    ["cate_s_learner"]
    .mean()
    .reset_index()
)

print(cate_segmentos)




##### Gráficos s_learner #####
orden_ingreso = ["Bajo", "Medio", "Alto"]
orden_edad = ["Joven", "Adulto", "Adulto Mayor", "Senior"]

cate_segmentos["segmento_ingreso"] = pd.Categorical(
    cate_segmentos["segmento_ingreso"],
    categories=orden_ingreso,
    ordered=True
)

cate_segmentos["segmento_edad"] = pd.Categorical(
    cate_segmentos["segmento_edad"],
    categories=orden_edad,
    ordered=True
)

### Una linea por grupo etario ###
plt.figure(figsize=(8, 5))

for edad in orden_edad:
    subset = cate_segmentos[cate_segmentos["segmento_edad"] == edad]
    subset = subset.sort_values("segmento_ingreso")

    plt.plot(
        subset["segmento_ingreso"],
        subset["cate_s_learner"],
        marker="o",
        label=edad
    )

plt.xlabel("Segmento de Ingreso")
plt.ylabel("CATE promedio")
plt.title("Efecto de la Publicidad por Ingreso y Edad")
plt.legend(title="Edad")
plt.grid(True)
plt.tight_layout()
plt.show()

### Una linea por nivel de ingreso ###
plt.figure(figsize=(8, 5))

for ingreso in orden_ingreso:
    subset = cate_segmentos[cate_segmentos["segmento_ingreso"] == ingreso]
    subset = subset.sort_values("segmento_edad")

    plt.plot(
        subset["segmento_edad"],
        subset["cate_s_learner"],
        marker="o",
        label=ingreso
    )

plt.xlabel("Segmento de Edad")
plt.ylabel("CATE promedio")
plt.title("Efecto de la Publicidad por Edad e Ingreso")
plt.legend(title="Ingreso")
plt.grid(True)
plt.tight_layout()
plt.show()





#####################################
##### Estimación CATE t-learner #####
#####################################

# Variables
X_t = data[["edad", "ingreso"]]
Y_t = data["Y"]
W_t = data["W"]

# Grupo tratado (W = 1)
X_1_t = X_t[W_t == 1]
Y_1_t = Y_t[W_t == 1]

# Grupo control (W = 0)
X_0_t = X_t[W_t == 0]
Y_0_t = Y_t[W_t == 0]

#Modelos
model_treated_t = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42
)

model_control_t = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42
)

model_treated_t.fit(X_1_t, Y_1_t)
model_control_t.fit(X_0_t, Y_0_t)


Y_hat_1_t = model_treated_t.predict(X_t)
Y_hat_0_t = model_control_t.predict(X_t)

data["cate_t_learner"] = Y_hat_1_t - Y_hat_0_t

cate_segmentos_t = (
    data
    .groupby(["segmento_edad", "segmento_ingreso"])["cate_t_learner"]
    .mean()
    .reset_index()
)

print(cate_segmentos_t)



##### Gráficos t_learner #####
# Categorías
orden_ingreso_t = ["Bajo", "Medio", "Alto"]
orden_edad_t = ["Joven", "Adulto", "Adulto Mayor", "Senior"]

cate_segmentos_t["segmento_ingreso"] = pd.Categorical(
    cate_segmentos_t["segmento_ingreso"],
    categories=orden_ingreso_t,
    ordered=True
)

cate_segmentos_t["segmento_edad"] = pd.Categorical(
    cate_segmentos_t["segmento_edad"],
    categories=orden_edad_t,
    ordered=True
)

### Una linea por grupo etario ###
plt.figure(figsize=(8, 5))

for edad_t in orden_edad_t:
    subset_t = cate_segmentos_t[
        cate_segmentos_t["segmento_edad"] == edad_t
    ].sort_values("segmento_ingreso")

    plt.plot(
        subset_t["segmento_ingreso"],
        subset_t["cate_t_learner"],
        marker="o",
        label=edad_t
    )

plt.xlabel("Segmento de Ingreso")
plt.ylabel("CATE promedio (T-Learner)")
plt.title("Efecto de la Publicidad por Ingreso y Edad (T-Learner)")
plt.legend(title="Edad")
plt.grid(True)
plt.tight_layout()
plt.show()


### Una linea por nivel de ingreso ###
plt.figure(figsize=(8, 5))

for ingreso_t in orden_ingreso_t:
    subset_t = cate_segmentos_t[
        cate_segmentos_t["segmento_ingreso"] == ingreso_t
    ].sort_values("segmento_edad")

    plt.plot(
        subset_t["segmento_edad"],
        subset_t["cate_t_learner"],
        marker="o",
        label=ingreso_t
    )

plt.xlabel("Segmento de Edad")
plt.ylabel("CATE promedio (T-Learner)")
plt.title("Efecto de la Publicidad por Edad e Ingreso (T-Learner)")
plt.legend(title="Ingreso")
plt.grid(True)
plt.tight_layout()
plt.show()
