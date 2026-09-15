import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

# ============================================================
# RUTAS
# ============================================================

spectra_path = r"C:\DAVID\Universidad\TFM\Minerales de prueba\TEST\Euclidea\Distancia euclidea"

# ============================================================
# CONFIGURACIÓN
# ============================================================

start_x = 220
end_x = 1100
dx = 1

# Malla Raman común para todos los espectros
common_x = np.arange(start_x, end_x + dx, dx)

# ============================================================
# LECTURA DE LOS ESPECTROS
# ============================================================

spectra = {}

for file_name in os.listdir(spectra_path):

    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(spectra_path, file_name)

    x = []
    y = []

    with open(file_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("##"):
                continue

            try:
                values = line.split(",")

                x.append(float(values[0]))
                y.append(float(values[1]))

            except (ValueError, IndexError):
                continue

    x = np.array(x)
    y = np.array(y)

    # Normalización independiente de cada espectro
    norm = np.linalg.norm(y)

    if norm != 0:
        y = y / norm

    # Interpolación a una misma escala Raman
    interpolation = interp1d(
        x,
        y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    y_resampled = interpolation(common_x)

    spectra[file_name.replace(".txt", "")] = y_resampled


# ============================================================
# DISTANCIA EUCLÍDEA
# ============================================================

archaeological = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

geological = ["M1", "M2", "M6", "M7", "M12", "M21", "M14"]

results = []

for archaeological_sample in archaeological:

    distances = {}

    for geological_sample in geological:

        distance = euclidean(
            spectra[archaeological_sample],
            spectra[geological_sample]
        )

        distances[geological_sample] = distance

    closest = min(distances, key=distances.get)

    row = {
        "Archaeological": archaeological_sample,
        "Closest geological": closest,
        "Distance": distances[closest]
    }

    for sample, distance in distances.items():
        row[sample] = distance

    results.append(row)


# ============================================================
# RESULTADOS
# ============================================================

results_df = pd.DataFrame(results)

print("\nDISTANCIAS EUCLÍDEAS\n")
print(results_df.to_string(index=False))

# Guardar resultados
results_df.to_excel(
    "distancias_euclideas.xlsx",
    index=False
)