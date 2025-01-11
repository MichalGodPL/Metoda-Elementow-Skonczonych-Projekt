import numpy as np

# Definicja funkcji kształtu dla 1D na brzegu
# N1 i N2 to funkcje kształtu dla dwóch węzłów w elemencie liniowym

def shape_functions(ksi):
    N1 = 0.5 * (1 - ksi)
    N2 = 0.5 * (1 + ksi)
    return np.array([N1, N2])

# Współrzędne punktów całkowania i wagi dla metody Gaussa (dla różnych ilości punktów całkowania)
gauss_points_2 = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
gauss_weights_2 = np.array([1, 1])

gauss_points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
gauss_weights_3 = np.array([5/9, 8/9, 5/9])

gauss_points_4 = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
gauss_weights_4 = np.array([0.347855, 0.652145, 0.652145, 0.347855])

def calculate_vector_p(tot, alpha, det_j, points, weights):
    """
    Oblicza wektor P dla podanego boku elementu skończonego.
    
    Parametry:
    - tot: temperatura otoczenia (lub inne obciążenie brzegowe)
    - alpha: współczynnik wymiany ciepła (lub inny odpowiedni parametr fizyczny)
    - det_j: wyznacznik macierzy Jacobiego dla danego boku
    - points: punkty całkowania
    - weights: wagi punktów całkowania
    
    Zwraca:
    - wektor P jako numpy array
    """
    # Inicjalizacja wektora P (dla dwóch węzłów)
    P = np.zeros(2)

    # Pętla po punktach całkowania
    for ksi, w in zip(points, weights):
        N = shape_functions(ksi)  # Funkcje kształtu w punkcie całkowania
        P += w * alpha * tot * N * det_j

    return P

# Parametry dla przykładu
alpha = 25  # Współczynnik wymiany ciepła
tot = 1200  # Temperatura otoczenia
det_j = 0.0125  # Wyznacznik Jacobiego

# Obliczenie wektora P dla różnych ilości punktów całkowania
P_2 = calculate_vector_p(tot, alpha, det_j, gauss_points_2, gauss_weights_2)
P_3 = calculate_vector_p(tot, alpha, det_j, gauss_points_3, gauss_weights_3)
P_4 = calculate_vector_p(tot, alpha, det_j, gauss_points_4, gauss_weights_4)

# Wyniki
print("Wektor P (2 punkty całkowania):", P_2)
print("Wektor P (3 punkty całkowania):", P_3)
print("Wektor P (4 punkty całkowania):", P_4)
