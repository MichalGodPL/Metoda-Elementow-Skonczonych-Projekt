import numpy as np

import pandas as pd


# Wczytaj macierz H z pliku CSV

H_globalne = pd.read_csv('C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/MacierzHGlobalna_Zagregowana_Wyniki_31x31.csv', header=None).values


# Wczytaj wektor P z pliku CSV

P_globalne = pd.read_csv('C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/WektorP_Zagregowany_Wyniki_31x31.csv', header=None).values.flatten()


# Rozwiąż równanie H{t} + {P} = 0, czyli H{t} = -{P}

t = np.linalg.solve(H_globalne, P_globalne)


# Wydrukuj wynikowy wektor {t}

print("Wektor {t}:")

print(t)