import numpy as np

import pandas as pd


# Wczytaj macierz H z pliku CSV

H_globalne = pd.read_csv('C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/HGlobalneZagregowane.csv', header=None).values


# Wczytaj wektor P z pliku CSV

P_globalne = pd.read_csv('C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/WektorPWynikiZagregowany1_4_4.csv', header=None).values.flatten()


# Rozwiąż równanie H{t} + {P} = 0, czyli H{t} = -{P}

t = np.linalg.solve(H_globalne, P_globalne)


# Wydrukuj wynikowy wektor {t}

print("Wektor {t}:")

print(t)