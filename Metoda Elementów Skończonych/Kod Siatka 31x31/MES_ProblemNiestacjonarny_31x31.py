import numpy as np

import pandas as pd

import os

from MES_EliminacjaGaussa_31x31 import gauss_elimination


# Wczytaj wartości z pliku tekstowego

def AnalizujPlikWejsciowy(filename):

    global SimulationTime, SimulationStepTime

    param_map = {

        "SimulationTime": ("SimulationTime", int),

        "SimulationStepTime": ("SimulationStepTime", int),

        "InitialTemp": ("InitialTemp", int),

        "Nodesnumber": ("Nodesnumber", int)

    }

    with open(filename, 'r') as file:

        lines = file.readlines()

        for line in lines:

            line = line.strip()

            for key, (var_name, var_type) in param_map.items():

                if line.startswith(key):

                    globals()[var_name] = var_type(line.split()[1])

                    break


# Ścieżki do plików

H_file_path = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/MacierzHGlobalna_Zagregowana_Wyniki_31x31.csv'

C_file_path = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/MacierzCGlobalna_Zagregowana_Wyniki_31x31.csv'

P_file_path = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/WektorP_Zagregowany_Wyniki_31x31.csv'

input_file_path = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe/Pliki Tekstowe Siatka 31x31/Test3_31_31_kwadrat.txt'


# Wczytaj wartości z pliku tekstowego

AnalizujPlikWejsciowy(input_file_path)


# Wczytaj macierz H z pliku CSV

H_globalne = pd.read_csv(H_file_path, header=None).values


# Wczytaj macierz C z pliku CSV

C_globalne = pd.read_csv(C_file_path, header=None).values


# Wczytaj wektor P z pliku CSV

P_globalne = pd.read_csv(P_file_path, header=None).values  # Wczytaj jako macierz numpy


# Upewnij się, że ma wymiary 16x1

if P_globalne.shape[1] != 1:  # Jeśli P jest w jednym wierszu, przekształć

    P_globalne = P_globalne.reshape(-1, 1)


# Wartości z pliku tekstowego

Delta_tau = SimulationStepTime

CzasSymulacji = SimulationTime

TemperaturaPoczatkowa = InitialTemp


t0 = np.full((Nodesnumber, 1), TemperaturaPoczatkowa ) # W plikach tekstowych zmień żeby było tak jak tutaj

obecny_czas = 0.0

while (obecny_czas<CzasSymulacji):


    H_plus_DeltaTau_C = H_globalne + (C_globalne/ Delta_tau)

    DeltaTau_C_t0 = P_globalne+((C_globalne/ Delta_tau) @ t0)

    t1 = gauss_elimination(H_plus_DeltaTau_C, DeltaTau_C_t0)

    print(f"Obecny czas: {obecny_czas + Delta_tau}")

    print(np.min(t1), np.max(t1), "\n")

    obecny_czas += Delta_tau

    t0 = t1