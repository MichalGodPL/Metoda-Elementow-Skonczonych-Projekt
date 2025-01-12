import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Zdefiniuj Globalne Zmienne Do Przechowywania Parametrów Symulacji
SimulationTime = None
SimulationStepTime = None
Conductivity = None
Alfa = None
Tot = None
InitialTemp = None
Density = None
SpecificHeat = None
NodesNumber = None
ElementsNumber = None
Nodes = []
Elements = []

# Definicja Klasy Elementu 4-Węzłowego
class Element4Wezlowy:
    def __init__(self, nodes):
        # Przechowuj Współrzędne Węzłów Elementu (X, Y)
        self.nodes = np.array(nodes, dtype=float)
        # Punkty Gaussa Dla Kwadratury 2x2
        self.PunktyGaussa = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
        self.KrawedzieBrzegowe = []  # Przechowywanie indeksów krawędzi brzegowych

    def ZnajdzKrawedzieBrzegowe(self, global_nodes, tolerance=1e-6):
        # Granice siatki (min i max współrzędnych)
        min_x = min(coord[0] for coord in global_nodes)
        max_x = max(coord[0] for coord in global_nodes)
        min_y = min(coord[1] for coord in global_nodes)
        max_y = max(coord[1] for coord in global_nodes)

        # Definicje krawędzi elementu (użycie lokalnych indeksów elementu)
        Krawedzie = [
            (self.nodes[0], self.nodes[1]),  # Dolna krawędź
            (self.nodes[1], self.nodes[2]),  # Prawa krawędź
            (self.nodes[2], self.nodes[3]),  # Górna krawędź
            (self.nodes[3], self.nodes[0])  # Lewa krawędź
        ]

        # Znajdowanie krawędzi na brzegu siatki
        KrawedzieBrzegowe = []
        for edge_idx, (node1, node2) in enumerate(Krawedzie):
            coord1 = node1
            coord2 = node2
            # Sprawdzanie, czy węzły krawędzi leżą na granicy siatki
            if (
                (abs(coord1[0] - min_x) < tolerance and abs(coord2[0] - min_x) < tolerance) or
                (abs(coord1[0] - max_x) < tolerance and abs(coord2[0] - max_x) < tolerance) or
                (abs(coord1[1] - min_y) < tolerance and abs(coord2[1] - min_y) < tolerance) or
                (abs(coord1[1] - max_y) < tolerance and abs(coord2[1] - max_y) < tolerance)
            ):
                KrawedzieBrzegowe.append(edge_idx)  # Dodaj numer krawędzi

        return KrawedzieBrzegowe

    def PochodneFunkcjiKsztaltu(self, xi, eta):
        dN_dxi = np.array([[-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
                           [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]])
        return dN_dxi

    def Jakobian(self, dN_dxi):
        J = np.dot(dN_dxi, self.nodes)
        return J

    def JakobianWPunktachCalkowania(self):
        Wyniki = []
        for xi, eta in self.PunktyGaussa:
            dN_dxi = self.PochodneFunkcjiKsztaltu(xi, eta)
            J = self.Jakobian(dN_dxi)
            WyznacznikJakobiego = np.linalg.det(J)
            OdwrotnoscJakobiego = np.linalg.inv(J)
            Wyniki.append({"Jakobian": J, "Det(J)": WyznacznikJakobiego, "J^(-1)": OdwrotnoscJakobiego})
        return Wyniki

    # Oblicz lokalną Macierz H dla Elementu
    def ObliczHLokalne(self):
        HLokalne = np.zeros((4, 4))
        global Conductivity

        # Iteruj Przez Każdy Punkt Gaussa Dla Całkowania Numerycznego
        for xi, eta in self.PunktyGaussa:
            # Pochodne Funkcji Kształtu w Punkcie (xi, eta)
            dN_dxi = self.PochodneFunkcjiKsztaltu(xi, eta)

            # Oblicz Macierz Jakobiego
            J = self.Jakobian(dN_dxi)
            WyznacznikJakobiego = np.linalg.det(J)
            OdwrotnoscJakobiego = np.linalg.inv(J)

            # Przekształć Pochodne Funkcji Kształtu do Przestrzeni Globalnej
            dN_dx_dy = np.dot(OdwrotnoscJakobiego, dN_dxi)

            # Macierz B Zawierająca Pochodne Funkcji Kształtu Wzglęgem X i Y
            B = np.zeros((2, 4))
            B[0, :] = dN_dx_dy[0, :]  # dN/dx
            B[1, :] = dN_dx_dy[1, :]  # dN/dy

            # Składnik Macierzy H w Punkcie Całkowania
            HSkladnik = Conductivity * np.dot(B.T, B) * WyznacznikJakobiego

            # Dodaj Składnik do Macierzy H Elementu
            HLokalne += HSkladnik

        return HLokalne

    # Oblicz lokalną Macierz C dla Elementu
    def ObliczCLokalne(self):
        CLokalne = np.zeros((4, 4))
        global Density, SpecificHeat

        # Iteruj Przez Każdy Punkt Gaussa Dla Całkowania Numerycznego
        for xi, eta in self.PunktyGaussa:
            # Funkcje Kształtu w Punkcie (xi, eta)
            N = np.array([
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta)
            ])

            # Pochodne Funkcji Kształtu w Punkcie (xi, eta)
            dN_dxi = self.PochodneFunkcjiKsztaltu(xi, eta)

            # Oblicz Macierz Jakobiego
            J = self.Jakobian(dN_dxi)
            WyznacznikJakobiego = np.linalg.det(J)

            # Składnik Macierzy C w Punkcie Całkowania
            CSkladnik = Density * SpecificHeat * np.outer(N, N) * WyznacznikJakobiego

            # Dodaj Składnik do Macierzy C Elementu
            CLokalne += CSkladnik

        return CLokalne

    def ObliczLokalneHbc(self, alfa, tot):
        HbcLokalne = np.zeros((4, 4))  # Lokalna macierz Hbc
        GaussWagi = [1, 1]  # Wagi Gaussa dla dwóch punktów
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]  # Punkty Gaussa

        # Przejście po krawędziach (4 krawędzie w elemencie 4-węzłowym)
        for edge in self.KrawedzieBrzegowe:
            ksi_eta_points = []
            WezlyKrawedzi = []

            # Zidentyfikuj Węzły na Krawędzi
            if edge == 0:  # Dolna krawędź (1-2)
                WezlyKrawedzi = [0, 1]
                ksi_eta_points = [(gauss, -1) for gauss in gauss_points]
            elif edge == 1:  # Prawa krawędź (2-3)
                WezlyKrawedzi = [1, 2]
                ksi_eta_points = [(1, gauss) for gauss in gauss_points]
            elif edge == 2:  # Górna krawędź (3-4)
                WezlyKrawedzi = [2, 3]
                ksi_eta_points = [(gauss, 1) for gauss in gauss_points]
            elif edge == 3:  # Lewa krawędź (4-1)
                WezlyKrawedzi = [3, 0]
                ksi_eta_points = [(-1, gauss) for gauss in gauss_points]

            # Przejście przez punkty całkowania na krawędzi
            for i, (ksi, eta) in enumerate(ksi_eta_points):
                # Oblicz funkcje kształtu w punkcie (ksi, eta)
                N = np.array([
                    0.25 * (1 - ksi) * (1 - eta),
                    0.25 * (1 + ksi) * (1 - eta),
                    0.25 * (1 + ksi) * (1 + eta),
                    0.25 * (1 - ksi) * (1 + eta)
                ])
                # Długość krawędzi (elementarny Jakobian dla 1D)
                DlugoscKrawedzi = np.linalg.norm(self.nodes[WezlyKrawedzi[1]] - self.nodes[WezlyKrawedzi[0]])
                WyznacznikJakobianaKrawedzi = DlugoscKrawedzi / 2

                # Dodaj składnik do macierzy Hbc
                SkladnikHbc = alfa * (np.outer(N, N) * GaussWagi[i] * WyznacznikJakobianaKrawedzi)
                HbcLokalne += SkladnikHbc

        return HbcLokalne

def AnalizujPlikWejsciowy(filename):
    global SimulationTime, SimulationStepTime, Conductivity, Alfa, Tot, InitialTemp
    global Density, SpecificHeat, NodesNumber, ElementsNumber, Nodes, Elements

    param_map = {
        "SimulationTime": ("SimulationTime", int),
        "SimulationStepTime": ("SimulationStepTime", int),
        "Conductivity": ("Conductivity", float),
        "Alfa": ("Alfa", float),
        "Tot": ("Tot", float),
        "InitialTemp": ("InitialTemp", float),
        "Density": ("Density", float),
        "SpecificHeat": ("SpecificHeat", float),
        "Nodes number": ("NodesNumber", int),
        "Elements number": ("ElementsNumber", int)
    }

    with open(filename, 'r') as file:
        lines = file.readlines()
        mode = "parameters"
        for line in lines:
            line = line.strip()
            if line.startswith('*Node'):
                mode = "nodes"
                continue
            elif line.startswith('*Element'):
                mode = "elements"
                continue
            elif line.startswith('*BC'):
                break  # Przestań Czytać, Ponieważ Nie Potrzebujemy Tutaj Warunków Brzegowych

            if mode == "parameters":
                for key, (var_name, var_type) in param_map.items():
                    if line.startswith(key):
                        if key in ["Nodes number", "Elements number"]:
                            globals()[var_name] = var_type(line.split()[2])
                        else:
                            globals()[var_name] = var_type(line.split()[1])
                        break
            elif mode == "nodes":
                parts = line.split(',')
                node_id = int(parts[0].strip())
                x = float(parts[1].strip())
                y = float(parts[2].strip())
                Nodes.append((x, y))
            elif mode == "elements":
                parts = line.split(',')
                element_id = int(parts[0].strip())
                node_ids = list(map(lambda n: int(n.strip()), parts[1:]))
                Elements.append(node_ids)

# Ponowna próba agregacji
filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test2_4_4_MixGrid.txt"
AnalizujPlikWejsciowy(filename)

ObiektyElementow = []
for elem_nodes in Elements:
    element_coords = [Nodes[i - 1] for i in elem_nodes]
    element = Element4Wezlowy(element_coords)
    element.KrawedzieBrzegowe = element.ZnajdzKrawedzieBrzegowe(Nodes)
    ObiektyElementow.append(element)

# Zainicjuj globalną macierz H zerami
H_globalne = np.zeros((NodesNumber, NodesNumber))

# Zainicjuj globalną macierz C zerami
C_globalne = np.zeros((NodesNumber, NodesNumber))

# Iteruj przez każdy element i jego lokalne macierze Hbc, H i C
for element, elem_nodes in zip(ObiektyElementow, Elements):
    Hbc_local = element.ObliczLokalneHbc(Alfa, Tot)
    H_localne = element.ObliczHLokalne()
    C_localne = element.ObliczCLokalne()

    # Dodaj wartości z lokalnych macierzy do odpowiednich miejsc w globalnej macierzy H i C
    for i, global_i in enumerate(elem_nodes):
        for j, global_j in enumerate(elem_nodes):
            H_globalne[global_i - 1, global_j - 1] += Hbc_local[i, j] + H_localne[i, j]
            C_globalne[global_i - 1, global_j - 1] += C_localne[i, j]

# Wydrukuj zagregowaną globalną macierz H
print("H Globalne Zagregowane")
print(pd.DataFrame(H_globalne))

# Wydrukuj zagregowaną globalną macierz C
print("C Globalne Zagregowane")
print(pd.DataFrame(C_globalne))

# Zapisz zagregowaną globalną macierz H do pliku CSV
output_folder = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe'
output_filename_H = 'HGlobalneZagregowane.csv'
output_path_H = os.path.join(output_folder, output_filename_H)

# Ensure the folder exists
os.makedirs(output_folder, exist_ok=True)

# Save the DataFrame to a CSV file
pd.DataFrame(H_globalne).to_csv(output_path_H, index=False, header=False)
print(f"Zagregowana globalna macierz H została zapisana do pliku: {output_path_H}")

# Zapisz zagregowaną globalną macierz C do pliku CSV
output_filename_C = 'CGlobalneZagregowane.csv'
output_path_C = os.path.join(output_folder, output_filename_C)

# Save the DataFrame to a CSV file
pd.DataFrame(C_globalne).to_csv(output_path_C, index=False, header=False)
print(f"Zagregowana globalna macierz C została zapisana do pliku: {output_path_C}")