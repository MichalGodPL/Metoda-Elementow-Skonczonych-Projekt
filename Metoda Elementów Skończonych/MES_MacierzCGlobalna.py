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

        self.nodes = np.array(nodes, dtype=float)

        self.PunktyGaussa = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]


    def PochodneFunkcjiKsztaltu(self, xi, eta):

        dN_dxi = np.array([

            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],

            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)] ])

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


    # Oblicz lokalną Macierz C dla Elementu

    def ObliczCLokalne(self):

        CLokalne = np.zeros((4, 4))

        global Density, SpecificHeat

        for xi, eta in self.PunktyGaussa:

            dN_dxi = self.PochodneFunkcjiKsztaltu(xi, eta)

            J = self.Jakobian(dN_dxi)

            WyznacznikJakobiego = np.linalg.det(J)

            N = self.FunkcjeKsztaltu(xi, eta)

            N_T = N.reshape(-1, 1)

            CSkladnik = Density * SpecificHeat * np.outer(N, N_T) * WyznacznikJakobiego

            CLokalne += CSkladnik

        return CLokalne


    def FunkcjeKsztaltu(self, xi, eta):

        N1 = 0.25 * (1 - xi) * (1 - eta)

        N2 = 0.25 * (1 + xi) * (1 - eta)

        N3 = 0.25 * (1 + xi) * (1 + eta)

        N4 = 0.25 * (1 - xi) * (1 + eta)

        return np.array([N1, N2, N3, N4])


# Funkcja Do Odczytu I Analizowania Pliku Tekstowego

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


def AgregacjaMacierzyC(ObiektyElementow, nodes_number):

    CGlobalne = np.zeros((nodes_number, nodes_number))

    for elem, elem_nodes in zip(ObiektyElementow, Elements):

        CLokalne = elem.ObliczCLokalne()

        for i, global_i in enumerate(elem_nodes):

            for j, global_j in enumerate(elem_nodes):

                CGlobalne[global_i - 1, global_j - 1] += CLokalne[i, j]

    return CGlobalne


filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test1_4_4.txt"

AnalizujPlikWejsciowy(filename)


ObiektyElementow = []

for elem_nodes in Elements:

    element_coords = [Nodes[i - 1] for i in elem_nodes]

    element = Element4Wezlowy(element_coords)

    ObiektyElementow.append(element)


CGlobalne = AgregacjaMacierzyC(ObiektyElementow, NodesNumber)


# Przedstawienie Macierzy C za pomocą Numpy

CGlobalne_np = np.array(CGlobalne)

np.set_printoptions(precision=3, suppress=True)


# Przedstawienie Macierzy C za Pomocą Pandas

CGlobalne_df = pd.DataFrame(CGlobalne_np)


# Wypisz Macierz C

print("Macierz C (Pandas):")

print(CGlobalne_df)


# Wyświetlenie Macierzy C jako Wykres Ciepła

plt.figure(figsize=(10, 8))

plt.imshow(CGlobalne_np, cmap='viridis', interpolation='none')

plt.colorbar(label='Wartość')

plt.title('Macierz C')

plt.xlabel('Indeks Węzła')

plt.ylabel('Indeks Węzła')

plt.show()


output_folder = 'C:/Users/Admin/Documents/GitHub/MetodaEliminacjiStudentow/Metoda Elementów Skończonych/Pliki Tekstowe'

output_filename_C = 'MacierzCGlobalnaWyniki.csv'

output_path_C = os.path.join(output_folder, output_filename_C)


# Upewnij się, że Folder Istnieje

os.makedirs(output_folder, exist_ok=True)


# Zapisz DataFrame do pliku CSV

CGlobalne_df.to_csv(output_path_C, index=False, header=False)

print(f"Macierz C została zapisana do pliku: {output_path_C}")