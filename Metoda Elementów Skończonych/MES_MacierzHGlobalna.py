import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

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


    def PochodneFunkcjiKsztaltu(self, xi, eta):

        dN_dxi = np.array([ [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],

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

            Wyniki.append({ "Jakobian": J,  "Det(J)": WyznacznikJakobiego, "J^(-1)": OdwrotnoscJakobiego })

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


# Funkcja Do Odczytu I Analizowania Pliku Tekstowego

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

        "Elements number": ("ElementsNumber", int) }


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
         

def AgregacjaMacierzyZLokalnejDoGlobalnej(ObiektyElementow, nodes_number):

    # Zainicjuj Globalną Macierz H Zerami

    HGlobalne = np.zeros((nodes_number, nodes_number))

    # Iteruj Przez Każdy Element I Jego Lokalną Macierz H

    for elem, elem_nodes in zip(ObiektyElementow, Elements):

        HLokalne = elem.ObliczHLokalne()

        # Agreguj Lokalną Macierz H Do Globalnej Macierzy H

        for i, global_i in enumerate(elem_nodes):

            for j, global_j in enumerate(elem_nodes):

                HGlobalne[global_i - 1, global_j - 1] += HLokalne[i, j]

    return HGlobalne


filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test1_4_4.txt"

AnalizujPlikWejsciowy(filename)

ObiektyElementow = []

for elem_nodes in Elements:

    element_coords = [Nodes[i - 1] for i in elem_nodes]

    element = Element4Wezlowy(element_coords)

    ObiektyElementow.append(element)

HGlobalne = AgregacjaMacierzyZLokalnejDoGlobalnej(ObiektyElementow, NodesNumber)


# Przedstawienie Macierzy H za pomocą Numpy

HGlobalne_np = np.array(HGlobalne)

np.set_printoptions(precision=3, suppress=True)


# Przedstawienie Macierzy H za Pomocą Pandas

HGlobalne_df = pd.DataFrame(HGlobalne_np)


# Wypisz Macierz H

print("Macierz H (Pandas):")

print(HGlobalne_df)

# Wyświetlenie Macierzy H jako Wykres Ciepła

plt.figure(figsize=(10, 8))

plt.imshow(HGlobalne_np, cmap='viridis', interpolation='none')

plt.colorbar(label='Wartość')

plt.title('Macierz H')

plt.xlabel('Indeks Węzła')

plt.ylabel('Indeks Węzła')

plt.show()