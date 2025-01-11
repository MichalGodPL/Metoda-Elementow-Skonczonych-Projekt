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

def AnalizujPlikWejsciowy(filename):

    global SimulationTime, SimulationStepTime, Conductivity, Alfa, Tot, InitialTemp

    global Density, SpecificHeat, NodesNumber, ElementsNumber, Nodes, Elements

    with open(filename, 'r') as file:

        lines = file.readlines()

        # Śledź Którą Sekcję Przetwarzamy: "Parametry", "Węzły", "Elementy"

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

                # Analizowanie Globalnych Parametrów

                if line.startswith("SimulationTime"):

                    SimulationTime = int(line.split()[1])

                elif line.startswith("SimulationStepTime"):

                    SimulationStepTime = int(line.split()[1])

                elif line.startswith("Conductivity"):

                    Conductivity = float(line.split()[1])

                elif line.startswith("Alfa"):

                    Alfa = float(line.split()[1])

                elif line.startswith("Tot"):

                    Tot = float(line.split()[1])

                elif line.startswith("InitialTemp"):

                    InitialTemp = float(line.split()[1])

                elif line.startswith("Density"):

                    Density = float(line.split()[1])

                elif line.startswith("SpecificHeat"):

                    SpecificHeat = float(line.split()[1])

                elif line.startswith("Nodes number"):

                    NodesNumber = int(line.split()[2])

                elif line.startswith("Elements number"):

                    ElementsNumber = int(line.split()[2])


            elif mode == "nodes":

                # Parse node coordinates

                parts = line.split(',')

                node_id = int(parts[0].strip())

                x = float(parts[1].strip())

                y = float(parts[2].strip())
                
                Nodes.append((x, y))


            elif mode == "elements":

                # Parse element connectivity

                parts = line.split(',')
                
                element_id = int(parts[0].strip())

                node_ids = list(map(lambda n: int(n.strip()), parts[1:]))

                Elements.append(node_ids)


def AgregacjaMacierzyZLokalnejDoGlobalnej(elements_objects, nodes_number):

    # Initialize the global H matrix with zeros
    H_global = np.zeros((nodes_number, nodes_number))

    # Loop through each element and its local H matrix
    for elem, elem_nodes in zip(elements_objects, Elements):
        HLokalne = elem.ObliczHLokalne()

        # Aggregate local H into the global H matrix
        for i, global_i in enumerate(elem_nodes):
            for j, global_j in enumerate(elem_nodes):
                H_global[global_i - 1, global_j - 1] += HLokalne[i, j]

    return H_global


filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test1_4_4.txt"

AnalizujPlikWejsciowy(filename)

elements_objects = []
for elem_nodes in Elements:
    element_coords = [Nodes[i - 1] for i in elem_nodes]
    element = Element4Wezlowy(element_coords)
    elements_objects.append(element)

H_global = AgregacjaMacierzyZLokalnejDoGlobalnej(elements_objects, NodesNumber)

# Przedstawienie macierzy H za pomocą numpy
H_global_np = np.array(H_global)

# Ustaw opcje wyświetlania numpy
np.set_printoptions(precision=3, suppress=True)

# Przedstawienie macierzy H za pomocą pandas
H_global_df = pd.DataFrame(H_global_np)

# Wypisz macierz H w czytelny sposób za pomocą pandas
print("Macierz H (pandas):")
print(H_global_df)

# Wyświetlenie macierzy H jako wykres ciepła za pomocą matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(H_global_np, cmap='viridis', interpolation='none')
plt.colorbar(label='Value')
plt.title('Macierz H')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()