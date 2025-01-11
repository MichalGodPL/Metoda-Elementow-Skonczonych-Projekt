# Ponowne załadowanie kodu z macierzH.py

# Kod wprowadzony do środowiska wykonawczego
import numpy as np

# Define the global variables to store simulation parameters
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


# Element 4-node class definition
class Elem4:
    def __init__(self, nodes):
        self.nodes = np.array(nodes)
        self.gauss_points = [(-1 / np.sqrt(3), -1 / np.sqrt(3)),
                             (1 / np.sqrt(3), -1 / np.sqrt(3)),
                             (1 / np.sqrt(3), 1 / np.sqrt(3)),
                             (-1 / np.sqrt(3), 1 / np.sqrt(3))]
        self.boundary_edges = []  # Przechowywanie indeksów krawędzi brzegowych

    def find_boundary_edges(self, global_nodes, tolerance=1e-6):

        # Granice siatki (min i max współrzędnych)
        min_x = min(coord[0] for coord in global_nodes)
        max_x = max(coord[0] for coord in global_nodes)
        min_y = min(coord[1] for coord in global_nodes)
        max_y = max(coord[1] for coord in global_nodes)

        # Definicje krawędzi elementu (użycie lokalnych indeksów elementu)
        edges = [
            (self.nodes[0], self.nodes[1]),  # Dolna krawędź
            (self.nodes[1], self.nodes[2]),  # Prawa krawędź
            (self.nodes[2], self.nodes[3]),  # Górna krawędź
            (self.nodes[3], self.nodes[0])  # Lewa krawędź
        ]

        # Znajdowanie krawędzi na brzegu siatki
        boundary_edges = []
        for edge_idx, (node1, node2) in enumerate(edges):
            coord1 = node1
            coord2 = node2

            # Sprawdzanie, czy węzły krawędzi leżą na granicy siatki
            if (
                    (abs(coord1[0] - min_x) < tolerance and abs(coord2[0] - min_x) < tolerance) or
                    (abs(coord1[0] - max_x) < tolerance and abs(coord2[0] - max_x) < tolerance) or
                    (abs(coord1[1] - min_y) < tolerance and abs(coord2[1] - min_y) < tolerance) or
                    (abs(coord1[1] - max_y) < tolerance and abs(coord2[1] - max_y) < tolerance)
            ):
                boundary_edges.append(edge_idx)  # Dodaj numer krawędzi

        return boundary_edges

    def pochodne_funkcji_ksztaltu(self, xi, eta):
        dN_dxi = np.array([
            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        ])
        return dN_dxi

    def jacobian(self, dN_dxi):
        J = np.dot(dN_dxi, self.nodes)
        return J

    def jacobian_w_pkt_calkowania(self):
        results = []
        for xi, eta in self.gauss_points:
            dN_dxi = self.pochodne_funkcji_ksztaltu(xi, eta)
            J = self.jacobian(dN_dxi)
            det_J = np.linalg.det(J)
            inv_J = np.linalg.inv(J)
            results.append({
                "jacobi": J,
                "det(J)": det_J,
                "J^(-1)": inv_J
            })
        return results

    # Oblicz lokalną macierz H dla elementu
    def calculate_local_H(self):
        H_local = np.zeros((4, 4))
        global Conductivity

        # Iterate over each Gauss point for numerical integration
        for xi, eta in self.gauss_points:
            # Pochodne funkcji kształtu w punkcie (xi, eta)
            dN_dxi = self.pochodne_funkcji_ksztaltu(xi, eta)

            # Oblicz macierz Jacobiego
            J = self.jacobian(dN_dxi)
            det_J = np.linalg.det(J)
            inv_J = np.linalg.inv(J)

            # Przekształć pochodne funkcji kształtu do przestrzeni globalnej
            dN_dx_dy = np.dot(inv_J, dN_dxi)

            # Macierz B (zawiera pochodne funkcji kształtu względem x i y)
            B = np.zeros((2, 4))
            B[0, :] = dN_dx_dy[0, :]  # dN/dx
            B[1, :] = dN_dx_dy[1, :]  # dN/dy

            # Składnik macierzy H w punkcie całkowania
            H_contrib = Conductivity * (B.T @ B) * det_J

            # Dodaj składnik do macierzy H elementu
            H_local += H_contrib

        return H_local

    def calculate_local_Hbc(self, alfa, tot):
        Hbc_local = np.zeros((4, 4))  # Lokalna macierz Hbc
        gauss_weights = [1, 1]  # Wagi Gaussa dla dwóch punktów
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]  # Punkty Gaussa

        # Przejście po krawędziach (4 krawędzie w elemencie 4-węzłowym)

        for edge in self.boundary_edges:
            ksi_eta_points = []
            edge_nodes = []
            # Zidentyfikuj węzły na krawędzi
            if edge == 0:  # Dolna krawędź (1-2)
                edge_nodes = [0, 1]
                ksi_eta_points = [(gauss, -1) for gauss in gauss_points]
            elif edge == 1:  # Prawa krawędź (2-3)
                edge_nodes = [1, 2]
                ksi_eta_points = [(1, gauss) for gauss in gauss_points]
            elif edge == 2:  # Górna krawędź (3-4)
                edge_nodes = [2, 3]
                ksi_eta_points = [(gauss, 1) for gauss in gauss_points]
            elif edge == 3:  # Lewa krawędź (4-1)
                edge_nodes = [3, 0]
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
                edge_length = np.linalg.norm(self.nodes[edge_nodes[1]] - self.nodes[edge_nodes[0]])
                det_J_edge = edge_length / 2

                # Dodaj składnik do macierzy Hbc
                Hbc_contrib = alfa * (np.outer(N, N) * gauss_weights[i] * det_J_edge)
                Hbc_local += Hbc_contrib

        return Hbc_local


# Function to read and parse the text file
def parse_input_file(filename):
    global SimulationTime, SimulationStepTime, Conductivity, Alfa, Tot, InitialTemp
    global Density, SpecificHeat, NodesNumber, ElementsNumber, Nodes, Elements

    with open(filename, 'r') as file:
        lines = file.readlines()

        # Read parameters and data
        mode = "parameters"  # Track which section we're in: "parameters", "nodes", "elements"
        for line in lines:
            line = line.strip()

            if line.startswith('*Node'):
                mode = "nodes"
                continue
            elif line.startswith('*Element'):
                mode = "elements"
                continue
            elif line.startswith('*BC'):
                break  # Stop reading as we don't need boundary conditions here

            if mode == "parameters":
                # Parse global parameters
                if line.startswith("SimulationTime"):
                    SimulationTime = int(line.split()[1])
                elif line.startswith("SimulationStepTime"):
                    SimulationStepTime = int(line.split()[1])
                elif line.startswith("Conductivity"):
                    Conductivity = float(line.split()[1])
                elif line.startswith("Alfa"):
                    Alfa = int(line.split()[1])
                elif line.startswith("Tot"):
                    Tot = int(line.split()[1])
                elif line.startswith("InitialTemp"):
                    InitialTemp = int(line.split()[1])
                elif line.startswith("Density"):
                    Density = int(line.split()[1])
                elif line.startswith("SpecificHeat"):
                    SpecificHeat = int(line.split()[1])
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

    # Tworzenie obiektów elementów i wyznaczanie krawędzi brzegowych
    global_elements = []
    for elem_nodes in Elements:
        element_coords = [Nodes[i - 1] for i in elem_nodes]
        element = Elem4(element_coords)
        element.boundary_edges = element.find_boundary_edges(Nodes)  # Wyznacz krawędzie brzegowe
        global_elements.append(element)

    return global_elements

def aggregate_local_to_global(elements_objects, nodes_number):
    # Globalna macierz H i Hbc
    H_global = np.zeros((nodes_number, nodes_number))

    # Dodanie alfa i tot (przykładowe wartości, można wczytać z pliku)
    global Alfa, Tot

    # Pętla po elementach
    for elem, elem_nodes in zip(elements_objects, Elements):
        H_local = elem.calculate_local_H()
        Hbc_local = elem.calculate_local_Hbc(Alfa, Tot)

        # Agreguj lokalne macierze H i Hbc do globalnej
        for i, global_i in enumerate(elem_nodes):
            for j, global_j in enumerate(elem_nodes):
                H_global[global_i - 1, global_j - 1] += H_local[i, j] + Hbc_local[i, j]

    return H_global


# Ponowna próba agregacji
filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test2_4_4_MixGrid.txt"


elements_objects = parse_input_file(filename)

H_global = aggregate_local_to_global(elements_objects, NodesNumber)
print(H_global)
