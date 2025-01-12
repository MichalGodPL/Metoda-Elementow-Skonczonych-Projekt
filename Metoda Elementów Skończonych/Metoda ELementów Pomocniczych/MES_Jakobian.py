import numpy as np

import pandas as pd

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


class Elementy4:

    def __init__(self, nodes):

        self.nodes = np.array(nodes)

        self.gauss_points = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]


    def PochodneKsztaltu(self, xi, eta):

        dN_dxi = np.array([ [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)], [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]])

        return dN_dxi


    def Jacobian(self, dN_dxi):

        J = np.dot(dN_dxi, self.nodes)
        
        return J

    def JacobianPunktCalkowania(self):

        results = []

        for xi, eta in self.gauss_points:

            dN_dxi = self.PochodneKsztaltu(xi, eta)

            J = self.Jacobian(dN_dxi)

            det_J = np.linalg.det(J)

            inv_J = np.linalg.inv(J)

            results.append({ "jacobi": J, "det(J)": det_J,"J^(-1)": inv_J })
            
        return results

 
    def HLokalne(self):

        H_local = np.zeros((4, 4))

        global Conductivity

        for xi, eta in self.gauss_points:


            dN_dxi = self.PochodneKsztaltu(xi, eta)

            J = self.Jacobian(dN_dxi)

            det_J = np.linalg.det(J)

            inv_J = np.linalg.inv(J)

            dN_dx_dy = np.dot(inv_J, dN_dxi)


            B = np.zeros((2, 4))

            B[0, :] = dN_dx_dy[0, :]  # dN/dx

            B[1, :] = dN_dx_dy[1, :]  # dN/dy


            H_contrib = Conductivity * (B.T @ B) * det_J

            H_local += H_contrib

        return H_local


def AnalizaWejsciowego(filename):

    global SimulationTime, SimulationStepTime, Conductivity, Alfa, Tot, InitialTemp

    global Density, SpecificHeat, NodesNumber, ElementsNumber, Nodes, Elements

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

                break


            if mode == "parameters":

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


filename = r"C:\Users\Admin\Desktop\Metoda Elementów Skończonych\Pliki Tekstowe\Test1_4_4.txt"

AnalizaWejsciowego(filename)


elements_objects = []

for elem_nodes in Elements:

    element_coords = [Nodes[i - 1] for i in elem_nodes]

    element = Elementy4(element_coords)

    elements_objects.append(element)


# Prezentacja Wyników Jacobiego Za Pomocą Pandas

for idx, element in enumerate(elements_objects, start=1):

    print(f"Element {idx} Jacobian Results:")

    jacobian_results = element.HLokalne()

    print(jacobian_results)

    print()