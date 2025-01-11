import numpy as np

import matplotlib.pyplot as plt

from matplotlib.widgets import Button

import random


# Klasa do reprezentacji punktu siatki

class MeshNode:

    def __init__(self, node_id, x_coord, y_coord):

        self.node_id = node_id

        self.x_coord = x_coord

        self.y_coord = y_coord


# Klasa do reprezentacji elementu siatki

class MeshElement:

    def __init__(self, element_id, connected_nodes, temp_value=None):

        self.element_id = element_id

        self.connected_nodes = connected_nodes

        self.temp_value = temp_value


# Klasa siatki, zawierająca węzły i elementy

class FiniteElementMesh:

    def __init__(self, nodes, elements):

        self.mesh_nodes = nodes

        self.mesh_elements = elements


# Klasa do wczytywania danych globalnych z pliku

class SimulationDataLoader:

    def __init__(self, file_name):

        self.simulation_time = None

        self.step_time = None

        self.thermal_conductivity = None

        self.heat_transfer_coeff = None

        self.environment_temp = None

        self.initial_temperature = None

        self.material_density = None

        self.specific_heat_capacity = None

        self.nodes = []

        self.elements = []

        self.boundary_conditions = []

        self._parse_file(file_name)


    def _parse_file(self, file_name):

        with open(file_name, 'r') as file:

            content_lines = file.readlines()

            in_nodes_section = False

            in_elements_section = False


            for line in content_lines:

                if line.startswith("SimulationTime"):

                    self.simulation_time = float(line.split()[1])

                elif line.startswith("SimulationStepTime"):

                    self.step_time = float(line.split()[1])

                elif line.startswith("Conductivity"):

                    self.thermal_conductivity = float(line.split()[1])

                elif line.startswith("Alfa"):

                    self.heat_transfer_coeff = float(line.split()[1])

                elif line.startswith("Tot"):

                    self.environment_temp = float(line.split()[1])

                elif line.startswith("InitialTemp"):

                    self.initial_temperature = float(line.split()[1])

                elif line.startswith("Density"):

                    self.material_density = float(line.split()[1])

                elif line.startswith("SpecificHeat"):

                    self.specific_heat_capacity = float(line.split()[1])

                elif line.startswith("*Node"):

                    in_nodes_section = True

                    in_elements_section = False

                elif line.startswith("*Element"):

                    in_nodes_section = False

                    in_elements_section = True

                elif line.startswith("*BC"):

                    in_nodes_section = False

                    in_elements_section = False

                    self.boundary_conditions = list(map(int, line.split(",")[1:]))

                elif in_nodes_section:

                    data_parts = list(map(float, line.split(",")))

                    self.nodes.append(MeshNode(int(data_parts[0]), data_parts[1], data_parts[2]))

                elif in_elements_section:

                    data_parts = list(map(int, line.split(",")))

                    temperature = random.uniform(20, 100)

                    self.elements.append(MeshElement(data_parts[0], data_parts[1:], temperature))


# Funkcja do wyświetlania siatki MES

def draw_mesh(ax, mesh, show_node_labels=True, show_element_labels=True):

    ax.clear()


    # Ustal kolor dla temperatury

    color_map = plt.get_cmap('coolwarm')

    temperature_norm = plt.Normalize(20, 100)


    # Określ czcionkę i krok etykiety

    num_nodes = len(mesh.mesh_nodes)

    font_scale = max(7, 12 - int(num_nodes / 100))
    
    label_interval = max(1, num_nodes // 100)


    # Rysuj Elementy z Wypełnieniem

    for element in mesh.mesh_elements:

        polygon_points = np.array([[mesh.mesh_nodes[node_id - 1].x_coord, mesh.mesh_nodes[node_id - 1].y_coord] 

                                   for node_id in element.connected_nodes])

        element_color = color_map(temperature_norm(element.temp_value))

        ax.fill(polygon_points[:, 0], polygon_points[:, 1], color=element_color, edgecolor='black')


        # Opcjonalnie: Dodaj Etykiety Elementów

        if show_element_labels and element.element_id % label_interval == 0:

            center_x = np.mean(polygon_points[:, 0])

            center_y = np.mean(polygon_points[:, 1])

            ax.text(center_x, center_y, f'E{element.element_id}', color='red', fontsize=font_scale, 

                    bbox=dict(facecolor='white', alpha=0.5))


    # Opcjonalnie: Dodaj Etykiety Węzłów

    if show_node_labels:

        for node in mesh.mesh_nodes:

            if node.node_id % label_interval == 0:

                ax.plot(node.x_coord, node.y_coord, 'bo')

                ax.text(node.x_coord, node.y_coord, f'N{node.node_id}', fontsize=font_scale, ha='right',

                        bbox=dict(facecolor='white', alpha=0.5))


    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_aspect('equal', 'box')

    plt.draw()


# Funkcja do Obsługi Wyświetlania z Przyciskami

def interactive_mesh_view(mesh):

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.05, right=0.75)


    show_nodes = True

    show_elements = True

    draw_mesh(ax, mesh, show_nodes, show_elements)


    # Przyciski do przełączania etykiet

    node_button_ax = plt.axes([0.8, 0.05, 0.15, 0.075])

    element_button_ax = plt.axes([0.8, 0.15, 0.15, 0.075])


    node_button = Button(node_button_ax, 'Toggle Nodes')

    element_button = Button(element_button_ax, 'Toggle Elements')


    # Funkcje do obsługi kliknięcia przycisków

    def toggle_node_display(event):

        nonlocal show_nodes

        show_nodes = not show_nodes

        draw_mesh(ax, mesh, show_nodes, show_elements)


    def toggle_element_display(event):

        nonlocal show_elements

        show_elements = not show_elements

        draw_mesh(ax, mesh, show_nodes, show_elements)

    node_button.on_clicked(toggle_node_display)

    element_button.on_clicked(toggle_element_display)

    plt.show()


# Wczytanie danych i wyświetlenie siatki

data_loader = SimulationDataLoader("C:\\Users\\Admin\\Desktop\\Metoda Elementów Skończonych\\Pliki Tekstowe\\Test2_4_4_MixGrid.txt")

mesh = FiniteElementMesh(data_loader.nodes, data_loader.elements)

interactive_mesh_view(mesh)
