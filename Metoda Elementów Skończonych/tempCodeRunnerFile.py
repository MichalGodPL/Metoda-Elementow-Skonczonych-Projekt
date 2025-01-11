def AnalizujPlikWejsciowy(filename):
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
            break  # Przestań Czytać, Ponieważ Nie Potrzebujemy Tutaj Warunków Brzegowych

        if mode == "parameters":
            # Analizowanie Globalnych Parametrów
            parts = line.split()
            if not parts:
                continue
            key, value = parts[0], parts[1]
            if key == "SimulationTime":
                SimulationTime = int(value)
            elif key == "SimulationStepTime":
                SimulationStepTime = int(value)
            elif key == "Conductivity":
                Conductivity = float(value)
            elif key == "Alfa":
                Alfa = float(value)
            elif key == "Tot":
                Tot = float(value)
            elif key == "InitialTemp":
                InitialTemp = float(value)
            elif key == "Density":
                Density = float(value)
            elif key == "SpecificHeat":
                SpecificHeat = float(value)
            elif key == "Nodes":
                NodesNumber = int(parts[2])
            elif key == "Elements":
                ElementsNumber = int(parts[2])

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
            node_ids = list(map(int, map(str.strip, parts[1:])))
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