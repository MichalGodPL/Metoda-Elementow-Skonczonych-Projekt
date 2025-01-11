import numpy as np

import matplotlib.pyplot as plt


def gauss_legendre_integration(func, points):

    """ Funkcja wykonująca całkowanie numeryczne w układzie (-1, 1), przy użyciu schematów 2-, 3- i 4-punktowych Gaussa-Legendre'a.

    :param func: Funkcja 2D do całkowania  :param points: Liczba punktów (2, 3 lub 4) w schemacie całkowania

    :return: Wynik całkowania.

    """

    print(f"Całkowana funkcja: {func.__doc__.strip() if func.__doc__ else func}\n")
    
    if points == 2:

        # Wagi i punkty dla schematu 2-punktowego

        weights = [1.0, 1.0]

        nodes = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    elif points == 3:

        # Wagi i punkty dla schematu 3-punktowego

        weights = [5/9, 8/9, 5/9]

        nodes = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]

    elif points == 4:

        # Wagi i punkty dla schematu 4-punktowego

        weights = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]

        nodes = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]

    else:

        raise ValueError("Tylko schematy 2, 3 i 4-punktowe są obsługiwane.")


    integral = 0

    node_values = []


    for weight, node in zip(weights, nodes):

        value = weight * func(node)

        integral += value

        node_values.append((node, func(node)))

        print(f"Punkt: {node}, Waga: {weight}, Wartość funkcji: {func(node)}\n")

    print(f"Całkowity wynik całkowania: {integral}\n")


    # Rysowanie wykresu

    plot_integration(func, nodes, weights, node_values, points)
    
    return integral


def plot_integration(func, nodes, weights, node_values, points):

    """ Funkcja rysująca wykres funkcji i obszarów odpowiadających całkowaniu """

    # Generowanie punktów do wykresu funkcji

    x = np.linspace(-1, 1, 500)

    y = func(x)


    # Tworzenie wykresu

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label=f'Funkcja: {func.__doc__.strip() if func.__doc__ else "Brak opisu"}', color='blue')


    # Zaznaczanie punktów całkowania

    plt.scatter(nodes, [func(n) for n in nodes], color='red', label=f'Punkty {points}-punktowe Gaussa')


    # Rysowanie obszarów odpowiadających wagom

    for node, weight in zip(nodes, weights):

        plt.fill_between([node - 0.1, node + 0.1], 0, [func(node)] * 2, alpha=0.3, label=f'Waga {weight}', color='orange')


    # Dodatkowe ustawienia wykresu

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.title(f'Schemat {points}-punktowy Gaussa-Legendre\'a')

    plt.xlabel('x')

    plt.ylabel('f(x)')

    plt.legend()

    plt.grid(True)

    plt.show()


# Przykład użycia

def example_function(x):

    """ Przykładowa funkcja do całkowania: x^2 """

    return x**2


# Całkowanie 2-punktowe

result_2_points = gauss_legendre_integration(example_function, points=2)

print(f"Wynik całkowania 2-punktowego: {result_2_points}\n\n")


# Całkowanie 3-punktowe

result_3_points = gauss_legendre_integration(example_function, points=3)

print(f"Wynik całkowania 3-punktowego: {result_3_points}\n\n")


# Całkowanie 4-punktowe

result_4_points = gauss_legendre_integration(example_function, points=4)

print(f"Wynik całkowania 4-punktowego: {result_4_points}\n\n")