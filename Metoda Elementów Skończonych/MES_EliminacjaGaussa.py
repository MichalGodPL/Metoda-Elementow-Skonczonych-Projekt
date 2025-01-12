import numpy as np

def gauss_elimination(A, b):

    """ Rozwiązuje układ równań liniowych Ax = b metodą eliminacji Gaussa.

    Parametry:

        A (numpy.ndarray): Macierz współczynników (n x n).

        b (numpy.ndarray): Wektor wyrazów wolnych (n x 1 lub n,).

    Zwraca:

        numpy.ndarray: Wektor rozwiązania x. """

    n = len(b)

    # Tworzymy rozszerzoną macierz [A|b]

    Ab = np.hstack((A.astype(float), b.reshape(-1, 1)))

    # Etap eliminacji

    for i in range(n):

        # Szukamy wiersza z maksymalnym elementem w kolumnie i, aby uniknąć dzielenia przez zero

        max_row = np.argmax(np.abs(Ab[i:, i])) + i

        if Ab[max_row, i] == 0:

            raise ValueError("Układ równań nie ma jednoznacznego rozwiązania (macierz osobliwa).")
        

        # Zamieniamy wiersze, jeśli trzeba

        if max_row != i:

            Ab[[i, max_row]] = Ab[[max_row, i]]


        # Normalizujemy wiersz

        Ab[i] = Ab[i] / Ab[i, i]


        # Eliminujemy elementy poniżej

        for j in range(i + 1, n):

            Ab[j] -= Ab[j, i] * Ab[i]


    # Etap podstawienia wstecznego
    
    x = np.zeros((n, 1))

    for i in range(n - 1, -1, -1):

        x[i] = (Ab[i, -1]-np.dot(Ab[i, i+1:n], x[i+1:n]))/Ab[i, i]

    return x
