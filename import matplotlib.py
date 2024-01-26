import matplotlib.pyplot as plt

# La tua lista di coordinate
coordinate = [(0, 0), (0, 1), (1.2, 3), (0.5, 5)]

# Estrai le coordinate x e y separatamente
x, y = zip(*coordinate)

# Crea il plot
plt.plot(x, y, marker='o', linestyle='-')

# Aggiungi etichette agli assi e un titolo al grafico
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot delle coordinate')

# Mostra il grafico
plt.show()
