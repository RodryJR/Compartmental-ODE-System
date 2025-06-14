import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import random
from src.fitness import *


class GeneticGraphSearch:
    def __init__(self, fitness_evaluator, num_generations=50, population_size=30, mutation_rate=0.2, crossover_rate=0.5):
        self.fitness_evaluator = fitness_evaluator
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.seen_matrices = set()  # Para evitar duplicados usando matrices
        self.population = []  # Lista de tuplas: (matriz_adyacencia, fitness)
        self.num_nodes = None  # Se establecerá en run()
    
    def _matrix_to_hashable(self, matrix):
        """
        Convierte una matriz de adyacencia a un formato hashable.
        """
        return tuple(map(tuple, matrix))
    
    def _matrix_to_graph(self, matrix):
        """
        Convierte una matriz de adyacencia a formato de grafo con lista de aristas.
        """
        nodes = list(range(self.num_nodes))
        edges = []
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if matrix[i, j] == 1:
                    edges.append((i, j, f'X{i}*X{j}'))  # término predeterminado
        
        return {'nodes': nodes, 'edges': edges}
    
    def _graph_to_matrix(self, graph):
        """
        Convierte un grafo a su representación de matriz de adyacencia.
        """
        num_nodes = len(graph['nodes'])
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        for i, j, _ in graph['edges']:
            matrix[i, j] = 1
        
        return matrix
    
    def _is_valid_matrix(self, matrix):
        """
        Verifica si una matriz representa un grafo válido:
        - diagonal cero
        - Conexo como grafo no dirigido
        """
        # Verificar diagonal (sin bucles)
        if np.any(np.diag(matrix) != 0):
            return False
        
        # Verificar conexidad como grafo no dirigido
        # Unir matriz con su transpuesta para obtener versión no dirigida
        undirected = np.logical_or(matrix, matrix.T).astype(int)
        
        G = nx.Graph()
        G.add_nodes_from(range(matrix.shape[0]))
        
        for i in range(matrix.shape[0]):
            for j in range(i+1, matrix.shape[0]):  # Solo mitad superior
                if undirected[i, j] == 1:
                    G.add_edge(i, j)
        
        return nx.is_connected(G)
    
    def _generate_random_matrix(self, num_nodes, edge_prob=0.5):
        """
        Genera una matriz de adyacencia aleatoria válida.
        """
        while True:
            # Crear grafo no dirigido aleatorio y conexo
            G_undirected = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)
            
            if nx.is_connected(G_undirected):
                # Inicializar matriz de adyacencia con ceros
                matrix = np.zeros((num_nodes, num_nodes), dtype=int)
                
                # Para cada arista no dirigida, asignar dirección(es) aleatoria(s)
                for u, v in G_undirected.edges():
                    if u == v:
                        continue  # Saltamos bucles
                    
                    # Decidir dirección(es)
                    add_u_to_v = random.random() < 0.5
                    add_v_to_u = random.random() < 0.5
                    
                    # Si ninguna fue seleccionada, elegir una por defecto
                    if not (add_u_to_v or add_v_to_u):
                        add_u_to_v = True
                    
                    if add_u_to_v:
                        matrix[u, v] = 1
                    if add_v_to_u:
                        matrix[v, u] = 1
                
                # Convertir a hashable para verificar si ya existe
                matrix_hash = self._matrix_to_hashable(matrix)
                
                if matrix_hash not in self.seen_matrices and self._is_valid_matrix(matrix):
                    self.seen_matrices.add(matrix_hash)
                    return matrix
    
    # def _crossover(self, matrix1, matrix2):
    #     """
    #     Realiza cruzamiento entre dos matrices de adyacencia.
    #     """
    #     # Punto de cruce aleatorio (por filas)
    #     crossover_point = random.randint(1, self.num_nodes - 1)
        
    #     # Crear hijos
    #     child = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
    #     # Primer hijo: primeras filas de matriz1, resto de matriz2
    #     child[:crossover_point, :] = matrix1[:crossover_point, :]
    #     child[crossover_point:, :] = matrix2[crossover_point:, :]
        
    #     # Asegurar que no hay bucles
    #     np.fill_diagonal(child, 0)
        
    #     # Verificar validez y unicidad
    #     if self._is_valid_matrix(child):
    #         child_hash = self._matrix_to_hashable(child)
    #         if child_hash not in self.seen_matrices:
    #             self.seen_matrices.add(child_hash)
    #             return child
        
    #     # Si el hijo no es válido o ya existe, intentar otro tipo de cruce
    #     # Intercambiar columnas en lugar de filas
    #     child = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
    #     child[:, :crossover_point] = matrix1[:, :crossover_point]
    #     child[:, crossover_point:] = matrix2[:, crossover_point:]
    #     np.fill_diagonal(child, 0)
        
    #     if self._is_valid_matrix(child):
    #         child_hash = self._matrix_to_hashable(child)
    #         if child_hash not in self.seen_matrices:
    #             self.seen_matrices.add(child_hash)
    #             return child
        
    #     # Si aún no es válido, intentar combinación aleatoria de aristas
    #     edges1 = set((i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if matrix1[i, j] == 1)
    #     edges2 = set((i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if matrix2[i, j] == 1)
        
    #     # Tomar mitad aleatoria de cada conjunto
    #     half1 = random.sample(list(edges1), len(edges1)//2)
    #     half2 = random.sample(list(edges2), len(edges2)//2)
        
    #     child = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
    #     for i, j in half1 + half2:
    #         child[i, j] = 1
    #     np.fill_diagonal(child, 0)
        
    #     # Asegurar conexidad
    #     if not self._is_valid_matrix(child):
    #         # Si no es conexo, añadir aristas hasta lograr conexidad
    #         child = self._ensure_connected(child)
        
    #     child_hash = self._matrix_to_hashable(child)
    #     if child_hash not in self.seen_matrices:
    #         self.seen_matrices.add(child_hash)
    #         return child
        
    #     print("no se pudo people")
    #     return None  # No se pudo generar un hijo válido
                
    def _crossover(self, matrix1, matrix2) -> tuple:
        """
        [VERSIÓN CON RETORNO DE TUPLA]
        Realiza un cruzamiento en dos etapas y devuelve siempre una tupla de dos elementos:
        (hijo1, hijo2), (hijo1, None) o (None, None).
        """
        
        # --- ETAPA 1: CRUCE POR FILAS ---
        stage1_children = []
        if self.num_nodes > 2:
            crossover_point = random.randint(1, self.num_nodes - 1)
        else:
            crossover_point = 1

        # Generar y validar los dos hijos posibles del cruce por filas
        # Hijo A: (P1-top, P2-bottom)
        child_a = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        child_a[:crossover_point, :] = matrix1[:crossover_point, :]
        child_a[crossover_point:, :] = matrix2[crossover_point:, :]
        np.fill_diagonal(child_a, 0)
        if self._is_valid_matrix(child_a):
            child_a_hash = self._matrix_to_hashable(child_a)
            if child_a_hash not in self.seen_matrices:
                self.seen_matrices.add(child_a_hash)
                stage1_children.append(child_a)

        # Hijo B: (P2-top, P1-bottom)
        child_b = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        child_b[:crossover_point, :] = matrix2[:crossover_point, :]
        child_b[crossover_point:, :] = matrix1[crossover_point:, :]
        np.fill_diagonal(child_b, 0)
        if self._is_valid_matrix(child_b):
            child_b_hash = self._matrix_to_hashable(child_b)
            if child_b_hash not in self.seen_matrices:
                self.seen_matrices.add(child_b_hash)
                stage1_children.append(child_b)
        
        # Si la Etapa 1 tuvo éxito, devolver el resultado según el número de hijos
        if len(stage1_children) == 2:
            return stage1_children[0], stage1_children[1]
        if len(stage1_children) == 1:
            return stage1_children[0], None
        
        # --- ETAPA 2: CRUCE POR COLUMNAS (SI LA ETAPA 1 FALLÓ) ---
        stage2_children = []

        # Hijo C: (P1-left, P2-right)
        child_c = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        child_c[:, :crossover_point] = matrix1[:, :crossover_point]
        child_c[:, crossover_point:] = matrix2[:, crossover_point:]
        np.fill_diagonal(child_c, 0)
        if self._is_valid_matrix(child_c):
            child_c_hash = self._matrix_to_hashable(child_c)
            if child_c_hash not in self.seen_matrices:
                self.seen_matrices.add(child_c_hash)
                stage2_children.append(child_c)
        
        # Hijo D: (P2-left, P1-right)
        child_d = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        child_d[:, :crossover_point] = matrix2[:, :crossover_point]
        child_d[:, crossover_point:] = matrix1[:, crossover_point:]
        np.fill_diagonal(child_d, 0)
        if self._is_valid_matrix(child_d):
            child_d_hash = self._matrix_to_hashable(child_d)
            if child_d_hash not in self.seen_matrices:
                self.seen_matrices.add(child_d_hash)
                stage2_children.append(child_d)
                
        # Devolver el resultado de la Etapa 2
        if len(stage2_children) == 2:
            return stage2_children[0], stage2_children[1]
        if len(stage2_children) == 1:
            return stage2_children[0], None

        # Si ninguna etapa tuvo éxito, devolver None, None
        return None, None
    
    def _ensure_connected(self, matrix):
        """
        Asegura que la matriz representa un grafo conexo.
        """
        # Versión no dirigida
        undirected = np.logical_or(matrix, matrix.T).astype(int)
        
        # Crear grafo no dirigido
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if undirected[i, j] == 1:
                    G.add_edge(i, j)
        
        # Si ya es conexo, no hacer nada
        if nx.is_connected(G):
            return matrix
        
        # Obtener componentes conectados
        components = list(nx.connected_components(G))
        
        # Crear una copia para modificar
        result = matrix.copy()
        
        # Mientras haya más de un componente, conectarlos
        while len(components) > 1:
            # Tomar dos componentes
            comp1 = list(components[0])
            comp2 = list(components[1])
            
            # Seleccionar nodos aleatorios de cada componente
            node1 = random.choice(comp1)
            node2 = random.choice(comp2)
            
            # Añadir arista dirigida (dirección aleatoria)
            if random.random() < 0.5:
                result[node1, node2] = 1
            else:
                result[node2, node1] = 1
            
            # Actualizar grafo y recalcular componentes
            G.add_edge(node1, node2)
            components = list(nx.connected_components(G))
        
        return result
    
    def _mutate(self, matrix):
        """
        [VERSIÓN CON LÓGICA ESPECÍFICA DEL USUARIO]
        Aplica una mutación siguiendo reglas precisas para añadir, quitar o voltear
        una arista en una posición (i,j) elegida al azar.
        """
        # Se intentará encontrar una mutación válida un máximo de 3 veces.
        for _ in range(5):
            
            # 1. Elegir una posición aleatoria (i, j) donde i != j
            i = random.randrange(self.num_nodes)
            j = random.randrange(self.num_nodes)
            while i == j:
                j = random.randrange(self.num_nodes)

            temp = matrix.copy()
            
            # 2. Comprobar si ya existe una arista en la posición (i, j)
            if matrix[i, j] == 1:
                # --- CASO: LA ARISTA (i, j) EXISTE ---
                
                # Definir las acciones posibles
                possible_actions = ['remove']
                # La acción 'flip' solo es posible si la arista inversa no existe
                if matrix[j, i] == 0:
                    possible_actions.append('flip')
                
                action = random.choice(possible_actions)
                
                if action == 'remove':
                    # Eliminar la arista
                    temp[i, j] = 0
                    # Comprobar si el grafo resultante es válido (conexo), como se pidió
                    if not self._is_valid_matrix(temp):
                        continue # El intento falla, pasa a la siguiente iteración del bucle de 3 intentos
                
                elif action == 'flip':
                    # Voltear la arista
                    temp[i, j] = 0
                    temp[j, i] = 1
                    # No se comprueba la validez, como se pidió
            
            else: # matrix[i, j] == 0
                # --- CASO: LA ARISTA (i, j) NO EXISTE ---

                # Definir las acciones posibles
                possible_actions = ['add']
                # La acción 'flip_reverse' solo es posible si la arista inversa SÍ existe
                if matrix[j, i] == 1:
                    possible_actions.append('flip_reverse')
                
                action = random.choice(possible_actions)

                if action == 'add':
                    # Añadir la arista
                    temp[i, j] = 1

                elif action == 'flip_reverse':
                    # Voltear la arista inversa
                    temp[j, i] = 0
                    temp[i, j] = 1
                
                # No se comprueba la validez para ninguna de estas acciones, como se pidió

            # 4. Comprobar si el grafo resultante ya ha sido analizado
            mutated_hash = self._matrix_to_hashable(temp)
            if mutated_hash not in self.seen_matrices:
                # Si es nuevo, la mutación es un éxito
                self.seen_matrices.add(mutated_hash)
                return temp # Devolver el nuevo grafo y salir de la función

            # Si el grafo ya existía, el intento falla y el bucle for continúa.

        # 5. Si se agotan los 3 intentos sin éxito, no se devuelve nada.
        return None
    
    def _evaluate_population(self):
        for i in range(len(self.population)):
            matrix, fit = self.population[i]
            if fit is None:
                # Convertir matriz a formato de grafo para evaluación
                graph = self._matrix_to_graph(matrix)
                details = self.fitness_evaluator.evaluate_with_details(graph)
                fitness = details["fitness"]
                self.population[i] = (matrix, fitness)
    
    
    
    def run(self, num_nodes):
        self.num_nodes = num_nodes
        
        print(f"🧬 Iniciando búsqueda genética con {self.population_size} individuos y {self.num_generations} generaciones")
        
        # Generación inicial (sin cambios)
        while len(self.population) < self.population_size:
            matrix = self._generate_random_matrix(num_nodes)
            self.population.append((matrix, None))
        
        self._evaluate_population()
        self.population = sorted(self.population, key=lambda x: x[1])
        
        best_matrix, best_fitness = self.population[0]
        print(f"🏆 Población inicial - Mejor fitness: {best_fitness:.6f}")
        
        # --- INICIO DEL BUCLE EVOLUTIVO MODIFICADO ---
        for gen in range(self.num_generations):
            print(f"\n🧬 Generación {gen + 1}")
            
            # --- 1. CRUZAMIENTO POR EMPAREJAMIENTO ALEATORIO ---
            offspring = []
            parent_indices = list(range(len(self.population)))
            random.shuffle(parent_indices)
            
            for i in range(0, len(parent_indices), 2):
                if i + 1 < len(parent_indices):
                    parent1, _ = self.population[parent_indices[i]]
                    parent2, _ = self.population[parent_indices[i+1]]
                    
                    child1, child2 = self._crossover(parent1.copy(), parent2.copy())
                    
                    if child1 is not None:
                        offspring.append((child1, None))
                    if child2 is not None:
                        offspring.append((child2, None))

            # --- 2. MUTACIÓN SOBRE LA POBLACIÓN ORIGINAL ---
            mutants = []
            # Calcular cuántos individuos mutar basado en el mutation_rate
            num_to_mutate = int(len(self.population) * self.mutation_rate)
            
            # Seleccionar individuos al azar de la población original para mutar
            # Usamos random.sample para evitar elegir al mismo individuo varias veces
            indices_to_mutate = random.sample(range(len(self.population)), num_to_mutate)

            for index in indices_to_mutate:
                parent_matrix, _ = self.population[index]
                mutated = self._mutate(parent_matrix)
                if mutated is not None:
                    mutants.append((mutated, None))
            
            # --- 3. SELECCIÓN Y ELITISMO (SUPERVIVENCIA) ---
            
            # Nueva población combinada: padres + hijos + mutantes
            # Es importante notar que self.population aquí contiene a los "padres"
            self.population.extend(offspring)
            self.population.extend(mutants)
            
            # Evaluar a los nuevos individuos
            self._evaluate_population()
            
            # Seleccionar los mejores para la siguiente generación
            self.population = sorted(self.population, key=lambda x: x[1])[:self.population_size]
            
            # Mostrar el mejor de la generación
            best_matrix, best_fitness = self.population[0]
            best_graph = self._matrix_to_graph(best_matrix)
            print(f"🏆 Mejor fitness: {best_fitness:.6f}, Aristas: {len(best_graph['edges'])}")
        
        # --- FIN DEL BUCLE EVOLUTIVO MODIFICADO ---

        # Convertir los mejores a formato de grafo para retornar (sin cambios)
        best_results = []
        for matrix, fitness in self.population[:5]:
            graph = self._matrix_to_graph(matrix)
            best_results.append((graph, fitness))
        
        return best_results
    
    
    # def run(self, num_nodes):
    #     self.num_nodes = num_nodes
        
    #     print(f"🧬 Iniciando búsqueda genética con {self.population_size} individuos y {self.num_generations} generaciones")
        
    #     # Generación inicial
    #     while len(self.population) < self.population_size:
    #         matrix = self._generate_random_matrix(num_nodes)
    #         self.population.append((matrix, None))
        
    #     # Evaluar población inicial
    #     self._evaluate_population()
        
    #     # Ordenar por fitness
    #     self.population = sorted(self.population, key=lambda x: x[1])
        
    #     best_matrix, best_fitness = self.population[0]
    #     print(f"🏆 Población inicial - Mejor fitness: {best_fitness:.6f}")
        
    #     # Evolucionar por generaciones
    #     for gen in range(self.num_generations):
    #         print(f"\n🧬 Generación {gen + 1}")
            
    #         # Cruzamiento
    #         offspring = []
    #         crossover_attempts = 0
    #         max_attempts = self.population_size * 3  # Límite para evitar bucles infinitos
            
    #         while len(offspring) < int(self.crossover_rate * self.population_size) and crossover_attempts < max_attempts:
    #             # Seleccionar padres con sesgo hacia los mejores
    #             parent_indices = list(range(len(self.population)))
    #             weights = [1.0/(i+1) for i in range(len(self.population))]  # Favorecer individuos con mejor fitness
                
    #             idx1, idx2 = random.choices(parent_indices, weights=weights, k=2)
    #             parent1, _ = self.population[idx1]
    #             parent2, _ = self.population[idx2]
                
    #             # child = self._crossover(parent1, parent2)
    #             # if child is not None:
    #             #     offspring.append((child, None))
                
    #             # crossover_attempts += 1
    #             # _crossover ahora devuelve una tupla (hijo1, hijo2)
    #             child1, child2 = self._crossover(parent1, parent2)
                
    #             # Añadir los hijos que no sean None a la descendencia
    #             if child1 is not None:
    #                 offspring.append((child1, None))
    #             if child2 is not None:
    #                 offspring.append((child2, None))
                
    #             crossover_attempts += 1
    #         # Mutación
    #         mutants = []
    #         mutation_attempts = 0
    #         max_attempts = self.population_size * 3
            
    #         while len(mutants) < int(self.mutation_rate * self.population_size) and mutation_attempts < max_attempts:
    #             parent, _ = random.choice(self.population)
    #             mutated = self._mutate(parent)
    #             if mutated is not None:
    #                 mutants.append((mutated, None))
                
    #             mutation_attempts += 1
            
    #         # Nueva población combinada
    #         self.population.extend(offspring)
    #         self.population.extend(mutants)
            
    #         # Evaluar
    #         self._evaluate_population()
            
    #         # Seleccionar los mejores
    #         self.population = sorted(self.population, key=lambda x: x[1])[:self.population_size]
            
    #         # Mostrar el mejor de la generación
    #         best_matrix, best_fitness = self.population[0]
    #         best_graph = self._matrix_to_graph(best_matrix)
    #         print(f"🏆 Mejor fitness: {best_fitness:.6f}, Aristas: {len(best_graph['edges'])}")
        
    #     # Convertir los mejores a formato de grafo para retornar
    #     best_results = []
    #     for matrix, fitness in self.population[:5]:
    #         graph = self._matrix_to_graph(matrix)
    #         best_results.append((graph, fitness))
        
    #     return best_results

if __name__ == "__main__":

    # --- Modelo SIR sintético ---
    def sir_model(t, y, beta=0.3, gamma=0.1):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Condiciones iniciales y tiempo
    y0 = [0.99, 0.01, 0.0]
    t = np.linspace(0, 100, 50)

    # Simulación + ruido
    solution = solve_ivp(sir_model, [0, 100], y0, t_eval=t)
    data = solution.y.T
    np.random.seed(42)
    noisy_data = data + np.random.normal(0, 0.005, data.shape)

    # --- Evaluador de fitness ---
    evaluator = CompartmentalGraphFitness(
        data=noisy_data,
        time_points=t,
        threshold=1.2,
        conserve_population=True,
        enforce_non_negative=True,
        debug=False  # Opcional: cambiar a False para ejecución silenciosa
    )

    # --- Algoritmo genético ---
    search = GeneticGraphSearch(
        fitness_evaluator=evaluator,
        population_size=5,
        num_generations=8,
        mutation_rate=0.3,
        crossover_rate=0.9
    )

    # Correr el algoritmo genético para 3 compartimentos: S, I, R
    # Usa esto:
    best_results = search.run(num_nodes=3)
    best_graph1, best_fitness1 = best_results[0]  # Tomar solo el mejor resultado
    best_graph2, best_fitness2 = best_results[1]
    best_graph3, best_fitness3 = best_results[2]
    best_graph4, best_fitness4 = best_results[3]
    best_graph5, best_fitness5 = best_results[4]

    print("\n===== MEJOR GRAFO ENCONTRADO =====")
    print(f"Fitness: {best_fitness1:.6f}")
    print("Nodos:", best_graph1['nodes'])
    print("Aristas:")
    for edge in best_graph1['edges']:
        print(f"  {edge[0]} -> {edge[1]} con término: {edge[2]}")

    print("\n===== MEJOR GRAFO ENCONTRADO =====")
    print(f"Fitness: {best_fitness2:.6f}")
    print("Nodos:", best_graph2['nodes'])
    print("Aristas:")
    for edge in best_graph2['edges']:
        print(f"  {edge[0]} -> {edge[1]} con término: {edge[2]}")

    print("\n===== MEJOR GRAFO ENCONTRADO =====")
    print(f"Fitness: {best_fitness3:.6f}")
    print("Nodos:", best_graph3['nodes'])
    print("Aristas:")
    for edge in best_graph3['edges']:
        print(f"  {edge[0]} -> {edge[1]} con término: {edge[2]}")

    print("\n===== MEJOR GRAFO ENCONTRADO =====")
    print(f"Fitness: {best_fitness4:.6f}")
    print("Nodos:", best_graph4['nodes'])
    print("Aristas:")
    for edge in best_graph4['edges']:
        print(f"  {edge[0]} -> {edge[1]} con término: {edge[2]}")

    print("\n===== MEJOR GRAFO ENCONTRADO =====")
    print(f"Fitness: {best_fitness5:.6f}")
    print("Nodos:", best_graph5['nodes'])
    print("Aristas:")
    for edge in best_graph5['edges']:
        print(f"  {edge[0]} -> {edge[1]} con término: {edge[2]}")
