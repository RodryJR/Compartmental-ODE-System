import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx
import random
import time
from src.fitness import CompartmentalGraphFitness
from src.ga import GeneticGraphSearch

def generate_random_compartmental_model(num_compartments, edge_probability=0.3, min_edges=None):
    """
    Genera un modelo compartimental aleatorio válido.
    
    Args:
        num_compartments: Número de compartimentos
        edge_probability: Probabilidad de arista entre dos nodos
        min_edges: Mínimo número de aristas (por defecto: num_compartments-1)
        
    Returns:
        adjacency_matrix: Matriz de adyacencia del grafo
        rate_constants: Constantes de velocidad para cada arista
    """
    if min_edges is None:
        min_edges = num_compartments - 1
    
    # Intentar generar un grafo válido
    max_attempts = 100
    for _ in range(max_attempts):
        # Generar grafo aleatorio
        adjacency = np.zeros((num_compartments, num_compartments), dtype=int)
        
        # Para cada par de nodos, decidir si hay arista
        for i in range(num_compartments):
            for j in range(num_compartments):
                if i != j and random.random() < edge_probability:
                    adjacency[i, j] = 1
        
        # Verificar conectividad en versión no dirigida
        undirected = np.logical_or(adjacency, adjacency.T).astype(int)
        G = nx.Graph()
        for i in range(num_compartments):
            G.add_node(i)
        
        for i in range(num_compartments):
            for j in range(i+1, num_compartments):
                if undirected[i, j] == 1:
                    G.add_edge(i, j)
        
        # Si es conexo y tiene suficientes aristas, es válido
        if nx.is_connected(G) and np.sum(adjacency) >= min_edges:
            # Generar constantes de velocidad para cada arista
            rate_constants = {}
            for i in range(num_compartments):
                for j in range(num_compartments):
                    if adjacency[i, j] == 1:
                        # Asignar constante entre 0.1 y 0.5
                        rate_constants[(i, j)] = random.uniform(0.1, 0.5)
            
            return adjacency, rate_constants
    
    # Si llegamos aquí, no pudimos generar grafo válido
    raise ValueError("No se pudo generar modelo compartimental válido")

def build_ode_function(adjacency_matrix, rate_constants):
    """
    Construye la función ODE basada en la matriz de adyacencia y constantes.
    
    Args:
        adjacency_matrix: Matriz de adyacencia
        rate_constants: Diccionario con constantes de velocidad
        
    Returns:
        function: Función ODE para resolver
    """
    num_compartments = adjacency_matrix.shape[0]
    
    def ode_system(t, y):
        dydt = np.zeros(num_compartments)
        
        # Calcular población total
        N = sum(y)
        
        for i in range(num_compartments):
            # Calcular flujos de salida desde compartimento i
            for j in range(num_compartments):
                if adjacency_matrix[i, j] == 1:
                    # Transferencia de población de i a j
                    # Usamos interacción masiva: velocidad proporcional a la población en i
                    flow = rate_constants[(i, j)] * y[i]
                    
                    # Flujo negativo para i (salida)
                    dydt[i] -= flow
                    # Flujo positivo para j (entrada)
                    dydt[j] += flow
        
        return dydt
    
    return ode_system

def model_to_graph_dict(adjacency_matrix, terms=None):
    """
    Convierte matriz de adyacencia a formato de grafo para el evaluador.
    
    Args:
        adjacency_matrix: Matriz de adyacencia
        terms: Diccionario de términos para aristas (opcional)
        
    Returns:
        dict: Diccionario con nodos y aristas
    """
    num_nodes = adjacency_matrix.shape[0]
    
    nodes = list(range(num_nodes))
    edges = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                # Término predeterminado si no se especifica
                term = f"linear" if terms is None else terms.get((i, j), "linear")
                edges.append((i, j, term))
    
    return {'nodes': nodes, 'edges': edges}

def compare_adjacency_matrices(original, recovered):
    """
    Compara matriz original con la recuperada.
    
    Args:
        original: Matriz de adyacencia original
        recovered: Matriz de adyacencia recuperada
        
    Returns:
        float: Similitud entre 0 y 1
    """
    # Convertir a conjuntos de aristas
    original_edges = set()
    recovered_edges = set()
    
    num_nodes = original.shape[0]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if original[i, j] == 1:
                original_edges.add((i, j))
            if recovered[i, j] == 1:
                recovered_edges.add((i, j))
    
    # Calcular similitud
    if not original_edges and not recovered_edges:
        return 1.0  # Ambos vacíos = coincidencia perfecta
    
    intersection = len(original_edges.intersection(recovered_edges))
    union = len(original_edges.union(recovered_edges))
    
    return intersection / union

def graph_dict_to_adjacency(graph_dict, num_nodes):
    """
    Convierte grafo en formato diccionario a matriz de adyacencia.
    
    Args:
        graph_dict: Grafo en formato diccionario
        num_nodes: Número de nodos
        
    Returns:
        numpy.ndarray: Matriz de adyacencia
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for edge in graph_dict['edges']:
        i, j, _ = edge
        adjacency[i, j] = 1
    
    return adjacency

def test_genetic_algorithm_with_random_model(num_compartments=4, population_size=10, generations=10):
    """
    Prueba el algoritmo genético con un modelo compartimental aleatorio.
    
    Args:
        num_compartments: Número de compartimentos
        population_size: Tamaño de población para el algoritmo genético
        generations: Número de generaciones
    """
    print(f"🧪 Generando modelo aleatorio con {num_compartments} compartimentos...")
    
    # 1. Generar modelo aleatorio
    adjacency, rate_constants = generate_random_compartmental_model(
        num_compartments, 
        edge_probability=0.3, 
        min_edges=num_compartments
    )
    
    # Visualizar modelo original
    G_original = nx.DiGraph()
    for i in range(num_compartments):
        G_original.add_node(i)
    
    for i in range(num_compartments):
        for j in range(num_compartments):
            if adjacency[i, j] == 1:
                G_original.add_edge(i, j, weight=rate_constants[(i, j)])
    
    plt.figure(figsize=(8, 8))
    pos = nx.circular_layout(G_original)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G_original, pos, node_color='lightblue', node_size=800)
    
    # Dibujar aristas con grosores basados en rates
    edges = G_original.edges(data=True)
    weights = [d['weight']*5 for _, _, d in edges]  # Multiplicar por 5 para mejor visualización
    
    nx.draw_networkx_edges(G_original, pos, width=weights, 
                         edge_color='gray', arrowsize=20, arrowstyle='->')
    
    # Etiquetas de nodos
    labels = {i: f'X{i}' for i in G_original.nodes()}
    nx.draw_networkx_labels(G_original, pos, labels=labels, font_weight='bold', font_size=12)
    
    # Etiquetas de aristas con rates
    edge_labels = {(i, j): f"{rate_constants[(i, j)]:.2f}" for i, j in G_original.edges()}
    nx.draw_networkx_edge_labels(G_original, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Modelo Original Aleatorio")
    plt.axis('off')
    plt.savefig("random_model_original.png")
    plt.close()
    
    print(f"✓ Modelo generado con {np.sum(adjacency)} aristas")
    
    # 2. Construir función ODE y simular datos
    ode_system = build_ode_function(adjacency, rate_constants)
    
    # Condiciones iniciales (distribución aleatoria pero que sume 1.0)
    initial_values = np.random.random(num_compartments)
    initial_values = initial_values / np.sum(initial_values)
    
    total_population = 1000  # Población total
    y0 = initial_values * total_population
    
    # Tiempo de simulación
    t_span = [0, 100]
    t = np.linspace(0, 100, 50)  # 50 puntos de datos
    
    # Resolver ODEs
    solution = solve_ivp(ode_system, t_span, y0, t_eval=t, method='RK45')
    data = solution.y.T
    
    # Añadir ruido
    noise_level = 0.02 * total_population  # 2% de ruido
    np.random.seed(42)  # Para reproducibilidad
    noisy_data = data + np.random.normal(0, noise_level, data.shape)
    noisy_data = np.maximum(noisy_data, 0)  # No valores negativos
    
    # Visualizar datos
    plt.figure(figsize=(10, 6))
    for i in range(num_compartments):
        plt.plot(t, data[:, i], '-', label=f'X{i} (real)')
        plt.scatter(t, noisy_data[:, i], alpha=0.3, s=20, label=f'X{i} (ruido)')
    
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Datos del Modelo Aleatorio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("random_model_data.png")
    plt.close()
    
    print(f"✓ Datos generados con {len(t)} puntos temporales")
    
    # 3. Configurar y ejecutar algoritmo genético
    evaluator = CompartmentalGraphFitness(
        data=noisy_data,
        time_points=t,
        threshold=0.001,
        conserve_population=True,
        enforce_non_negative=True,
        debug=False
    )
    
    search = GeneticGraphSearch(
        fitness_evaluator=evaluator,
        population_size=population_size,
        num_generations=generations,
        mutation_rate=1.0,
        crossover_rate=1.0
    )
    
    print(f"\n🧬 Iniciando búsqueda genética con población {population_size}, generaciones {generations}...")
    start_time = time.time()
    best_results = search.run(num_nodes=num_compartments)
    elapsed_time = time.time() - start_time
    print(f"✓ Búsqueda completada en {elapsed_time:.2f} segundos")
    
    # 4. Analizar resultados
    print("\n🔍 Analizando los 5 mejores grafos encontrados...")
    
    # Convertir resultados a matrices de adyacencia
    recovered_matrices = []
    for graph, fitness in best_results:
        adj = graph_dict_to_adjacency(graph, num_compartments)
        recovered_matrices.append((adj, fitness))
    
    # Comparar con original
    found_match = False
    best_similarity = 0
    best_index = -1
    
    for i, (matrix, fitness) in enumerate(recovered_matrices):
        similarity = compare_adjacency_matrices(adjacency, matrix)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i
        
        if similarity == 1.0:
            found_match = True
        
        print(f"\n===== GRAFO #{i+1} (Fitness: {fitness:.6f}) =====")
        print(f"Similitud con estructura original: {similarity:.2%}")
        
        # Mostrar aristas
        edges = []
        for i_node in range(num_compartments):
            for j_node in range(num_compartments):
                if matrix[i_node, j_node] == 1:
                    edges.append((i_node, j_node))
        
        print(f"Aristas ({len(edges)}):")
        for edge in edges:
            print(f"  X{edge[0]} → X{edge[1]}")
    
    # Resultado global
    print("\n📊 RESULTADOS FINALES 📊")
    if found_match:
        print("✅ El algoritmo encontró EXACTAMENTE la estructura original del modelo")
    else:
        print(f"⚠️ Mejor similitud: {best_similarity:.2%} en grafo #{best_index+1}")
        if best_similarity >= 0.8:
            print("✅ Muy buena aproximación a la estructura original")
        elif best_similarity >= 0.6:
            print("👍 Buena aproximación a la estructura original")
        else:
            print("❌ Estructura significativamente diferente del original")
    
    # 5. Visualizar mejor modelo encontrado
    best_graph_dict = best_results[best_index][0]
    best_fitness = best_results[best_index][1]
    best_matrix = recovered_matrices[best_index][0]
    
    # Crear grafo para visualización
    G_best = nx.DiGraph()
    for i in range(num_compartments):
        G_best.add_node(i)
    
    for edge in best_graph_dict['edges']:
        G_best.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(8, 8))
    pos = nx.circular_layout(G_best)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G_best, pos, node_color='lightgreen', node_size=800)
    
    # Dibujar aristas
    nx.draw_networkx_edges(G_best, pos, 
                         edge_color='blue', arrowsize=20, arrowstyle='->')
    
    # Etiquetas de nodos
    labels = {i: f'X{i}' for i in G_best.nodes()}
    nx.draw_networkx_labels(G_best, pos, labels=labels, font_weight='bold', font_size=12)
    
    # Etiquetas de aristas con términos
    edge_labels = {}
    for edge in best_graph_dict['edges']:
        edge_labels[(edge[0], edge[1])] = edge[2]
    
    nx.draw_networkx_edge_labels(G_best, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(f"Mejor Modelo Encontrado (Similitud: {best_similarity:.2%})")
    plt.axis('off')
    plt.savefig("random_model_best.png")
    plt.close()
    
    print("\n📊 Visualizaciones guardadas:")
    print("- 'random_model_original.png': Estructura original")
    print("- 'random_model_data.png': Datos generados")
    print("- 'random_model_best.png': Mejor estructura encontrada")
    
    print("\n🧪 Prueba finalizada 🧪")
    
    return {
        'original_adjacency': adjacency,
        'best_adjacency': best_matrix,
        'similarity': best_similarity,
        'found_exact_match': found_match,
        'best_fitness': best_fitness
    }

# Ejecutar múltiples pruebas con diferentes configuraciones
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    # Prueba con modelo pequeño (3 compartimentos)
    print("\n" + "="*80)
    print("PRUEBA 1: Modelo pequeño (3 compartimentos)")
    print("="*80)
    results1 = test_genetic_algorithm_with_random_model(
        num_compartments=3,
        population_size=30,
        generations=12
    )
    
    # Prueba con modelo mediano (4 compartimentos)
    print("\n" + "="*80)
    print("PRUEBA 2: Modelo mediano (4 compartimentos)")
    print("="*80)
    results2 = test_genetic_algorithm_with_random_model(
        num_compartments=4,
        population_size=30,
        generations=20
    )
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"Prueba 1 (3 compartimentos): {results1['similarity']:.2%} similitud")
    print(f"Prueba 2 (4 compartimentos): {results2['similarity']:.2%} similitud")
    
    print("\n¡Pruebas completadas!")