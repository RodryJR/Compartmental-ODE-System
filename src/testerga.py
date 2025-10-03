import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from fitness import*
from ga import*

# Asume que tus clases CompartmentalGraphFitness y GeneticGraphSearch
# están en el mismo archivo o importadas.
# from your_file import CompartmentalGraphFitness, GeneticGraphSearch


def rebuild_and_get_model(evaluator: 'CompartmentalGraphFitness', graph: dict):
    """
    Función de ayuda para reconstruir y obtener el sistema de EDOs de un grafo.
    
    NOTA: La lógica de esta función debería ser un método dentro de tu clase
    CompartmentalGraphFitness para evitar duplicar código.
    Ej: `evaluator.get_model_from_graph(graph)`

    Esta función simula los pasos internos de `evaluate` para obtener el modelo final.
    """
    # Pasos internos de la clase 'CompartmentalGraphFitness' para reconstruir un modelo
    expanded_equations = evaluator._generate_expanded_equations()
    coefficients = evaluator._fit_coefficients(expanded_equations)
    pruned_equations = evaluator._prune_insignificant_terms(expanded_equations, coefficients)
    shared_terms = evaluator._identify_relevant_shared_terms(pruned_equations, graph)
    reconstructed_equations = evaluator._reconstruct_equations_correctly(shared_terms, graph)
    adjusted_equations = evaluator._refit_coefficients(reconstructed_equations)
    
    # Este es el paso clave: obtener el sistema de EDOs y las ecuaciones simbólicas
    ode_system, symbolic_eqs = evaluator._build_ode_system(adjusted_equations)
    
    return ode_system, symbolic_eqs


if __name__ == "__main__":
    # --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---

    # Modelo SIR sintético (la "verdad fundamental")
    def sir_model(t, y, beta=0.3, gamma=0.1):
        S, I, R = y
        # Aseguramos que no haya valores negativos
        S, I, R = max(S, 0), max(I, 0), max(R, 0)
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Condiciones iniciales y tiempo
    y0 = [0.90, 0.20, 0.0]
    t = np.linspace(0, 100, 100) # Más puntos para una mejor visualización

    # Generar datos limpios (para validación) y con ruido (para entrenamiento)
    solution = solve_ivp(sir_model, [0, 100], y0, t_eval=t)
    clean_data = solution.y.T
    np.random.seed(42)
    noisy_data = clean_data + np.random.normal(0, 0.005, clean_data.shape)

    # --- 2. EJECUCIÓN DEL ALGORITMO GENÉTICO ---

    # Instanciar el evaluador de fitness
    evaluator = CompartmentalGraphFitness(
        data=noisy_data,
        time_points=t,
        threshold=0.01,
        conserve_population=True,
        enforce_non_negative=True,
        debug=False # Silencioso para no saturar la salida
    )

    # Instanciar el algoritmo genético
    # NOTA: Aumentar population_size y num_generations para mejores resultados
    search = GeneticGraphSearch(
        fitness_evaluator=evaluator,
        population_size=30,
        num_generations=20,
        mutation_rate=0.8,
        crossover_rate=1.0
    )

    print("Iniciando búsqueda con Algoritmo Genético... (puede tardar un momento)")
    # Ejecutar para un sistema de 3 nodos (S, I, R)
    top_solutions = search.run(num_nodes=3)
    print("Búsqueda finalizada. Analizando las 5 mejores soluciones...")

    # --- 3. ANÁLISIS Y VALIDACIÓN DE LAS MEJORES SOLUCIONES ---

    for i, (graph, fitness) in enumerate(top_solutions[:5]):
        print("\n" + "="*80)
        print(f"ANALIZANDO LA SOLUCIÓN #{i+1}")
        print("="*80)
        
        # Imprimir información básica del grafo encontrado
        print(f"Fitness del Algoritmo Genético: {fitness:.6f}")
        print(f"Estructura del Grafo: {graph['edges']}")

        # 1. Reconstruir el modelo y obtener sus ecuaciones
        print("\n[+] Reconstruyendo modelo y extrayendo ecuaciones...")
        try:
            final_ode_system, symbolic_eqs = rebuild_and_get_model(evaluator, graph)
            print("Ecuaciones descubiertas:")
            for j, eq in enumerate(symbolic_eqs):
                # Usamos sp.N para simplificar la expresión a formato decimal
                print(f"    dX{j}/dt = {sp.N(eq, 4)}") 
        except Exception as e:
            print(f"No se pudo reconstruir el modelo para esta solución. Error: {e}")
            continue

        # 2. Generar puntos simulando el modelo descubierto
        print("[+] Simulando trayectoria con el modelo descubierto...")
        discovered_solution = solve_ivp(
            final_ode_system,
            [t[0], t[-1]],
            y0,
            t_eval=t,
            method='BDF' # Usar un solver robusto
        )
        discovered_data = discovered_solution.y.T

        # 3. Calcular el MSE de validación contra los datos LIMPIOS originales
        validation_mse = np.mean((discovered_data - clean_data)**2)
        print(f"\n[!] MSE de Validación (vs datos originales limpios): {validation_mse:.8f}")

        # 4. Generar gráfica de comparación
        print("[+] Generando gráfica de comparación...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Graficar datos originales (con ruido)
        for j in range(clean_data.shape[1]):
            ax.plot(t, noisy_data[:, j], 'o', markersize=5, alpha=0.5, label=f'Datos X{j} (con ruido)')
        
        # Graficar la simulación del modelo descubierto
        # Cambiar el color para la línea del modelo
        colors = ['blue', 'red', 'green']
        for j in range(discovered_data.shape[1]):
             ax.plot(t, discovered_data[:, j], '-', linewidth=2.5, color=colors[j], label=f'Modelo Descubierto X{j}')
        
        ax.set_title(f'Validación de la Solución #{i+1}\nFitness: {fitness:.4f} | MSE (vs real): {validation_mse:.6f}', fontsize=16)
        ax.set_xlabel("Tiempo", fontsize=12)
        ax.set_ylabel("Población", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.show()