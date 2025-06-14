import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.fitness import CompartmentalGraphFitness
from src.ga import GeneticGraphSearch
import os

def run_genetic_experiment(
    input_filename: str,
    analysis_filename: str,
    ga_params: dict,
    fitness_params: dict
):
    """
    Ejecuta un experimento completo del algoritmo genético desde un archivo CSV.

    Esta función carga los datos, configura el algoritmo genético, lo ejecuta,
    y finalmente analiza, guarda e imprime los resultados de las mejores soluciones.

    Args:
        input_filename (str): Ruta al archivo CSV con los datos de entrada.
        analysis_filename (str): Ruta al archivo de texto donde se guardará el análisis.
        ga_params (dict): Diccionario con los parámetros para el algoritmo genético.
                          Ej: {'population_size': 30, 'num_generations': 20, 'mutation_rate': 1.0}
        fitness_params (dict): Diccionario con los parámetros para la función de fitness.
                               Ej: {'threshold': 1.0, 'conserve_population': True}
    """
    # --- 1. CARGA Y PREPARACIÓN DE DATOS ---
    print(f"[*] Cargando datos desde '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{input_filename}'.")
        return

    time_points = df['time'].values
    data = df.drop(columns='time').values
    num_nodes = data.shape[1]
    print(f"[*] Datos cargados: {data.shape[0]} puntos de tiempo para {num_nodes} compartimentos.")

    # --- 2. CONFIGURACIÓN Y EJECUCIÓN DEL ALGORITMO GENÉTICO ---
    evaluator = CompartmentalGraphFitness(
        data=data,
        time_points=time_points,
        threshold=fitness_params.get('threshold', 1.0),
        conserve_population=fitness_params.get('conserve_population', True),
        enforce_non_negative=True,
        debug=False
    )

    search = GeneticGraphSearch(
        fitness_evaluator=evaluator,
        population_size=ga_params.get('population_size', 30),
        num_generations=ga_params.get('num_generations', 20),
        mutation_rate=ga_params.get('mutation_rate', 1.0),
        crossover_rate=ga_params.get('crossover_rate', 1.0)
    )

    print("\n[*] Iniciando búsqueda con Algoritmo Genético...")
    top_solutions = search.run(num_nodes=num_nodes)
    print("[*] Búsqueda finalizada. Analizando las mejores soluciones...")

    # <-- 2. Crear una carpeta para guardar las gráficas ---
    plot_dir = "analysis_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    with open(f'{analysis_filename}.txt', "w") as f:
        f.write(f"ANÁLISIS DE LAS MEJORES SOLUCIONES\nArchivo de datos: {input_filename}\n")
        f.write("="*60 + "\n")

    for i, (graph, fitness) in enumerate(top_solutions[:5]):
        print("\n" + "="*80)
        print(f"ANALIZANDO LA SOLUCIÓN #{i+1}")
        # ... (La lógica de análisis y guardado de texto sigue igual)
        print("="*80)
        details = evaluator.evaluate_with_details(graph)
        texto_resultado = evaluator.format_result_as_string(details)
        print(texto_resultado)
        ode_system_func = details.get("ode_system_func")
        if not ode_system_func:
            print("No se pudo generar un modelo para esta solución.")
            continue
        initial_conditions_from_data = data[0]
        discovered_solution = solve_ivp(ode_system_func, [time_points[0], time_points[-1]], initial_conditions_from_data, t_eval=time_points, method='BDF')
        if not discovered_solution.success:
            print(f"\n[!] ADVERTENCIA: La simulación para la solución #{i+1} falló: {discovered_solution.message}")
            continue
        discovered_data = discovered_solution.y.T
        validation_mse = np.mean((discovered_data - data)**2)
        print(f"\n[!] MSE de Validación (vs datos originales): {validation_mse:.8f}")
        with open(f'{analysis_filename}.txt', 'a') as f:
            f.write(f"\n--- INFORME DE LA SOLUCIÓN #{i+1} ---\n")
            f.write(texto_resultado)
            f.write(f"\n\nMSE de Validación (vs datos originales): {validation_mse:.8f}\n")
            f.write("-" * 40 + "\n")

        # --- 3. MODIFICACIÓN: Generar, GUARDAR y CERRAR la gráfica ---
        print("[+] Generando y guardando gráfica de comparación...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Dibujar los datos
        for j in range(num_nodes):
            ax.plot(time_points, data[:, j], 'o', markersize=5, alpha=0.5, label=f'Datos X{j}')
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for j in range(num_nodes):
            ax.plot(time_points, discovered_data[:, j], '-', linewidth=2.5, color=colors[j % len(colors)], label=f'Modelo Descubierto X{j}')
        
        # Configurar la gráfica
        ax.set_title(f'Análisis de la Solución #{i+1}\nFitness: {details["fitness"]:.6f} | MSE (vs Original): {validation_mse:.6f}')
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Población")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Guardar la figura en un archivo
        plot_filename = f"{analysis_filename}__solution_{i+1}.png"
        full_plot_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(full_plot_path, dpi=150) # dpi=150 para una buena resolución
        print(f"    -> Gráfica guardada en: '{full_plot_path}'")
        
        # Cerrar la figura para liberar memoria
        plt.close(fig)

    print(f"\n✅ Análisis completo. Los resultados de texto se han guardado en '{analysis_filename}.txt' y los gráficos en la carpeta '{plot_dir}'.")
    # <-- 4. ELIMINAR plt.show() para que no se muestren las ventanas
    # plt.show() 