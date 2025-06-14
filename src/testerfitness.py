import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from fitness import*
import itertools

# Asumimos que tu clase CompartmentalGraphFitness está en el mismo archivo o importada
# from your_file import CompartmentalGraphFitness

# =============================================================================
# MODELOS EPIDEMIOLÓGICOS
# =============================================================================

def sir_model(t, y, beta, gamma):

    S, I, R = y

    S, I, R = max(S, 0), max(I, 0), max(R, 0)

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]

def seir_model(t, y, beta, sigma, gamma):

    S, E, I, R = y

    S, E, I, R = max(S, 0), max(E, 0), max(I, 0), max(R, 0)

    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]

def sird_model(t, y, beta, gamma, mu):

    S, I, R, D = y

    S, I, R, D = max(S, 0), max(I, 0), max(R, 0), max(D, 0)

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I

    return [dSdt, dIdt, dRdt, dDdt]

def seirv_model(t, y, beta, sigma, gamma, nu):

    S, E, I, R, V = y

    S, E, I, R, V = max(S, 0), max(E, 0), max(I, 0), max(R, 0), max(V, 0)

    dSdt = -beta * S * I - nu * S
    dEdt = beta * S * I
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S

    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def siqrd_model(t, y, beta, gamma, delta, mu, eta, kappa):

    S, I, Q, R, D = y

    S, I, Q, R, D = max(S, 0), max(I, 0), max(Q, 0), max(R, 0), max(D, 0)

    dSdt = -beta * S * I
    dIdt = beta * S * I - (gamma + delta + mu) * I
    dQdt = delta * I - (eta + kappa) * Q
    dRdt = gamma * I + eta * Q
    dDdt = mu * I + kappa * Q
    
    return [dSdt, dIdt, dQdt, dRdt, dDdt]

def svv_eir_model(t, y, beta, sigma, gamma, nu1, nu2, epsilon1, epsilon2):

    S, V1, V2, E, I, R = y

    S, V1, V2, E, I, R = max(S, 0), max(V1, 0), max(V2, 0), max(E, 0), max(I, 0), max(R, 0)


    infection_from_S = beta * S * I
    infection_from_V1 = epsilon1 * beta * V1 * I
    infection_from_V2 = epsilon2 * beta * V2 * I
    total_new_exposed = infection_from_S + infection_from_V1 + infection_from_V2

    dSdt = -infection_from_S - nu1 * S
    dV1dt = nu1 * S - infection_from_V1 - nu2 * V1
    dV2dt = nu2 * V1 - infection_from_V2
    dEdt = total_new_exposed - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dV1dt, dV2dt, dEdt, dIdt, dRdt]

# =============================================================================
# FUNCIONES DE AYUDA PARA EL TESTER
# =============================================================================

def generate_and_visualize_data(scenario_name, model_func, y0, t_points, params, compartment_names, noise_level=0.01):
    """Resuelve EDOs, añade ruido y visualiza los datos."""
    print(f"\n[+] Generando datos para: {scenario_name}")
    t_span = [t_points[0], t_points[-1]]
    
    solution = solve_ivp(
        model_func, 
        t_span, 
        y0, 
        t_eval=t_points, 
        args=tuple(params.values()), # Pasa los parámetros al solver
        method='BDF' # BDF es robusto para estos modelos
    )
    clean_data = solution.y.T
    
    np.random.seed(42)
    noisy_data = clean_data + np.random.normal(0, noise_level, clean_data.shape)
    noisy_data = np.maximum(noisy_data, 0)

    plt.figure(figsize=(10, 6))
    for i in range(noisy_data.shape[1]):
        plt.plot(t_points, noisy_data[:, i], 'o', markersize=4, label=f'{compartment_names[i]}')
    
    plt.title(f'Datos Sintéticos para {scenario_name}', fontsize=16)
    plt.xlabel('Tiempo')
    plt.ylabel('Proporción de Población')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()
    
    return noisy_data

def define_candidate_graphs(true_graph, n_compartments, model_name):
    """Define un conjunto de grafos candidatos para un escenario."""
    graphs = {}
    
    # 1. El grafo verdadero
    graphs[f'Grafo Correcto ({model_name})'] = true_graph
    
    # 2. Grafo de cadena lineal simple
    linear_edges = [(i, i + 1, None) for i in range(n_compartments - 1)]
    graphs['Grafo de Cadena Lineal'] = {'nodes': list(range(n_compartments)), 'edges': linear_edges}

    # 3. Grafo con un enlace faltante (si es posible)
    if len(true_graph['edges']) > 1:
        missing_link_edges = true_graph['edges'][:-1]
        graphs['Grafo con Enlace Faltante'] = {'nodes': list(range(n_compartments)), 'edges': missing_link_edges}

    # 4. Grafo con un enlace extra (incorrecto)
    extra_link_edges = true_graph['edges'][:]
    # Añadir un enlace plausible pero incorrecto, p.ej., R -> I
    if n_compartments > 2:
        extra_link_edges.append((n_compartments - 1, 1, None)) 
        graphs['Grafo con Enlace Extra'] = {'nodes': list(range(n_compartments)), 'edges': extra_link_edges}

    # 5. Grafo en estrella (un nodo central, aquí el nodo 1 'I')
    star_edges = []
    for i in range(n_compartments):
        if i != 1:
            star_edges.append((i, 1, None))
            star_edges.append((1, i, None))
    graphs['Grafo en Estrella (central=I)'] = {'nodes': list(range(n_compartments)), 'edges': list(set(star_edges))}


    return graphs


# =============================================================================
# SCRIPT PRINCIPAL DE EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    
    # --- Definición de Escenarios de Prueba ---
    
    scenarios_to_test = [
        {
            'name': 'Modelo SIR',
            'model_func': sir_model,
            'compartments': ['S', 'I', 'R'],
            'params': {'beta': 0.4, 'gamma': 0.1},
            'y0': [0.99, 0.01, 0.0],
            'true_graph': {'nodes': [0, 1, 2], 'edges': [(0, 1, None), (1, 2, None)]},
            'conserve_population': True
        },
        {
            'name': 'Modelo SEIR',
            'model_func': seir_model,
            'compartments': ['S', 'E', 'I', 'R'],
            'params': {'beta': 0.5, 'sigma': 0.2, 'gamma': 0.1},
            'y0': [0.99, 0.0, 0.01, 0.0],
            'true_graph': {'nodes': [0, 1, 2, 3], 'edges': [(0, 1, None), (1, 2, None), (2, 3, None)]},
            'conserve_population': True
        },
        {
            'name': 'Modelo SIRD',
            'model_func': sird_model,
            'compartments': ['S', 'I', 'R', 'D'],
            'params': {'beta': 0.3, 'gamma': 0.05, 'mu': 0.02},
            'y0': [0.99, 0.01, 0.0, 0.0],
            'true_graph': {'nodes': [0, 1, 2, 3], 'edges': [(0, 1, None), (1, 2, None), (1, 3, None)]},
            'conserve_population': False # La población no se conserva debido a D
        },
        {
            'name': 'Modelo SEIRV',
            'model_func': seirv_model,
            'compartments': ['S', 'E', 'I', 'R', 'V'],
            'params': {'beta': 0.6, 'sigma': 0.1, 'gamma': 0.1, 'nu': 0.05},
            'y0': [0.99, 0.0, 0.01, 0.0, 0.0],
            'true_graph': {'nodes': [0,1,2,3,4], 'edges': [(0, 1, None), (0, 4, None), (1, 2, None), (2, 3, None)]},
            'conserve_population': True # S+E+I+R+V es constante
        }
    ]

    t_points = np.linspace(0, 150, 150)

    # --- Bucle Principal de Pruebas ---

     # Bucle principal que itera sobre los escenarios
    for scenario in scenarios_to_test:
        n_compartments = len(scenario['compartments'])
        
        print("\n" + "="*80)
        print(f"INICIANDO PRUEBA PARA: {scenario['name']} ({n_compartments} compartimentos)")
        print("="*80)

        # 1. Generar datos (sin cambios aquí)
        noisy_data = generate_and_visualize_data(
            scenario_name=scenario['name'],
            model_func=scenario['model_func'],
            y0=scenario['y0'],
            t_points=t_points,
            params=scenario['params'],
            compartment_names=scenario['compartments']
        )
        
        # 2. Definir grafos candidatos (sin cambios aquí)
        candidate_graphs = define_candidate_graphs(scenario['true_graph'], n_compartments, scenario['name'])
        
        # 3. Configurar el evaluador (sin cambios aquí)
        evaluator = CompartmentalGraphFitness(
            data=noisy_data,
            time_points=t_points,
            threshold=1.2,
            conserve_population=scenario['conserve_population'],
            enforce_non_negative=True,
            debug=False # Poner en False para una salida limpia
        )

        # 4. Evaluar todos los grafos candidatos
        print(f"\n[*] Evaluando {len(candidate_graphs)} grafos candidatos para {scenario['name']}...")
        results = []
        for name, graph in candidate_graphs.items():
            print(f"    - Probando '{name}'...")
            
            # --- CAMBIO 1: Usar evaluate_with_details y guardar el diccionario completo ---
            details = evaluator.evaluate_with_details(graph)
            details['name'] = name # Añadimos el nombre del grafo a los detalles
            results.append(details)
            
        # 5. Presentar el ranking de resultados para el escenario actual
        print("\n" + "-"*80)
        print(f"🏆 RANKING DE GRAFOS PARA: {scenario['name']} 🏆")
        print("(menor fitness es mejor)")
        
        results_sorted = sorted(results, key=lambda x: x['fitness'])
        
        # --- CAMBIO 2: Modificar la presentación para mostrar todos los detalles ---
        for i, res in enumerate(results_sorted):
            trophy = "🥇" if i == 0 else "  "
            
            # Imprimir una cabecera clara para cada resultado en el ranking
            print("\n" + "~"*80)
            print(f"{trophy} RANKING #{i+1} | CANDIDATO: '{res['name']}'")
            print("~"*80)

            # Usar el método de formato para obtener el string con todos los detalles
            texto_del_resultado = evaluator.format_result_as_string(res)
            
            # Imprimir el resultado formateado
            print(texto_del_resultado)
        
        print("\n" + "="*80)

    print("\nTODAS LAS PRUEBAS HAN FINALIZADO.")