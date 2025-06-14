# generate_all_proportional_noise_datasets.py
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# --------------------------------------------------------------------------
#                            1. MODELOS
# --------------------------------------------------------------------------
# (Las definiciones de los modelos: sir_model, seir_model, etc., van aquí sin cambios)
def sir_model(t, y, beta, gamma):
    S, I, R = y; S, I, R = max(S, 0), max(I, 0), max(R, 0)
    dSdt = -beta * S * I; dIdt = beta * S * I - gamma * I; dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y; S, E, I, R = max(S, 0), max(E, 0), max(I, 0), max(R, 0)
    dSdt = -beta * S * I; dEdt = beta * S * I - sigma * E; dIdt = sigma * E - gamma * I; dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def sird_model(t, y, beta, gamma, mu):
    S, I, R, D = y; S, I, R, D = max(S, 0), max(I, 0), max(R, 0), max(D, 0)
    dSdt = -beta * S * I; dIdt = beta * S * I - gamma * I - mu * I; dRdt = gamma * I; dDdt = mu * I
    return [dSdt, dIdt, dRdt, dDdt]

def seirv_model(t, y, beta, sigma, gamma, nu):
    S, E, I, R, V = y; S, E, I, R, V = max(S, 0), max(E, 0), max(I, 0), max(R, 0), max(V, 0)
    dSdt = -beta * S * I - nu * S; dEdt = beta * S * I; dIdt = sigma * E - gamma * I; dRdt = gamma * I; dVdt = nu * S
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def siqrd_model(t, y, beta, gamma, delta, mu, eta, kappa):
    S, I, Q, R, D = y; S, I, Q, R, D = max(S, 0), max(I, 0), max(Q, 0), max(R, 0), max(D, 0)
    dSdt = -beta * S * I; dIdt = beta * S * I - (gamma + delta + mu) * I; dQdt = delta * I - (eta + kappa) * Q; dRdt = gamma * I + eta * Q; dDdt = mu * I + kappa * Q
    return [dSdt, dIdt, dQdt, dRdt, dDdt]

def svv_eir_model(t, y, beta, sigma, gamma, nu1, nu2, epsilon1, epsilon2):
    S, V1, V2, E, I, R = y; S, V1, V2, E, I, R = max(S, 0), max(V1, 0), max(V2, 0), max(E, 0), max(I, 0), max(R, 0)
    infection_from_S = beta * S * I; infection_from_V1 = epsilon1 * beta * V1 * I; infection_from_V2 = epsilon2 * beta * V2 * I
    total_new_exposed = infection_from_S + infection_from_V1 + infection_from_V2
    dSdt = -infection_from_S - nu1 * S; dV1dt = nu1 * S - infection_from_V1 - nu2 * V1; dV2dt = nu2 * V1 - infection_from_V2; dEdt = total_new_exposed - sigma * E; dIdt = sigma * E - gamma * I; dRdt = gamma * I
    return [dSdt, dV1dt, dV2dt, dEdt, dIdt, dRdt]

# --------------------------------------------------------------------------
#                       2. CONFIGURACIÓN DE ESCENARIOS
# --------------------------------------------------------------------------
TIME_SETTINGS = {'start': 0, 'end': 150, 'points': 150}
# Estos niveles ahora representan el 'factor_ruido' o 'max_noise'
NOISE_LEVELS = [0.0, 0.015, 0.030] 

SCENARIOS = [
    {"name": "SIR", "model_func": sir_model, "initial_conditions": [0.99, 0.01, 0.0], "parameters": {'beta': 0.4, 'gamma': 0.1}},
    {"name": "SEIR", "model_func": seir_model, "initial_conditions": [0.99, 0.0, 0.01, 0.0], "parameters": {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1}},
    {"name": "SIRD", "model_func": sird_model, "initial_conditions": [0.99, 0.01, 0.0, 0.0], "parameters": {'beta': 0.4, 'gamma': 0.08, 'mu': 0.02}},
    {"name": "SEIRV", "model_func": seirv_model, "initial_conditions": [0.99, 0.0, 0.01, 0.0, 0.0], "parameters": {'beta': 0.5, 'sigma': 0.2, 'gamma': 0.1, 'nu': 0.05}},
    {"name": "SIQRD", "model_func": siqrd_model, "initial_conditions": [0.99, 0.01, 0.0, 0.0, 0.0], "parameters": {'beta': 0.6, 'gamma': 0.1, 'delta': 0.1, 'mu': 0.01, 'eta': 0.05, 'kappa': 0.02}},
    {"name": "SVVEIR", "model_func": svv_eir_model, "initial_conditions": [0.99, 0.0, 0.0, 0.0, 0.01, 0.0], "parameters": {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1, 'nu1': 0.05, 'nu2': 0.1, 'epsilon1': 0.5, 'epsilon2': 0.1}}
]

# --------------------------------------------------------------------------
#                       3. LÓGICA DE GENERACIÓN
# --------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*50)
    print("INICIANDO GENERACIÓN CON RUIDO PROPORCIONAL")
    print("="*50)
    
    total_files = 0
    np.random.seed(42) # Fijar la semilla una sola vez para la reproducibilidad

    for scenario in SCENARIOS:
        # La simulación se hace una vez por escenario
        model_name, model_func, y0, params = scenario["name"], scenario["model_func"], scenario["initial_conditions"], scenario["parameters"]
        t_points = np.linspace(TIME_SETTINGS['start'], TIME_SETTINGS['end'], TIME_SETTINGS['points'])
        
        print(f"\n[*] Simulando modelo base: {model_name}")
        solution = solve_ivp(
            model_func, 
            [TIME_SETTINGS['start'], TIME_SETTINGS['end']], 
            y0, 
            t_eval=t_points, 
            args=tuple(params.values())
        )
        clean_data = solution.y.T

        # Se aplica el ruido y se guarda un archivo para cada nivel de ruido
        for noise_factor in NOISE_LEVELS:
            print(f"    -> Aplicando ruido proporcional de {noise_factor*100:.1f}%...")
            
            # --- CAMBIO CLAVE: Aplicando la nueva fórmula de ruido ---
            # y_ruidoso = y_limpio + y_limpio * factor_ruido * random_standard_normal()
            noisy_data = clean_data + clean_data * noise_factor * np.random.normal(0, 1, clean_data.shape)

            # Lógica para guardar el archivo
            num_compartments = len(y0)
            column_names = ['time'] + [f'X{i}' for i in range(num_compartments)]
            final_data_matrix = np.c_[t_points, noisy_data]
            df = pd.DataFrame(final_data_matrix, columns=column_names)
            
            noise_str = str(noise_factor).replace('.', 'p')
            filename = f"{model_name}_noise_{noise_str}.csv"
            full_path = os.path.join(output_dir, filename)
            
            df.to_csv(full_path, index=False)
            print(f"        -> Guardado en: '{full_path}'")
            total_files += 1

    print("\n" + "="*50)
    print(f"✅ Proceso finalizado. Se han generado {total_files} archivos CSV en la carpeta '{output_dir}'.")
    print("="*50)