from genetic_experiment import*

# --- Bloque principal para ejecutar el script ---
if __name__ == "__main__":
    # Define aquí la configuración de un experimento específico
    experiment_config_sir_model = {
        "input_filename": "src/data/SIR_noise_0p0.csv",
        "analysis_filename": "SIR_noise_0p0_exp1",
        "ga_params": {
            "population_size": 30,
            "num_generations": 20,
            "mutation_rate": 0.8,
            "crossover_rate": 0.9
        },
        "fitness_params": {
            "threshold": 1.0,
            "conserve_population": True
        }
    }
    
    # Llama a la función principal con la configuración
    run_genetic_experiment(
        input_filename=experiment_config_sir_model["input_filename"],
        analysis_filename=experiment_config_sir_model["analysis_filename"],
        ga_params=experiment_config_sir_model["ga_params"],
        fitness_params=experiment_config_sir_model["fitness_params"]
    )