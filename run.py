from genetic_experiment import*

# --- Bloque principal para ejecutar el script ---
if __name__ == "__main__":
    # Define aquí la configuración de un experimento específico
    experiment_config = {
        "input_filename": "src/sir_data.csv",
        "analysis_filename": "analysis_report_SIR",
        "ga_params": {
            "population_size": 30,
            "num_generations": 20,
            "mutation_rate": 0.8,
            "crossover_rate": 0.9
        },
        "fitness_params": {
            "threshold": 0.8,
            "conserve_population": True
        }
    }
    
    # Llama a la función principal con la configuración
    run_genetic_experiment(
        input_filename=experiment_config["input_filename"],
        analysis_filename=experiment_config["analysis_filename"],
        ga_params=experiment_config["ga_params"],
        fitness_params=experiment_config["fitness_params"]
    )