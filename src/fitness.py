import numpy as np
import networkx as nx
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class CompartmentalGraphFitness:
    """
    Función fitness mejorada para grafos compartimentales que:
    1. Genera ecuaciones completas (términos lineales y cuadráticos) para cada nodo
    2. Ajusta coeficientes para estas ecuaciones
    3. Elimina términos con coeficientes menores que un umbral
    4. Identifica términos compartidos entre nodos conectados por aristas
    5. Reconstruye ecuaciones incluyendo SOLO términos compartidos en las aristas correspondientes
    6. Reajusta los coeficientes mediante minimos cuadrados
    7. Calcula el fitness comparando con los datos originales
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 time_points: np.ndarray,
                 threshold: float = 0.0001,
                 conserve_population: bool = True,
                 enforce_non_negative: bool = True,
                 debug: bool = False):
        """
        Inicializa el evaluador de fitness.
        
        Args:
            data: Array de forma (n_tiempo, n_compartimentos) con los datos de población
            time_points: Array con los puntos de tiempo correspondientes
            threshold: Umbral para considerar un coeficiente como significativo
            conserve_population: Si es True, penaliza modelos que no conservan la población total
            enforce_non_negative: Si es True, penaliza soluciones que producen valores negativos
            debug: Si es True, imprime información de depuración
        """
        self.data = np.array(data)
        self.time_points = np.array(time_points)
        self.n_compartments = self.data.shape[1]
        self.threshold = threshold
        self.conserve_population = conserve_population
        self.enforce_non_negative = enforce_non_negative
        self.debug = debug
        
        # Crear variables simbólicas para cada compartimento
        self.symbols = sp.symbols([f'X{i}' for i in range(self.n_compartments)])
        
        # Calcular derivadas numéricas de los datos
        self.derivatives = self._calculate_derivatives()
    
    def _debug_print(self, message):
        """Imprime un mensaje de depuración si el modo debug está activado"""
        if self.debug:
            print(message)
    
    def _calculate_derivatives(self) -> np.ndarray:
        """
        Calcula las derivadas numéricas de los datos usando diferencias finitas centradas.
        
        Returns:
            Array de forma (n_tiempo-1, n_compartimentos) con las derivadas
        """
        # Usar diferencias finitas centradas para más precisión
        dt = np.diff(self.time_points)
        y_diff = np.diff(self.data, axis=0)
        
        # Normalizar por dt
        derivatives = y_diff / dt[:, np.newaxis]
        
        return derivatives
    

    def evaluate_with_details(self, graph: Dict) -> Dict:
        """
        [NUEVO MÉTODO PRINCIPAL]
        Evalúa un grafo y devuelve un diccionario completo con los resultados,
        incluyendo fitness, grafo y las ecuaciones simbólicas finales.
        
        Args:
            graph: Diccionario que representa el grafo.
                
        Returns:
            Un diccionario con 'fitness', 'graph', y 'final_equations'.
        """
        try:
            self._debug_print(f"\nEvaluando grafo con {len(graph['edges'])} aristas: {graph['edges']}")
            
            # Guardar la estructura del grafo original para usarla más tarde
            self.original_edges = [(s, t) for s, t, _ in graph['edges']]
            
            # Paso 1: Generar todos los posibles términos
            expanded_equations = self._generate_expanded_equations()
            
            # Paso 2: Ajustar coeficientes iniciales
            coefficients = self._fit_coefficients(expanded_equations)
            
            # Paso 3: Podar términos insignificantes
            pruned_equations = self._prune_insignificant_terms(expanded_equations, coefficients)
            
            # Paso 4: Identificar términos compartidos
            shared_terms = self._identify_relevant_shared_terms(pruned_equations, graph)
            
            # Paso 5: Reconstruir ecuaciones con contexto
            reconstructed_equations = self._reconstruct_equations_correctly(shared_terms, graph)

            # Paso 6: Reajustar coeficientes finales con el método global
            global_coeffs = self._refit_coefficients(reconstructed_equations)
            
            # Paso 7: Construir y resolver el sistema de EDOs
            # --- CAMBIO EN LA LLAMADA ---
            ode_system, symbolic_eqs = self._build_ode_system(reconstructed_equations, global_coeffs)
        
            
            # Imprimir ecuaciones finales si estamos en modo debug
            if self.debug:
                print("\nEcuaciones finales del sistema:")
                for i, eq in enumerate(symbolic_eqs):
                    print(f"dX{i}/dt = {eq}")
            
            # Paso 8: Integrar y calcular el fitness final
            mse = self._integrate_and_calculate_error(ode_system, symbolic_eqs, graph['edges'])

            # --- MODIFICACIÓN CLAVE: EMPAQUETAR RESULTADOS ---
            results = {
                "fitness": mse,
                "graph": graph,
                "final_equations": symbolic_eqs,
                "ode_system_func": ode_system  
            }
            self._debug_print(f"Fitness calculado: {mse:.6f}")
            return results
            
        except Exception as e:
            if self.debug:
                print(f"Error en evaluate_with_details: {e}")
                import traceback
                traceback.print_exc()
            # Devolver un resultado de error consistente
            return {
                "fitness": 1e10,
                "graph": graph,
                "final_equations": []
            }
    

    def evaluate(self, graph: Dict) -> float:
        """
        Evalúa el fitness de un grafo con penalización mejorada por complejidad.
        
        Args:
            graph: Diccionario que representa el grafo con:
                - 'nodes': Lista de nodos (compartimentos)
                - 'edges': Lista de tuplas (origen, destino, término)
                
        Returns:
            Valor de fitness (menor es mejor)
        """
        try:
            self._debug_print(f"\nEvaluando grafo con {len(graph['edges'])} aristas: {graph['edges']}")
            
            # Guardar la estructura del grafo original para usarla más tarde
            self.original_edges = [(s, t) for s, t, _ in graph['edges']]
            
            # Paso 1: Generar todos los posibles términos lineales y cuadráticos para cada nodo
            expanded_equations = self._generate_expanded_equations()
            
            # Paso 2: Ajustar coeficientes para estos términos (normalizados entre 0 y 1)
            coefficients = self._fit_coefficients(expanded_equations)
            
            # Paso 3: Podar términos insignificantes
            pruned_equations = self._prune_insignificant_terms(expanded_equations, coefficients)
            
            # Paso 4: Identificar términos compartidos entre nodos conectados por aristas
            shared_terms = self._identify_relevant_shared_terms(pruned_equations, graph)
            
            # Paso 5: Reconstruir un nuevo conjunto de ecuaciones basadas en los términos compartidos
            reconstructed_equations = self._reconstruct_equations_correctly(shared_terms, graph)
            
            # Paso 6: Reajustar los coeficientes para las ecuaciones reconstruidas
            adjusted_equations = self._refit_coefficients(reconstructed_equations)
            
            # Paso 7: Construir sistema de EDOs basado en ecuaciones reajustadas
            ode_system, symbolic_eqs = self._build_ode_system(adjusted_equations)
            
            # Imprimir ecuaciones finales para depuración
            if self.debug:
                print("\nEcuaciones finales del sistema:")
                for i, eq in enumerate(symbolic_eqs):
                    print(f"dX{i}/dt = {eq}")
            
            # Paso 8: Integrar el sistema y calcular fitness con penalización mejorada
            mse = self._integrate_and_calculate_error(ode_system, symbolic_eqs, graph['edges'])

            n_terms = sum(len(terms) for terms in reconstructed_equations)
            complexity_penalty = n_terms * 0.1
            mse += complexity_penalty
            
            self._debug_print(f"Fitness calculado: {mse:.6f}")
            return mse
            
        except Exception as e:
            if self.debug:
                print(f"Error en evaluate: {e}")
                import traceback
                traceback.print_exc()
            # Si hay error, asignar un valor de fitness muy malo
            return 1e10
    
    
    def _integrate_and_calculate_error(self, ode_system, symbolic_eqs, graph_edges, mse=0.0):
        """
        [MODIFICADO] Integra el sistema de ODEs usando solo métodos Runge-Kutta
        y calcula el error con penalización por complejidad.
        
        ADVERTENCIA: Esta versión es menos robusta al eliminar el fallback a BDF para sistemas rígidos.
        """
        
        # Decidir el método basado en el número de compartimentos (NO RECOMENDADO)
        if self.n_compartments == 3:
            method = 'RK23'
            self._debug_print("Sistema de 3 ecuaciones. Usando método RK23.")
        else:
            # Usar RK45 para cualquier otro número de ecuaciones (4, 5, 6, etc.)
            method = 'RK45'
            self._debug_print(f"Sistema de {self.n_compartments} ecuaciones. Usando método RK45.")

        try:
            # Intentar integrar el sistema con el método Runge-Kutta seleccionado
            solution = solve_ivp(
                ode_system,
                [self.time_points[0], self.time_points[-1]],
                self.data[0],
                t_eval=self.time_points,
                method=method, # Método dinámico
                rtol=1e-3,
                atol=1e-6,
                max_step=1.0  # Un max_step puede ayudar a prevenir que el solver se "escape"
            )
            
            # Comprobar si la integración fue exitosa
            if not solution.success:
                self._debug_print(f"La integración con {method} falló. Mensaje: {solution.message}")
                # Si no tuvo éxito, se considera un error y se asigna una penalización alta.
                return 1e9

            # Calcular error cuadrático medio
            mse = np.mean((solution.y.T - self.data) ** 2)
            
        except Exception as e:
            # Si ocurre cualquier excepción durante la integración (muy probable con sistemas rígidos)
            self._debug_print(f"Error fatal en integración con {method}: {e}. Asignando penalización alta.")
            return 1e9 # Devolver una penalización muy alta
        
        # --- El resto del código de penalizaciones permanece igual ---

        # Aplicar penalizaciones comunes
        if self.conserve_population:
            total_mass_input = np.sum(self.data[0])
            total_mass_output = np.sum(solution.y, axis=0)
            conservation_error = np.var(total_mass_output) / total_mass_input
            mse += conservation_error * 10
        
        if self.enforce_non_negative:
            negative_penalty = np.sum(np.minimum(solution.y, 0) ** 2)
            mse += negative_penalty * 5
        
        # Penalización por complejidad
        n_edges = len(graph_edges)
        max_edges = self.n_compartments * (self.n_compartments - 1)
        if max_edges > 0:
            penalty_factor = 0.5 * (n_edges / max_edges) ** 2
            penalized_mse = mse * (1.0 + penalty_factor)
        else:
            penalized_mse = mse

        if n_edges == max_edges and max_edges > 0:
            penalized_mse *= 1.5

        return penalized_mse
    
    def _generate_expanded_equations(self) -> List[List[sp.Expr]]:
        """
        Genera ecuaciones expandidas con todos los posibles términos lineales y cuadráticos
        para cada compartimento.
        
        Returns:
            Lista de listas, donde cada lista interior contiene los términos para un compartimento
        """
        expanded_equations = []
        
        for _ in range(self.n_compartments):
            compartment_terms = []
            
            # Añadir términos lineales
            for i in range(self.n_compartments):
                compartment_terms.append(self.symbols[i])
            
            # Añadir términos cuadráticos
            for i in range(self.n_compartments):
                for j in range(i, self.n_compartments):
                    compartment_terms.append(self.symbols[i] * self.symbols[j])
            
            expanded_equations.append(compartment_terms)
        
        return expanded_equations
    
    
    def _fit_coefficients(self, expanded_equations: List[List[sp.Expr]]) -> List[np.ndarray]:
        """
        Ajusta coeficientes para las ecuaciones expandidas.
        
        Args:
            expanded_equations: Lista de listas de términos simbólicos
            
        Returns:
            Lista de arrays de coeficientes, uno por compartimento
        """
        coefficients = []
        
        # Para cada compartimento
        for compartment in range(self.n_compartments):
            # Construir la matriz de diseño X
            X = np.zeros((len(self.derivatives), len(expanded_equations[compartment])))
            
            for t in range(len(self.derivatives)):
                for j, term in enumerate(expanded_equations[compartment]):
                    # Evaluar el término en el punto de tiempo t
                    term_val = self._evaluate_term(term, self.data[t])
                    X[t, j] = term_val
            
            # Vector de derivadas para este compartimento
            y = self.derivatives[:, compartment]
            
            # Resolver el sistema de ecuaciones lineales
            try:
                coeffs, residuals, rank, s = la.lstsq(X, y, rcond=None)
                
                # Normalizar coeficientes entre 0 y 1
                # normalized_coeffs = self._normalize_coefficients(coeffs)
                # coefficients.append(normalized_coeffs)
                coefficients.append(coeffs)
                
            except Exception as e:
                self._debug_print(f"Error en regresión lineal: {e}")
                # Usar coeficientes nulos como fallback
                coefficients.append(np.zeros(len(expanded_equations[compartment])))
        
        return coefficients
    
    def _evaluate_term(self, term: sp.Expr, values: np.ndarray) -> float:
        """
        Evalúa un término simbólico dado un conjunto de valores.
        
        Args:
            term: Término simbólico a evaluar
            values: Valores para cada compartimento
            
        Returns:
            Valor numérico del término
        """
        # Sustituir cada símbolo por su valor
        subs_dict = {self.symbols[i]: values[i] for i in range(self.n_compartments)}
        result = term.subs(subs_dict)
        return float(result)
    
    def _prune_insignificant_terms(self, expanded_equations: List[List[sp.Expr]], 
                                  coefficients: List[np.ndarray]) -> List[Dict[sp.Expr, float]]:
        """
        Elimina términos con coeficientes menores que el umbral.
        
        Args:
            expanded_equations: Lista de listas de términos simbólicos
            coefficients: Lista de arrays de coeficientes
            
        Returns:
            Lista de diccionarios {término: coeficiente} para términos significativos
        """
        pruned_equations = []
        
        for compartment in range(self.n_compartments):
            terms = expanded_equations[compartment]
            coeffs = coefficients[compartment]
            
            # Crear diccionario de término -> coeficiente para términos significativos
            significant_terms = {}
            for i in range(len(terms)):
                if abs(coeffs[i]) >= self.threshold:
                    significant_terms[terms[i]] = coeffs[i]
            
            pruned_equations.append(significant_terms)
            
            self._debug_print(f"Compartimento {compartment}: {len(significant_terms)} términos significativos de {len(terms)} originales")
        
        return pruned_equations
    



    def _identify_relevant_shared_terms(self, pruned_equations: List[Dict[sp.Expr, float]], 
                                      graph: Dict) -> Dict[Tuple[int, int], List[Tuple[sp.Expr, float, float]]]:
        """
        [VERSIÓN CORRECTA Y FINAL PARA ESTE PASO]
        Identifica términos compartidos entre nodos conectados por aristas.
        Su salida es consumida por el nuevo `_reconstruct_equations_correctly` que añade el contexto.
        
        Args:
            pruned_equations: Lista de diccionarios {término: coeficiente}
            graph: Grafo con nodos y aristas
            
        Returns:
            Diccionario {(source, target): [(término, coef_source, coef_target), ...]}
        """
        shared_terms = {}
        
        # Para cada arista en el grafo
        for source, target, _ in graph['edges']:
            # Obtener términos significativos para ambos nodos
            source_terms = pruned_equations[source]
            target_terms = pruned_equations[target]
            
            # 1. Encontrar todos los términos que aparecen en ambos nodos (comunes)
            common_terms_with_coeffs = []
            for term, coef_s in source_terms.items():
                if term in target_terms:
                    coef_t = target_terms[term]
                    common_terms_with_coeffs.append((term, coef_s, coef_t))
            
            # 2. Filtrar los términos comunes para mantener solo los relevantes para la arista
            relevant_terms = []
            source_var_str = f'X{source}'
            target_var_str = f'X{target}'

            for term, coef_s, coef_t in common_terms_with_coeffs:
                term_str = str(term)
                # La condición es: el término debe contener la variable del nodo origen O la del nodo destino
                if source_var_str in term_str or target_var_str in term_str:
                    relevant_terms.append((term, coef_s, coef_t))
            
            # La lista final de términos es el resultado de nuestro nuevo filtrado
            final_terms = relevant_terms
            
            # Mantener el fallback por si el filtrado no encuentra ningún término
            if not final_terms:
                # Crear un término que represente explícitamente el flujo entre los nodos
                default_term = self.symbols[source] * self.symbols[target]
                final_terms = [(default_term, 0.5, 0.5)]  # Coeficientes arbitrarios
                self._debug_print(f"Arista {source} -> {target}: No se encontraron términos relevantes, usando default {default_term}")

            
            shared_terms[(source, target)] = final_terms
            
            self._debug_print(f"Arista {source} -> {target}: {len(final_terms)} términos relevantes encontrados")
            for term, coef_s, coef_t in final_terms:
                self._debug_print(f"  Término: {term} (Coefs originales: S={coef_s:.4f}, T={coef_t:.4f})")
        
        # Guardar los términos compartidos como atributo de la clase
        self.shared_terms = shared_terms

        return shared_terms
    

    def _reconstruct_equations_correctly(self, 
                                       shared_terms: Dict[Tuple[int, int], List[Tuple[sp.Expr, float, float]]], 
                                       graph: Dict) -> List[List[Tuple[sp.Expr, Tuple[int, int]]]]:
        """
        [VERSIÓN MODIFICADA CON CONTEXTO]
        Reconstruye ecuaciones donde cada término va acompañado del contexto de la arista 
        que lo originó. Esto permite que un mismo término aparezca varias veces en una 
        ecuación si es relevante para diferentes aristas, cada uno con su propio coeficiente.
        
        Args:
            shared_terms: Diccionario {(source, target): [(término, ...), ...]}
            graph: Grafo con nodos y aristas
            
        Returns:
            Lista de listas de tuplas (término, contexto_arista), uno por compartimento.
        """
        # Inicializar ecuaciones reconstruidas vacías
        reconstructed_equations = [[] for _ in range(self.n_compartments)]
        
        # Para cada arista y sus términos compartidos
        for (source, target), terms_with_coefs in shared_terms.items():
            context = (source, target) # El contexto es la propia arista
            for term, _, _ in terms_with_coefs:
                term_with_context = (term, context)
                
                # Añadir el término con su contexto a los nodos de la arista.
                # Ya no se evitan duplicados, porque el contexto los hace únicos.
                reconstructed_equations[source].append(term_with_context)
                reconstructed_equations[target].append(term_with_context)
        
        # Verificar que cada nodo tenga al menos un término
        for i, terms in enumerate(reconstructed_equations):
            if not terms:
                # Si un nodo no tiene términos, añadir un término simple con un contexto de "auto-interacción"
                default_context = (i, i)
                reconstructed_equations[i].append((self.symbols[i], default_context))
                self._debug_print(f"Añadido término por defecto (X{i}, ({i},{i})) al nodo {i} que no tenía términos")
        
        for i, terms in enumerate(reconstructed_equations):
            self._debug_print(f"Nodo {i}: {len(terms)} términos con contexto en ecuación reconstruida.")

        return reconstructed_equations
    

    def _refit_coefficients(self, reconstructed_equations: List[List[Tuple[sp.Expr, Tuple[int, int]]]]) -> Dict[Tuple[sp.Expr, Tuple[int, int]], float]:
        """
        [VERSIÓN GLOBALMENTE RESTRINGIDA]
        Reajusta los coeficientes para todo el sistema a la vez, forzando que los
        términos de flujo compartido tengan el mismo coeficiente (con signo opuesto).
        
        Args:
            reconstructed_equations: Lista de listas de tuplas (término, contexto_arista)
            
        Returns:
            Un único diccionario que mapea cada término de interacción único 
            a su coeficiente globalmente optimizado.
        """
        # 1. Identificar todos los parámetros de interacción únicos en todo el modelo
        # Un parámetro es un (término, contexto_arista)
        unique_params = sorted(list(set(term for terms_list in reconstructed_equations for term in terms_list)), key=str)
        param_to_col = {param: i for i, param in enumerate(unique_params)}
        num_params = len(unique_params)

        if num_params == 0:
            return {}

        # 2. Construir el vector 'y' global apilando las derivadas de cada compartimento
        n_time_points = len(self.derivatives)
        y_global = self.derivatives.T.flatten()

        # 3. Construir la matriz de diseño 'X' global
        X_global = np.zeros((self.n_compartments * n_time_points, num_params))

        # Rellenar la matriz X global
        for compartment_idx, terms_with_context in enumerate(reconstructed_equations):
            for term_tuple in terms_with_context:
                term, context = term_tuple
                
                # Encontrar a qué columna de la matriz X pertenece este parámetro
                col_idx = param_to_col[term_tuple]
                
                # Evaluar el término a lo largo del tiempo
                term_values = np.array([self._evaluate_term(term, self.data[t]) for t in range(n_time_points)])
                
                # Determinar el signo. El flujo sale del 'source' (-) y entra al 'target' (+)
                source, target = context
                sign = 0
                if compartment_idx == source:
                    sign = -1.0
                if compartment_idx == target:
                    # Si es un término de auto-interacción (source=target), el signo neto es 0 para un flujo
                    # Si ya era negativo, se vuelve 0. Si era 0, se vuelve +1
                    sign += 1.0

                # Asignar los valores con el signo correcto en el bloque de la matriz
                # que corresponde a este compartimento
                start_row = compartment_idx * n_time_points
                end_row = (compartment_idx + 1) * n_time_points
                X_global[start_row:end_row, col_idx] += sign * term_values
        
        # 4. Resolver el sistema de regresión lineal global
        try:
            coeffs, _, _, _ = la.lstsq(X_global, y_global, rcond=None)
            # Nos interesan solo los valores absolutos, el signo se aplica después
            coeffs = np.abs(coeffs)
        except Exception as e:
            self._debug_print(f"Error en regresión lineal global: {e}")
            coeffs = np.full(num_params, 0.5) # Fallback

        # 5. Crear el diccionario final de parámetro -> coeficiente
        final_coeffs = {param: coeffs[i] for param, i in param_to_col.items()}
        return final_coeffs

    def _build_ode_system(self, adjusted_equations: List[Dict], global_coeffs: Dict) -> Tuple[callable, List[sp.Expr]]:
        """
        [VERSIÓN MODIFICADA]
        Construye el sistema de EDOs usando los coeficientes globales restringidos.
        """
        symbolic_equations = [0 for _ in range(self.n_compartments)]
        
        for i, terms_list in enumerate(adjusted_equations):
            equation_for_i = 0
            for term_tuple in terms_list:
                term, context = term_tuple
                
                # Buscar el coeficiente global para este término de interacción
                k = global_coeffs.get(term_tuple, 0)
                
                # Determinar el signo
                source, target = context
                sign = 0
                if i == source: sign -= 1
                if i == target: sign += 1

                equation_for_i += sign * k * term
                    
            symbolic_equations[i] = equation_for_i

        # El resto del método para convertir a función numérica no cambia
        try:
            equation_funcs = [sp.lambdify(self.symbols, eq, "numpy") for eq in symbolic_equations]
            def ode_system(t, y):
                try:
                    y_safe = np.maximum(y, 0.0)
                    dy_dt = [eq_func(*y_safe) for eq_func in equation_funcs]
                    for i in range(len(y)):
                        if y[i] < 1e-9 and dy_dt[i] < 0:
                            dy_dt[i] = 0.0
                    return dy_dt
                except Exception as e:
                    self._debug_print(f"Error al evaluar el sistema de EDOs: {e}")
                    return [0] * self.n_compartments
            
        except Exception as e:
            self._debug_print(f"Error al convertir ecuaciones simbólicas a funciones: {e}")
            def ode_system(t, y): return [0] * self.n_compartments
                
        return ode_system, symbolic_equations
        
    
    def print_equations(self, graph: Dict) -> None:
        """
        Imprime las ecuaciones diferenciales del grafo.
        
        Args:
            graph: Diccionario que representa el grafo
        """
        try:
            details = self.evaluate_with_details(graph)
            
            print("Ecuaciones originales expandidas:")
            for eq in details['expanded_equations']:
                print(eq)
            
            print("\nEcuaciones podadas con coeficientes significativos:")
            for eq in details['pruned_equations']:
                print(eq)
            
            print("\nTérminos compartidos entre nodos conectados:")
            for edge, terms in details['shared_terms'].items():
                print(f"{edge}: {', '.join(terms) if terms else 'Ninguno'}")
            
            print("\nEcuaciones reconstruidas (solo con términos de aristas conectadas):")
            for eq in details['reconstructed_equations']:
                print(eq)
            
            print("\nEcuaciones con coeficientes reajustados:")
            for eq in details['adjusted_equations']:
                print(eq)
            
            print("\nEcuaciones compartimentales finales:")
            for eq in details['symbolic_equations']:
                print(eq)
                
        except Exception as e:
            print(f"Error al imprimir ecuaciones: {e}")

    def format_result_as_string(self, details: Dict) -> str:
        """
        [NUEVO MÉTODO DE FORMATO]
        Toma el diccionario de resultados de `evaluate_with_details` y lo
        convierte en una cadena de texto legible, ideal para imprimir o guardar.

        Args:
            details: El diccionario devuelto por evaluate_with_details.

        Returns:
            Una cadena de texto (string) con los resultados formateados.
        """
        if not details or not details.get("final_equations"):
            return "No se pudo generar un resultado válido."

        # Extraer datos del diccionario
        fitness = details["fitness"]
        graph = details["graph"]
        equations = details["final_equations"]
        
        lines = []
        
        # Añadir el Fitness
        lines.append("="*30)
        lines.append("  RESULTADO DEL MODELO")
        lines.append("="*30)
        lines.append(f"Fitness (Error Cuadrático Medio): {fitness:.8f}\n")
        
        # Añadir la estructura del grafo
        lines.append("Estructura del Grafo (Aristas):")
        if not graph['edges']:
            lines.append("  (Sin aristas)")
        else:
            for source, target, _ in graph['edges']:
                lines.append(f"  {source} -> {target}")
        lines.append("") # Línea en blanco para separar

        # Añadir las ecuaciones finales
        lines.append("Ecuaciones Diferenciales Finales:")
        for i, eq in enumerate(equations):
            lines.append(f"  d(X{i})/dt = {eq}")
        
        lines.append("="*30)

        return "\n".join(lines)
    
    def visualize_results(self, graph: Dict, figsize=(12, 8)):
        """
        Visualiza los resultados de la evaluación.
        
        Args:
            graph: Diccionario que representa el grafo
            figsize: Tamaño de la figura
            
        Returns:
            Figura de matplotlib
        """
        try:
            details = self.evaluate_with_details(graph)
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Graficar datos originales y simulación
            for i in range(self.n_compartments):
                # Datos originales
                ax1.scatter(self.time_points, self.data[:, i], label=f'X{i} (datos)', 
                           marker='o', s=30, alpha=0.7)
                
                # Simulación
                ax1.plot(details['simulation']['t'], details['simulation']['y'][i], 
                        label=f'X{i} (modelo)', linewidth=2)
            
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Población')
            ax1.set_title(f'Comparación Datos vs. Modelo (Fitness: {details["fitness"]:.6f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Crear grafo de NetworkX
            G = nx.DiGraph()
            
            # Añadir nodos
            for i in range(self.n_compartments):
                G.add_node(i, label=f'X{i}')
            
            # Añadir aristas con etiquetas
            edge_labels = {}
            for edge_str, terms in details['shared_terms'].items():
                s, t = map(int, edge_str.split(' -> '))
                label = terms[0].split(':')[0] if terms else ""  # Mostrar solo el primer término
                G.add_edge(s, t)
                if terms:
                    edge_labels[(s, t)] = label
            
            # Layout
            pos = nx.spring_layout(G, seed=42)
            
            # Dibujar nodos
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                   node_size=700, alpha=0.8, ax=ax2)
            
            # Dibujar etiquetas de nodos
            nx.draw_networkx_labels(G, pos, labels={i: f'X{i}' for i in G.nodes()}, 
                                   font_weight='bold', font_size=12, ax=ax2)
            
            # Dibujar aristas
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, 
                                   edge_color='gray', arrows=True, 
                                   arrowsize=15, ax=ax2)
            
            # Añadir etiquetas de aristas
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                           font_size=8, ax=ax2)
            
            ax2.set_title("Grafo de Dinámica Poblacional")
            ax2.axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error en visualize_results: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error al visualizar: {e}", 
                   horizontalalignment='center', verticalalignment='center')
            return fig