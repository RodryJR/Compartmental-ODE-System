# Makefile simple para instalar dependencias y ejecutar scripts de grafos compartimentales

# Variables
PYTHON = python3
PIP = pip3

# Lista de dependencias
DEPS = numpy networkx matplotlib scipy sympy

# Objetivo por defecto
.PHONY: all
all: setup

# Instalar dependencias
.PHONY: setup
setup:
	$(PIP) install $(DEPS)
	@echo "Todas las dependencias han sido instaladas correctamente."

# Ejecutar un script específico
.PHONY: run
run:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Error: Por favor especifica un script para ejecutar usando 'make run SCRIPT=tu_script.py'"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPT)


# Ayuda
.PHONY: help
help:
	@echo "Uso del Makefile:"
	@echo "  make setup           : Instala todas las dependencias"
	@echo "  make run SCRIPT=x.py : Ejecuta el script especificado"
	@echo "  make help            : Muestra esta ayuda"