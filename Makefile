PYTHON ?= python
PYTHON_BIN ?= $(PYTHON)

.PHONY: check validate-dynamics paper data-audit cache surface-train surface-eval underwater-train classical classical-results formation formation-results planner-rendering

check:
	$(PYTHON) -m compileall -q code scripts/check_runtime.py
	@for script in scripts/*.sh; do bash -n "$$script"; done
	$(PYTHON) scripts/check_runtime.py

validate-dynamics:
	$(PYTHON) -m code.official_experiments.validate_dynamics

paper:
	cd paper && tectonic -X compile --keep-intermediates --keep-logs main.tex

data-audit:
	$(PYTHON) -m code.data_preparation.audit_hycom_currents

cache:
	$(PYTHON) -m code.data_preparation.build_uwflow_cache 30 0.25 3

surface-train:
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_surface_vmax12_seed10.sh

surface-eval:
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_surface_distance_recompute.sh

underwater-train:
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_uwflow_7depth_completion.sh

classical:
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_classical_surface_matrix.sh
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_classical_underwater_matrix.sh

classical-results:
	$(PYTHON) -m code.baselines.aggregate_classical_baselines

formation:
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_formation_training.sh
	PYTHON_BIN=$(PYTHON_BIN) bash scripts/run_formation_full_matrix.sh

formation-results:
	$(PYTHON) -m code.formation.aggregate_formation_full_matrix
	$(PYTHON) -m code.formation.make_formation_comparison_figure

planner-rendering:
	$(PYTHON) -m code.analysis.make_traditional_planner_renderings
