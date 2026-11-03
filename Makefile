PYTHON ?= python
PYTHON_BIN ?= $(PYTHON)

.PHONY: paper data-audit cache surface-train surface-eval underwater-train classical formation

paper:
	cd Latex_Yihui_MARL && tectonic -X compile --keep-intermediates --keep-logs main.tex

data-audit:
	$(PYTHON) audit_hycom_currents.py

cache:
	$(PYTHON) build_uwflow_cache.py 30 0.25 3

surface-train:
	PYTHON_BIN=$(PYTHON_BIN) bash run_surface_vmax12_seed10.sh

surface-eval:
	PYTHON_BIN=$(PYTHON_BIN) bash run_surface_distance_recompute.sh

underwater-train:
	PYTHON_BIN=$(PYTHON_BIN) bash run_uwflow_7depth_completion.sh

classical:
	PYTHON_BIN=$(PYTHON_BIN) bash run_classical_surface_matrix.sh
	PYTHON_BIN=$(PYTHON_BIN) bash run_classical_underwater_matrix.sh

formation:
	bash run_formation_training.sh
	bash run_formation_full_matrix.sh
