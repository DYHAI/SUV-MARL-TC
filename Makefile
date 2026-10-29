PYTHON ?= python
PYTHON_BIN ?= $(PYTHON)

.PHONY: check-data cache-30 paper surface-eval underwater-complete

check-data:
	$(PYTHON) audit_hycom_currents.py

cache-30:
	UWFLOW_CACHE_DIR=$(UWFLOW_CACHE_DIR) $(PYTHON) build_uwflow_cache.py 30 0.25 3

paper:
	cd Latex_Yihui_MARL && tectonic -X compile --keep-intermediates --keep-logs main.tex

surface-eval:
	PYTHON_BIN=$(PYTHON_BIN) bash run_surface_distance_recompute.sh

underwater-complete:
	PYTHON_BIN=$(PYTHON_BIN) bash run_uwflow_7depth_completion.sh
