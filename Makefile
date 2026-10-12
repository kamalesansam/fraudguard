.PHONY: help install lint test security pipeline model drift-demo image scan serve up down clean
.DEFAULT_GOAL := help
export MLFLOW_DISABLE_AGENT_HINT := 1

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the locked environment (all extras + dev tools)
	uv sync --all-extras

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

test: ## Unit + integration tests with coverage
	uv run pytest --cov --cov-report=term-missing

security: ## SAST (Bandit, Semgrep) + SCA (pip-audit) locally
	uv run bandit -r src -ll
	uvx semgrep@1.180.0 scan --metrics=off --error --config p/python --config .semgrep
	uv export --frozen --no-dev --extra serve --no-emit-project --format requirements-txt -o .req-serve.txt
	uvx pip-audit==2.10.1 -r .req-serve.txt --disable-pip --require-hashes --strict
	@rm -f .req-serve.txt

pipeline: ## Reproduce the DVC pipeline (validate -> train -> gate -> package)
	uv run dvc repro

model: pipeline ## Alias: produce artifacts/model/

drift-demo: ## Simulate production batches and run the drift monitor on each
	uv run fraudguard generate --out data/production/stable.csv --n-rows 10000 --seed 7
	uv run fraudguard generate --out data/production/drifted.csv --n-rows 10000 --seed 8 --drift 0.8
	uv run fraudguard generate --out data/production/concept.csv --n-rows 10000 --seed 9 --concept-drift
	-uv run fraudguard drift data/production/stable.csv --report reports/drift_stable.json
	-uv run fraudguard drift data/production/drifted.csv --report reports/drift_drifted.json --html reports/drift_drifted.html
	-uv run fraudguard drift data/production/concept.csv --report reports/drift_concept.json

image: model ## Build the inference image
	docker build -t fraudguard:local .

scan: image ## Trivy scan of the local image
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.70.0 image --severity HIGH,CRITICAL --ignore-unfixed fraudguard:local

serve: ## Run the API locally (demo key: local-dev-key)
	FRAUDGUARD_API_KEY_HASHES=ed5a18fb8f807f996d649e379d3f35f39c543a91bdbf88c492f2ebd10d4df86c \
	FRAUDGUARD_ENABLE_DOCS=true uv run fraudguard-serve

up: model ## API + Prometheus + Grafana via docker compose
	GRAFANA_ADMIN_PASSWORD=$${GRAFANA_ADMIN_PASSWORD:-$$(openssl rand -hex 12)} docker compose up --build -d

down: ## Stop the compose stack
	docker compose down

clean: ## Remove generated data, models and caches
	rm -rf data artifacts mlflow.db mlruns .pytest_cache .ruff_cache .coverage coverage.xml
