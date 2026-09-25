.PHONY: help setup-pre-commit setup-github check

help: ## Show this help message
	@awk -F':.*?## ' '/^[a-zA-Z0-9_-]+:.*## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup-pre-commit: ## Set up pre-commit hooks
	pre-commit install
	pre-commit autoupdate

setup-github: ## Set up GitHub repository settings
	chmod +x ./scripts/setup_github.sh
	./scripts/setup_github.sh

check: ## Run pre-commit checks
	pre-commit run --all-files --show-diff-on-failure
