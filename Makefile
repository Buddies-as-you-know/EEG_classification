.PHONY: isort-check
isort-check:
	poetry run isort --check-only src tests

.PHONY: isort
isort:
	poetry run isort src tests


.PHONY: mdformat
mdformat:
	poetry run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	poetry run mdformat --check *.md

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: ruff-format
ruff-format:
	poetry run ruff format src tests 

.PHONY: ruff-lint
ruff-lint:
	poetry run ruff check --select I --fix src tests 


.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: lint
lint:
	$(MAKE) isort-check
	$(MAKE) mdformat-check
	$(MAKE) mypy
	$(MAKE) ruff lint

.PHONY: format
format:
	$(MAKE) ruff-format
	$(MAKE) isort
	$(MAKE) mdformat

.PHONY: test-all
test-all:
	$(MAKE) lint
	$(MAKE) test