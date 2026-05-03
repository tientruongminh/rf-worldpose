.PHONY: up down fmt test
up:
	docker compose -f infra/docker-compose/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose/docker-compose.yml down

fmt:
	@echo "TODO: run cargo fmt, ruff, prettier"

test:
	@echo "TODO: run unit/integration tests"
