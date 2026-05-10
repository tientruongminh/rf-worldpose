COMPOSE = docker compose -f infra/docker-compose/docker-compose.yml --env-file .env

.PHONY: up down logs ps fmt test

up:
	./scripts/dev_up.sh

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

fmt:
	@echo "TODO: run cargo fmt, ruff, prettier"

test:
	@echo "TODO: run unit/integration tests"
