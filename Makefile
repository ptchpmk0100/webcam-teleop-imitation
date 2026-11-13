.PHONY: setup test
setup:
	uv sync
	./scripts/fetch_assets.sh

test: setup
	env -u PYTHONPATH uv run pytest
