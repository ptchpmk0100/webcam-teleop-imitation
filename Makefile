.PHONY: setup test demo trial diagnose
setup:
	uv sync
	./scripts/fetch_assets.sh

test: setup
	env -u PYTHONPATH uv run pytest

demo: setup
	env -u PYTHONPATH uv run python -m teleop_il.live $(ARGS)

trial: setup
	env -u PYTHONPATH uv run python -m teleop_il.live --trial $(ARGS)

diagnose:
	env -u PYTHONPATH uv run python -m teleop_il.diagnose $(LOG)
