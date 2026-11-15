.PHONY: setup test demo trial diagnose parity policy-parity policy-v2-parity
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

parity: setup
	env -u PYTHONPATH uv run python -m teleop_il.cpp_parity

policy-parity: setup
	env -u PYTHONPATH uv run python -m teleop_il.cpp_policy_parity $(RUN)

policy-v2-parity: setup
	env -u PYTHONPATH uv run python -m teleop_il.cpp_policy_v2_parity $(RUN)
