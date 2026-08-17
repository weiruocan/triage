.PHONY: install dev clean test example

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test:
	python3 -c "from triage import triage_agent; print('Import OK')"

example:
	python3 examples/quickstart.py

example-e2e:
	python3 examples/end_to_end.py
