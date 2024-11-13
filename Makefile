.PHONY: clean test wheel clean-build

PYTHON ?=python3

test: venv
	venv/bin/pytest

venv: requirements.txt
	$(PYTHON) -m venv venv --clear --prompt arc-dsl
	venv/bin/pip install --upgrade pip
	venv/bin/pip install wheel
	venv/bin/pip install -r requirements.txt --verbose
	venv/bin/pip install -e .

wheel: venv clean-build
	venv/bin/pip install --upgrade build
	venv/bin/python -m build --wheel

clean-build:
	rm -rf build dist *.egg-info

clean: clean-build
	rm -rf venv
