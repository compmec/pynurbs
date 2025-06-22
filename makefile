
test:
	pytest --cov=src/pynurbs --cov-report=xml tests
	python3-coverage report -m --fail-under 90
	python3-coverage html

format:
	isort src
	isort tests
	black src
	black tests
	flake8 src
	pylint src

docs:
	sphinx-autobuild docs/ docs/_build/html

html:
	brave htmlcov/index.html