test: pytest-cov

pylint:
	find . -name "*.py" -exec pylint '{}' +

pytest:
	pytest -v --doctest-modules test

pytest-cov:
	pytest -v --doctest-modules --cov=pyints test
