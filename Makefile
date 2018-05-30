test: pytest-cov

nosetest:
	nosetests --verbosity=2 --with-doctest test

pytest:
	pytest -v --doctest-modules test

pytest-cov:
	pytest -v --doctest-modules --cov=pyints test
