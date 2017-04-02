nosetest:
	nosetests --verbosity=2 --with-doctest .

pytest:
	pytest -v --doctest-modules .
