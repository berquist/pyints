nosetest:
	nosetests2 --verbosity=2 --with-doctest test

pytest:
	pytest2 -v --doctest-modules test
