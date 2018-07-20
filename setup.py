import setuptools


def setup_pyints():

    setuptools.setup(
        name="pyints",
        version="0.0.1alpha",
        packages=setuptools.find_packages(exclude=["*test*"]),

        install_requires=[
            "basis_set_exchange", "cclib",
        ],

        # metadata
        author="Eric Berquist",
        maintainer="Eric Berquist",
        # TODO read this and long_description from README.md
        description="",
        license="BSD 3-Clause License",
        url="https://github.com/berquist/pyints",
        project_urls={
            "Documentation": "https://berquist.github.io/pyints_docs/",
        },
    )


if __name__ == "__main__":
    setup_pyints()
