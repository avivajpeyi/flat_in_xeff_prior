import codecs
import os
import re

from setuptools import find_packages, setup

NAME = "flat_in_xeff_prior"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join( "src", "flat_in_xeff_prior", "__init__.py")
INSTALL_REQUIRES = ["numpy", "matplotlib"]
EXTRA_REQUIRE = {"test": ["pytest>=3.6"]}
EXTRA_REQUIRE["dev"] = EXTRA_REQUIRE["test"] + [
    "pre-commit",
    "flake8",
    "black<=21.9b0",
    "isort",
]

# END PROJECT SPECIFIC

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


__author__ = "Avi Vajpeyi"
__email__ = "avi.vajpeyi@gmail.com"


setup(
    name=NAME,
    version="1.0.0",
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    url="https://github.com/avivajpeyi/flat_in_xeff_prior",
    license="MIT",
    description="Flat in xeff prior in bilby",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=PACKAGES,
    package_dir={"": "src"},
    keywords=[],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRE,
    classifiers=CLASSIFIERS,
    zip_safe=True,
    entry_points={"console_scripts": []},
)
