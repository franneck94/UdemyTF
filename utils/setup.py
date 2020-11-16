# python setup.py develop
from setuptools import setup


CLASSIFIERS = """\
License :: OSI Approved
Programming Language :: Python :: 3.7 :: or higher
Topic :: Software Development
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

DISTNAME = "tf_utils"
AUTHOR = "Jan Schaffranek"
DESCRIPTION = "Helper functions for the udemy course."

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

PYTHON_MIN_VERSION = "3.7"
PYTHON_MAX_VERSION = "3.8"

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    packages=["tf_utils"],
    python_requires=">={},<={}".format(PYTHON_MIN_VERSION, PYTHON_MAX_VERSION),
    author=AUTHOR,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS]
)


def setup_package():
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
