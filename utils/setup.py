# python setup.py develop
from setuptools import setup


CLASSIFIERS = """\
License :: OSI Approved
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
MINOR = 2
MICRO = 0
ISRELEASED = True
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    packages=["tf_utils"],
    author=AUTHOR,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
