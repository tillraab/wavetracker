import glob
import json
import os

from setuptools import setup

with open(os.path.join("info.json")) as infofile:
    infodict = json.load(infofile)

NAME = infodict["NAME"]
VERSION = infodict["VERSION"]
AUTHOR = infodict["AUTHOR"]
CONTACT = infodict["CONTACT"]
HOMEPAGE = infodict["HOMEPAGE"]
CLASSIFIERS = infodict["CLASSIFIERS"]
DESCRIPTION = infodict["DESCRIPTION"]

README = "README.md"
with open(README) as f:
    description_text = f.read()

packages = [
    "wavetracker",
]

install_req = ["PyQt5",
               'ruamel.yaml',
               'tqdm']

data_files = [("icons", glob.glob(os.path.join("wavetracker/gui_sym", "*.png"))),
              (".", ["LICENSE"])
              ]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=CONTACT,
    url=HOMEPAGE,
    packages=packages,
    install_requires=install_req,
    include_package_data=True,
    data_files=data_files,
    long_description=description_text,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    license="BSD",
    entry_points={
        # "gui_scripts": ["EODsorter = wavetracker.EODsorter:main",
        #                 "trackingGUI = wavetracker.trackingGUI:main"],
        "console_scripts": ["EODsorter = wavetracker.EODsorter:main",
                            "trackingGUI = wavetracker.trackingGUI:main",
                            'wavetracker = wavetracker.wavetracker:main',
                            'dataviewer = wavetracker.dataviewer:main_UI']
        }
)
