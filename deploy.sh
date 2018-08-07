#!/bin/bash

#anaconda login
#rm -f dist/*

#python3 setup.py sdist bdist_wheel
#twine upload dist/*

#--- Reset conda distrubution
#rm -rf conda_pyfloc
#conda skeleton pypi pyfloc --output-dir conda_pyfloc
#cp bld.bat conda_pyfloc/pyfloc
#cp meta.yaml conda_pyfloc/pyfloc

conda-build conda_pyfloc/pyfloc 
#conda convert -f --platform all /Users/simone/anaconda3/conda-bld/osx-64/pyfloc-0.0.2-py36_0.tar.bz2 -o ./


#anaconda upload /Users/simone/anaconda3/conda-bld/osx-64/pyfloc-0.0.2-py36_0.tar.bz2
#anaconda upload conda_pyfloc/*/*bz2

exit
