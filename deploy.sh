#!/bin/bash

#anaconda login
#conda config --set anaconda_upload yes

#--- Reset depository on PyPI - Change version in setup.py
#rm -f dist/*
#python3 setup.py sdist bdist_wheel
#twine upload dist/*

#--- Reset conda distrubution - Change version and add requirements in meta.yaml / Change version for conda conver
#rm -rf conda_pyfloc
#conda skeleton pypi pyfloc --output-dir conda_pyfloc
#cp bld.bat conda_pyfloc/pyfloc
##cp meta.yaml conda_pyfloc/pyfloc # don't copy this or the sha256 code would be different 
#conda-build conda_pyfloc/pyfloc
#conda convert -f --platform all /Users/simone/anaconda3/conda-bld/osx-64/pyfloc-0.0.3-py36_0.tar.bz2 -o conda_pyfloc/


anaconda upload conda_pyfloc/*/*bz2
##anaconda upload /Users/simone/anaconda3/conda-bld/osx-64/pyfloc-0.0.2-py36_0.tar.bz2 # useless if --set anaconda_upload yes

exit
