#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
pip install -q -U numpy
pip install -q "tensorflow==1.14"
pip install -q "tensorflow_probability==0.7"
