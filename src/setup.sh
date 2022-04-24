#!/bin/bash

export FLASK_APP=application.py

conda env create -f config/demsvsreps.yml

conda activate demsvsreps