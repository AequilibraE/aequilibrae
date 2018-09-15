#!/usr/bin/env bash

set -eu

main() {
    # stuff
    setup_pip
    install_requirements
    build_libs
    run_tests
}

setup_pip() {
    python -m pip install --upgrade pip
}

install_requirements() {
    pip install -r requirements.txt
    pip install numpy --upgrade
    pip install codecov
    pip install pytest pytest-cov coveralls pycodestyle
}

build_libs() {
    cd aequilibrae/paths
    python setup_Assignment.py build_ext --inplace
    cd ../../
}

run_tests() {
    pytest --cov aequilibrae --cov-report term-missing
}

main "$@"
