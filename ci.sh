#!/usr/bin/env bash

set -eu

main() {
    # stuff
    setup_pip
    install_requirements
    build_libs

    if [[ "$1" == "test" ]]; then
        run_tests
    else if [[ "$1" == "publish" ]]; then
        package_for_publication
    else
        echo "Unknown option ${1}"
        exit(-1)
    fi

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
    pytest \
        --cov aequilibrae \
        --cov-report term-missing \
        --junitxml=junit_report.xml
}

package_for_publication() {
    echo "I'm a publishing it now!!!"
}

main "$@"
