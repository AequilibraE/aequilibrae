#!/usr/bin/env bash

set -eu

main() {
    # stuff
    setup_pip
    install_requirements
    build_libs

    local command=$1
    if [[ "${command}" == "test" ]]; then
        run_tests
    elif [[ "${command}" == "publish" ]]; then
        local os=$2
        package_for_publication ${os}
    else
        echo "Unknown option ${command}"
        exit -1
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
    local os=$1
    echo "I'm a publishing it now!!!"
    echo "Getting it ready for OS [${os}]"
}

main "$@"
