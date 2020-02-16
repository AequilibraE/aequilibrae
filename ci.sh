#!/usr/bin/env bash

set -euo pipefail

main() {
    local command=$1; shift
    case $command in
    setup_dev )
        setup_pipenv
        setup_precommit
        ;;
    setup_docker )
        setup_venv
        setup_pipenv
        ;;
    run_unit_tests | test )
        setup_venv
        run_unit_tests "$@"
        ;;
    doc )
        setup_venv
        generate_docs
        ;;
    lint )
        setup_venv
        run_linting
        ;;
    ** )
        echo "unknown command: $command"
        exit 1
        ;;
    esac

}

run_unit_tests() {
    python3 -m unittest discover -s test -p *_test.py
}

generate_docs() {
    python3 -m pipenv run sphinx
}

run_linting() {
    python3 -m flake8 --format=checkstyle > flake8.xml
}

setup_venv() {
    virtualenv .venv
    source .venv/bin/activate
}

setup_precommit() {
    python3 -m pipenv run pre-commit-install
}

setup_pipenv() {
    python3 --version
    python3 -m pip install pipenv
    python3 -m pipenv install --dev
}

echo "Running $0"


main "$@"
