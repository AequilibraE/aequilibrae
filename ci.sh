#!/usr/bin/env bash

set -euo pipefail

main() {
    local command=$1; shift
    case $command in
    setup_dev )
        setup_pyenv
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
    python -m unittest discover -s test -p *_test.py
}

generate_docs() {
    sphinx-apidoc -fT -o docs/source/generated zenith_model_run
    ( cd docs && make html )
}

run_linting() {
    python -m flake8 --format=checkstyle > flake8.xml
}

setup_venv() {
    virtualenv .venv
    source .venv/bin/activate
}

setup_precommit() {
    python -m pipenv run pre-commit-install
}

setup_pipenv() {
    python --version
    python -m pip install pipenv
    python -m pipenv install --dev
}

setup_pyenv() {
    which pyenv || curl https://pyenv.run | bash
    (pyenv versions | grep 3.6) || pyenv install 3.6
}

echo "Running $0"


main "$@"
