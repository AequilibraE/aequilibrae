@echo off
setlocal

pushd %~dp0

python -m pip install -r "docs/requirements-docs.txt" || goto :error

sphinx-apidoc.exe -o docs/source/generated aequilibrae || goto :error

pushd docs

call make htmlhelp
