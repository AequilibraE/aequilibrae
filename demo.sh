set +x

export PYTHONPATH="$(spin run 'echo $PYTHONPATH')"
rm -r $1
python -m aequilibrae.project.tools.cli -p $1 create
python -m aequilibrae.project.tools.cli -p $1 copy-from-old $2
