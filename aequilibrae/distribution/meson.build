thisdir = 'aequilibrae/distribution'

pyfiles = [
  '__init__.py',
  'gravity_application.py',
  'gravity_calibration.py',
  'ipf.py',
  'ipf_core.py',
  'synthetic_gravity_model.py',
]

pkgs = [
  'cython',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
