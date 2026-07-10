# When updating the version, one must also update the docs/source/useful_links/version_history.rst file
project(
  'aequilibrae',
  'cpp',
  'cython',
  default_options: ['cpp_std=c++17', 'cython_language=cpp'],
  version: '2.0.0',
  license: 'LicenseRef-AequilibraE',
  license_files: ['LICENSE.TXT'],
)

py = import('python').find_installation(pure: false)


openmp = dependency('openmp')

# See note in resolve_cysignals.py.
inc_cysignals = run_command(py, 'src/aequilibrae/utils/resolve_cysignals.py', check: true).stdout().strip()
cysignals = declare_dependency(include_directories: inc_cysignals)

deps = [openmp, cysignals]

incdir = include_directories(
  'src/aequilibrae/utils/cpp/'
)

add_project_arguments(
  '-X', 'embedsignature=True',
  '-X', 'emit_code_comments=True',
  language : 'cython'
)

subdir('src/aequilibrae')
