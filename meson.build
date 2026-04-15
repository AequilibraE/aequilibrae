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
deps = [openmp]

add_project_arguments(
  '-X', 'embedsignature=True',
  '-X', 'emit_code_comments=True',
  language : 'cython'
)

subdir('src/aequilibrae')
