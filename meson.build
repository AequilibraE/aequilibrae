project(
  'aequilibrae',
  'cpp',
  'cython',
  default_options: ['cpp_std=c++17', 'cython_language=cpp']
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
