project('aequilibrae', 'cpp', 'cython',
    version: '1.6.0',
    default_options: ['cpp_std=c++17']
)

py = import('python').find_installation(pure: false)

# C++ arguments
cpp_args = []
if host_machine.system() == 'windows'
    cpp_args += ['/openmp']
else
    cpp_args += ['-fopenmp']
    if host_machine.system() == 'darwin'
        cpp_args += ['-Wno-unreachable-code']
    endif
endif

# ASAN options
link_args = []
if get_option('aeq_asan')
    if host_machine.system() == 'windows'
        cpp_args += ['/fsanitize=address']
        link_args += ['/fsanitize=address']
    else
        cpp_args += ['-fsanitize=address', '-fsanitize=undefined']
        link_args += ['-fsanitize=address', '-fsanitize=undefined']
    endif
endif

if host_machine.system() == 'windows'
    link_args += ['/openmp']
else
    link_args += ['-fopenmp']
endif

subdir('aequilibrae')
