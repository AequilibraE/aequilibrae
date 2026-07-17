:orphan:

.. _profiling:

Profiling
=========

Given AequilibraE is performance-critical software for many of its users, tracking down performance degradations and
searching for improvements while developing new features is vital work. This page aims to document our internal
profiling experiences and workflows. These techniques can be used on all platforms and versions, however, the best
results are obtained using Python 3.12 or later, compiled with particular flags, on a recent Linux kernel.

Differences from profiling a standard Python library
----------------------------------------------------

AequilibraE makes heavy use of CPython extension modules (written in Cython and C++) to implement its
performance-critical operations. These execute at a level below what normal Python profilers (and debuggers) can see.  A
pure Python profiler observes the interpreter, either by hooking function calls or by sampling the interpreter's call
stack, so the moment execution enters a compiled extension the profiler sees a single opaque call until control returns
to Python. For AequilibraE this is a problem: almost all of the interesting time (path-finding, traffic assignment,
matrix operations) is spent inside compiled code, often in OpenMP worker threads that a Python profiler does not even
know exist. For this reason, we make use of profilers designed to profile native code.

CProfile, `line_profiler <https://github.com/pyutils/line_profiler>`_, or the new tracing (``profiling.tracing``) and
sampling (``profiling.sampling``, nicknamed "Tachyon") profilers introduced in Python 3.15, are still usable on
AequilibraE, and they will work correctly for identifying Python-level bottlenecks. However, they are unable to see into
the Cython and C++ extensions used by AequilibraE and other libraries such as NumPy, Pandas, and SciPy. In fact, the new
profilers will provide *better* insight into what is happening at the Python level than the methods described below, so
the two approaches are complementary: use a Python profiler to understand the orchestration code, and a native profiler
to understand where the compute time actually goes.

Software
--------

`perf <https://perf.wiki.kernel.org/index.php/Main_Page>`_
    Performance analysis tools for Linux. An extremely powerful tool that can make use of specialised kernel features
    to access hardware performance counters (CPU cycles, cache misses, branch mispredictions, ...), kernel
    tracepoints, and scheduling events. It samples the whole process, including all native threads, with very low
    overhead.

`samply <https://github.com/mstange/samply>`_
    Command-line sampling profiler for macOS, Linux, and Windows. An alternative to perf that uses the Firefox
    Profiler as its UI. On Linux it builds on the same kernel infrastructure as perf, and it can also *import*
    existing ``perf.data`` files, which is how we combine the two tools below.

`Firefox Profiler <https://profiler.firefox.com/>`_
    Web app for performance analysis. A profiler UI originally built to profile Firefox itself, it includes support
    for perf profile files and provides a modern, shareable UI with call tree, flame graph, stack chart, and marker
    timeline views. Profiles can be uploaded and shared as links.

.. note::
   On macOS, Python >= 3.15 is required, as earlier macOS builds were not compiled with frame pointers and native
   stacks cannot be reliably unwound.

Setting up Python and AequilibraE for the best results
------------------------------------------------------

Sampling profilers capture a stack trace hundreds of times per second, and the cheapest and most reliable way to walk
the stack is by following frame pointers. The best profiles are therefore obtained by using a version of Python compiled
with frame and leaf-frame pointers enabled (``-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer``).  Without them,
the profiler must fall back to DWARF-based unwinding, which is slower, produces much larger recordings, and frequently
results in broken or truncated stacks. See the `Python documentation
<https://docs.python.org/3/howto/perf_profiling.html>`_ for how to check whether your interpreter was built with these
flags. From Python 3.15 onwards, frame pointers are enabled by default on all platforms (`PEP 831
<https://peps.python.org/pep-0831/>`_).

AequilibraE compiles its extensions with frame pointers enabled by default, to align with PEP 831. However, AequilibraE
should be installed using the following command (or similar) to make debug information and source code available to
profilers::

    uv pip install meson-python ninja cython cysignals
    uv pip install --no-build-isolation --editable ".[dev]" \
        -Cbuild-dir=build \
        -Ceditable-verbose=true \
        -Csetup-args="-Ddebug=true" \
        -Csetup-args="-Doptimization=3"

The important pieces here are:

* ``-Ddebug=true`` emits debug information, allowing profilers to map samples back to source lines, including the Cython
  sources the C/C++ was generated from.
* ``-Doptimization=3`` keeps full compiler optimisation despite the debug build, so the profile remains representative
  of a release build. Profiling an unoptimised build will mislead you.
* ``--editable`` with ``--no-build-isolation`` keeps the build tree (``build/``) and sources on disk, so the profiler UI
  can display the annotated source code.

This setup is applicable to Windows, macOS, and Linux, however it is best executed on Linux.

Recording a profile with samply
-------------------------------

On Linux::

    PYTHONPERFSUPPORT=1 samply record --reuse-threads -- <python command>

``PYTHONPERFSUPPORT=1`` asks the Python interpreter (3.12+) to emit small stack "trampolines" and a perf map file that
teach perf the relationship between an interpreter stack frame and the Python function being executed. With it
enabled, the recorded stacks are *mixed-mode*: you see the Python call stack and the native call stack interleaved,
exactly as they occur. Only Python on Linux supports this.

On Windows and macOS::

    samply record --reuse-threads -- <python command>

When the command finishes, samply opens the Firefox Profiler in your browser, displaying the execution of the Python
interpreter, the CPython extensions, and, if ``PYTHONPERFSUPPORT`` was enabled, which Python function was executing at
the time of each sample.

Recording with perf directly (Linux only)
-----------------------------------------

For more events and details, perf can do the recording and samply can be used purely as the viewer. This gives access to
kernel tracepoints, which show up as *markers* in the Firefox Profiler timeline, alongside the CPU sampling data.

.. code-block:: bash

    #!/usr/bin/env bash
    # profile.sh: perf record with markers/CPU tracks, viewed in Firefox Profiler via samply.
    # Usage: ./profile.sh python my_script.py [args...]
    set -euo pipefail

    OUT=perf.data
    FREQ=999

    # Tracepoints need perf_event_paranoid <= 0 (or root)
    EXTRA_EVENTS=(
        -e sched:sched_switch                      # scheduling markers
        -e syscalls:sys_enter_openat               # file-open markers
        -e syscalls:sys_enter_mmap                 # allocation-ish markers
        -e page-faults
    )

    PYTHONPERFSUPPORT=1 perf record \
        -o "$OUT" \
        -F "$FREQ" \
        --call-graph fp \
        -k mono \
        --switch-events \
        --sample-cpu \
        -e cycles \
        "${EXTRA_EVENTS[@]}" \
        -- "$@"

    samply import --reuse-threads "$OUT"

A note on ``--reuse-threads``: when ``AEQ_CPUS`` and ``AEQ_ELEMENTWISE_CPUS`` are set to different values, libgomp
(GCC's OpenMP runtime) destroys and recreates the differing number of worker threads every time consecutive parallel
regions request different team sizes. Each iteration then spawns fresh OS threads, and without ``--reuse-threads`` the
profile fills up with an enormous number of short-lived thread tracks. The actual runtime overhead of this is small, but
there is no clean way to avoid it in the profile without adding hacks, so instructing samply to merge threads with the
same name into a single track is good enough.

Interpreting the results
------------------------

A few things we have learned to look for in AequilibraE profiles:

* Start with the *inverted* call tree to find the hottest leaf functions, then un-invert to understand how execution
  reaches them.
* Significant time in libgomp barrier or spin functions (e.g. ``gomp_barrier_wait``, ``do_spin``) indicates load
  imbalance between OpenMP threads, or parallel regions that are entered far too frequently relative to the work they
  do.
* Use the "Script" vs "Native" filters to separate the Python interpreter from the compiled extensions and interpreter.
