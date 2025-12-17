.. _logging_internals:

Logging System Internals
========================

AequilibraE implements a custom logging infrastructure designed to handle three specific constraints:

1.  **Progress Bar Compatibility:**
    Preventing log messages from breaking the visual structure of ``tqdm`` bars in the terminal.

2.  **Thread Safety (GIL-free):**
    Allowing logging from heavy computational loops in Cython (``nogil``) and C++ without acquiring the GIL.

3.  **QGIS Integration:**
    Routing logs correctly when running inside the QGIS Python environment.

For an example of the logging to the terminal see :ref:`logging_to_terminal`.

Python Logging
--------------

All Python code should utilise the standard Python ``logging`` module.

We should aim to module-level loggers using ``__name__``.

.. code-block:: python

    import logging

    # Define at the top of the module
    logger = logging.getLogger(__name__)

    def complex_function():
        logger.info("Starting process...")
        try:
            # logic
            logger.debug("Detailed step info")
        except Exception:
            logger.error("Something went wrong", exc_info=True)


Logging configuration should be opt-in with easy-to-apply defaults
(:py:func:`aequilibrae.utils.logging_utils.basic_config`).

If a more complicated logging configuration is required the
:py:class:`aequilibrae.utils.logging_utils.AequilibraEStreamHandler` class should be utilised to preserve progress bars
on stream-based handlers.

Cython & C++ Logging (The Bridge object)
---------------------------------

When operating within performance-critical loops (Cython ``nogil`` blocks or pure C++), standard Python logging is
inaccessible because it requires the GIL. AequilibraE solves this using a ``Bridge`` object.

The ``Bridge`` accumulates log messages in a thread-safe C++ queue and consumes them in another Python thread,
forwarding them to the standard Python logging system.

Using the Bridge in Cython
~~~~~~~~~~~~~~~~~~~~~~~~~~

To log from ``nogil`` blocks, you must pass the ``Bridge`` object into your function/method.

1.  **Import the logging utilities:**

    .. code-block:: cython

        from aequilibrae.utils.cython.bridge cimport Bridge, log, f, DEBUG, INFO, WARN, ERROR

2.  **Use the log macro:**

    The syntax is ``log(bridge_instance, level, message_expression)``.

    .. code-block:: cython

        # Within a nogil block
        cdef void heavy_computation(Bridge bridge, int iteration) nogil:
            # String formatting must use the 'f' helper for delayed evaluation
            log(bridge, INFO, f("Starting iteration: ", iteration))

            if iteration < 0:
                log(bridge, ERROR, f("Invalid iteration: ", iteration))


The third argument to ``log`` is an expression. In this example the ``f()`` helper (which wraps C++ stream formatting)
is **only evaluated** if the log level threshold is met. This prevents expensive string construction for ``DEBUG`` logs
when the level is set to ``INFO``.

Using the Bridge in C++
~~~~~~~~~~~~~~~~~~~~~~~

When calling pure C++ code from Cython, pass the underlying pointer to the Bridge struct.

1.  **Include the header:**

    .. code-block:: cpp

        #include "aeq_log.h"
        // This ensures you have access to the generated bridge structure definition and all other required headers

2.  **Pass the pointer:**

    Your C++ function signature should accept a ``struct Bridge *bridge``. This struct is declared in the ``bridge.h``
    header which is generated from ``bridge.pxd`` at compile time. However, it's preferred to include the ``aeq_log.h``
    header as it also includes the other required headers for logging.

3.  **Use the macro:**

    The macro syntax mirrors the Cython implementation.

    .. code-block:: cpp

        #include "aeq_log.h"

        void run_algo(struct Bridge *bridge) {
            int nodes = 100;
            // AEQ_LOG is defined in _aeq_log.h
            // aeq_format_string handles variadic arguments
            AEQ_LOG(bridge, 20, aeq_format_string("Graph has ", nodes, " nodes"));
        }

Log Levels
~~~~~~~~~~

The bridge module also maps standard Python log levels to integers which are accessible without the GIL. They are
available as constants in ``aequilibrae.utils.cython.bridge``:

*   ``DEBUG``
*   ``INFO``
*   ``WARNING``
*   ``ERROR``
*   ``CRITICAL``
