.. _matrix_table:

========
Matrices
========

The infrastructure around matrices in AequilibraE is composed of an index table,
**matrices**, in the main project file that lists all matrices associated with
the project, and a folder inside the project main, which contains the actual
matrix files.

The need for having matrices sit outside the model file are two-fold: I/O
performance and to keep the model file with a more manageable size.
Have you ever wondered while every single transportation modeling platform keeps
matrices in separate files? It turns out this is a quite obvious design decision
given current models and available computer hardware.

Finally, AequilibraE is fully compatible with OMX, so you can keep (and generate)
all your matrices in that format.
