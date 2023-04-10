.. _tables_results:

Results
~~~~~~~

The **results** table exists to hold the metadata for the results stored in the
*results_database.sqlite* in the same folder as the model database.

Although those results could as be stored in the model database, it is possible
that the number of tables in the model file would grow too quickly and would
essentially clutter the *project_database.sqlite*.

This is just a matter of software design and can change in future versions of
the software, however.