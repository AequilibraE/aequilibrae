class DBProxy:
    """Proxy to allow direct access and usage as a context manager."""

    def __init__(self, context_manager):
        self._context_manager = context_manager
        self._conn = None  # Stores the connection when used directly

    def __enter__(self):
        self._conn = self._context_manager.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc_value, traceback):
        if self._conn:
            self._context_manager.__exit__(exc_type, exc_value, traceback)
            self._conn = None

    def __getattr__(self, name):
        """Forwards calls to the actual connection when used without 'with'."""
        if self._conn is None:
            with self._context_manager as conn:
                return getattr(conn, name)  # Returns the actual method of the connection

        return getattr(self._conn, name)  # Returns the actual method if inside 'with'
