--#
CREATE TRIGGER IF NOT EXISTS aequilibrae_results_require_table_on_insert
BEFORE INSERT ON results
WHEN NOT EXISTS (
    SELECT 1 FROM sqlite_master
    WHERE type = 'table' AND name = NEW.table_name
)
BEGIN
    SELECT RAISE(ABORT, 'results table does not exist');
END;
--#
CREATE TRIGGER IF NOT EXISTS aequilibrae_results_require_table_on_update
BEFORE UPDATE ON results
WHEN NOT EXISTS (
    SELECT 1 FROM sqlite_master
    WHERE type = 'table' AND name = NEW.table_name
)
BEGIN
    SELECT RAISE(ABORT, 'results table does not exist');
END;
--#
CREATE TRIGGER IF NOT EXISTS aequilibrae_results_require_absent_old_table_on_update
BEFORE UPDATE OF table_name ON results
WHEN OLD.table_name != NEW.table_name AND EXISTS (
    SELECT 1 FROM sqlite_master
    WHERE type = 'table' AND name = OLD.table_name
)
BEGIN
    SELECT RAISE(ABORT, 'results table still exists');
END;
--#
CREATE TRIGGER IF NOT EXISTS aequilibrae_results_require_table_on_delete
BEFORE DELETE ON results
WHEN EXISTS (
    SELECT 1 FROM sqlite_master
    WHERE type = 'table' AND name = OLD.table_name
)
BEGIN
    SELECT RAISE(ABORT, 'results table still exists');
END;
