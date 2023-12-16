--#
-- Prevent removal of default period
CREATE TRIGGER default_period_update BEFORE UPDATE ON periods
  WHEN new.period_id = 1
  BEGIN
       SELECT RAISE(ABORT,'Cannot update default period');
  END;

--#
CREATE TRIGGER default_period_delete BEFORE DELETE ON periods
  WHEN new.period_id = 1
  BEGIN
       SELECT RAISE(ABORT,'Cannot delete default period');
  END;
