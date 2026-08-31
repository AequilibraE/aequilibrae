DROP TRIGGER IF EXISTS aequilibrae_default_period_delete;
DROP TRIGGER IF EXISTS aequilibrae_root_scenario_delete;

CREATE TRIGGER aequilibrae_default_period_delete BEFORE DELETE ON periods
  WHEN old.period_id = 1
  BEGIN
       SELECT RAISE(ABORT,'Cannot delete default period');
  END;

CREATE TRIGGER aequilibrae_root_scenario_delete BEFORE DELETE ON scenarios
  WHEN old.scenario_name = 'root'
  BEGIN
       SELECT RAISE(ABORT,'Cannot delete root scenario');
  END;

