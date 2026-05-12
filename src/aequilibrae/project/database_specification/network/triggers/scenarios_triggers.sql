--#
-- Prevent removal of root scenario
CREATE TRIGGER aequilibrae_root_scenario_update BEFORE UPDATE ON scenarios
  WHEN new.scenario_name = 'root'
  BEGIN
       SELECT RAISE(ABORT,'Cannot update root scenario');
  END;

--#
CREATE TRIGGER aequilibrae_root_scenario_delete BEFORE DELETE ON scenarios
  WHEN new.scenario_name = 'root'
  BEGIN
       SELECT RAISE(ABORT,'Cannot delete root scenario');
  END;
