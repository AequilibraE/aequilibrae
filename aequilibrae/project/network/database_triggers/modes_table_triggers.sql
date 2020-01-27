-- Guarantees that the mode records have a single letter for mode_id
CREATE TRIGGER mode_single_letter_update BEFORE UPDATE OF mode_id ON "modes"
WHEN
length(new.mode_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to be a single letter');
END;

#
CREATE TRIGGER mode_single_letter_insert BEFORE INSERT ON "modes"
WHEN
length(new.mode_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to be a single letter');
END;

#

-- Prevents a mode record to be changed when it is in use for any link
CREATE TRIGGER mode_keep_if_in_use_updating BEFORE UPDATE OF mode_id ON "modes"
WHEN
(Select count(*) from links where instr(modes, old.mode_id) > 0)>0

BEGIN
    SELECT RAISE(ABORT, 'Mode in use on your network. Cannot change it');
END;

#
-- Prevents a mode record to be removed when it is in use for any link
CREATE TRIGGER mode_keep_if_in_use_deleting BEFORE DELETE ON "modes"
WHEN
(Select count(*) from links where instr(modes, old.mode_id) > 0)>0
BEGIN
    SELECT RAISE(ABORT, 'Mode in use on your network. Cannot change it');
END;

#
-- Ensures an ALTERED link does not reference a non existing mode
CREATE TRIGGER modes_on_links_update BEFORE UPDATE OF 'modes' ON "links"
WHEN
(select count(*) from modes where instr(new.modes, mode_id)>0)<length(new.modes)
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to exist in the modes table in order to be used');
END;

#
-- Ensures an added link does not reference a non existing mode
CREATE TRIGGER modes_on_links_insert BEFORE INSERT ON "links"
WHEN
(select count(*) from modes where instr(new.modes, mode_id)>0)<length(new.modes)
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to exist in the modes table in order to be used');
END;

#
-- Ensures an ALTERED link has at least one mode added to it
CREATE TRIGGER modes_length_on_links_update BEFORE UPDATE OF 'modes' ON "links"
WHEN
length(new.modes)<1
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to exist in the modes table in order to be used');
END;

#
-- Ensures an added link has at least one mode added to it
CREATE TRIGGER modes_length_on_links_insert BEFORE INSERT ON "links"
WHEN
length(new.modes)<1
BEGIN
    SELECT RAISE(ABORT, 'Mode codes need to exist in the modes table in order to be used');
END;