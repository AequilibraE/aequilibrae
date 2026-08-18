from aequilibrae.log import logger


def migrate(*, closure):
    """Move transit result metadata without constructing project gateways."""
    project = closure["project"]
    transit = closure["transit"]
    results = closure["results"]
    if transit.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'").fetchone() is None:
        logger.info("Migration finished, no transit results metadata table found")
        return

    payloads = {
        row[0]
        for row in results.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    recorded = {row[0] for row in project.execute("SELECT table_name FROM results").fetchall()}
    for table_name in payloads - recorded:
        project.execute(
            "INSERT INTO results (table_name, procedure, procedure_id, procedure_report) VALUES (?, '', '', 'null')",
            (table_name,),
        )

    project_columns = {row[1] for row in project.execute("PRAGMA table_info(results)").fetchall()}
    transit_columns = [row[1] for row in transit.execute("PRAGMA table_info(results)").fetchall()]
    columns = [column for column in transit_columns if column in project_columns]
    if "table_name" in columns:
        quoted = ",".join(f'"{column}"' for column in columns)
        updates = ",".join(
            f'"{column}"=excluded."{column}"' for column in columns if column != "table_name"
        )
        sql = f"INSERT INTO results ({quoted}) VALUES ({','.join('?' for _ in columns)})"
        if updates:
            sql += f' ON CONFLICT("table_name") DO UPDATE SET {updates}'
        rows = transit.execute(f"SELECT {quoted} FROM results").fetchall()
        project.executemany(sql, rows)

    transit.execute("DROP TABLE results")
