import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path


COPY_HEADER = re.compile(r'^COPY (?:"?public"?\.)?"?([a-z_][a-z_0-9]*)"? \((.*)\) FROM stdin;$')
COPY_ESCAPE = re.compile(r"\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)")


def decode_copy(value):
    if value == r"\N":
        return None
    if "\\" not in value:
        return value

    decoded = bytearray()
    cursor = 0
    for match in COPY_ESCAPE.finditer(value):
        decoded.extend(value[cursor:match.start()].encode("utf-8"))
        escaped = match.group(1)
        if escaped[0] in "01234567":
            decoded.append(int(escaped, 8))
        elif escaped.startswith("x") and len(escaped) > 1:
            decoded.append(int(escaped[1:], 16))
        else:
            decoded.extend({"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}.get(escaped, escaped).encode("utf-8"))
        cursor = match.end()
    decoded.extend(value[cursor:].encode("utf-8"))
    return decoded.decode("utf-8")


def copy_rows(lines, table, required):
    columns = None
    finished = False
    count = 0
    for line in lines:
        line = line.rstrip("\n")
        if columns is None:
            match = COPY_HEADER.match(line)
            if match:
                if match.group(1) != table:
                    raise ValueError("Unexpected COPY table: " + match.group(1))
                names = [name.strip().strip('"') for name in match.group(2).split(",")]
                missing = set(required) - set(names)
                if missing:
                    raise ValueError(table + " is missing columns: " + str(sorted(missing)))
                columns = {name: index for index, name in enumerate(names)}
            elif line.startswith("COPY "):
                raise ValueError("Unsupported COPY header: " + line)
            continue
        if finished:
            if line.startswith("COPY "):
                raise ValueError("Multiple COPY blocks for " + table)
            continue
        if line == r"\.":
            finished = True
            continue
        values = line.split("\t")
        if len(values) != len(columns):
            raise ValueError("Unexpected field count in " + table)
        count += 1
        if count % 1000000 == 0:
            print(json.dumps({"table": table, "rows_scanned": count}), flush=True)
        yield columns, values
    if columns is None or not finished:
        raise ValueError("Missing or incomplete COPY data for " + table)
    print(json.dumps({"table": table, "rows_scanned": count, "complete": True}), flush=True)


def field(columns, values, name):
    return decode_copy(values[columns[name]])


def extract_dump(archive_path, source_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / "chembl_37_postgresql.dmp"
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name != destination.name:
                continue
            if destination.exists() and destination.stat().st_size == member.size:
                with destination.open("rb") as stream:
                    if stream.read(5) == b"PGDMP":
                        print("Reusing extracted dump:", destination, flush=True)
                        return destination
            print("Extracting dump bytes:", member.size, flush=True)
            temporary = destination.with_suffix(".partial")
            try:
                with archive.extractfile(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise ValueError("Incomplete dump extraction")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return destination
    raise ValueError("ChEMBL 37 PostgreSQL dump not found in archive")


def table_rows(image, dump, table, required):
    environment = {**os.environ, "GOMAXPROCS": "2", "LC_ALL": "C", "LC_CTYPE": "C", "LANG": "C",
                   "APPTAINERENV_LC_ALL": "C", "APPTAINERENV_LC_CTYPE": "C", "APPTAINERENV_LANG": "C"}
    command = ["apptainer", "exec", "--bind", str(dump.parent) + ":/chembl_source:ro", str(image),
               "pg_restore", "--data-only", "--no-owner", "--no-privileges", "--table=" + table,
               "--file=-", "/chembl_source/" + dump.name]
    print("Reading table:", table, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=environment)
    try:
        yield from copy_rows(process.stdout, table, required)
        returncode = process.wait()
        if returncode:
            raise RuntimeError("pg_restore failed for " + table + ": " + str(returncode))
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def insert_records(connection, statement, records):
    batch = []
    count = 0
    for record in records:
        batch.append(record)
        if len(batch) >= 10000:
            connection.executemany(statement, batch)
            connection.commit()
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(statement, batch)
        connection.commit()
        count += len(batch)
    print("Retained rows:", count, flush=True)
    return count


def populate(connection, reader):
    connection.executescript("""
        CREATE TABLE targets (tid INTEGER PRIMARY KEY, target_id TEXT, target_name TEXT);
        CREATE TABLE assays (assay_id INTEGER PRIMARY KEY, tid INTEGER, confidence_score INTEGER, assay_type TEXT);
        CREATE TABLE activities (activity_id INTEGER PRIMARY KEY, molregno INTEGER, assay_id INTEGER,
            tid INTEGER, standard_type TEXT, pchembl_value REAL, standard_value TEXT, standard_units TEXT);
        CREATE TABLE molecules (molregno INTEGER PRIMARY KEY, chembl_id TEXT, compound_name TEXT,
            canonical_smiles TEXT, standard_inchi_key TEXT);
        CREATE TABLE target_genes (tid INTEGER, gene_symbol TEXT, PRIMARY KEY (tid, gene_symbol));
        CREATE TABLE target_accessions (tid INTEGER, accession TEXT, PRIMARY KEY (tid, accession));
    """)

    def targets():
        for columns, values in reader("target_dictionary", ["tid", "chembl_id", "pref_name", "organism", "target_type"]):
            if field(columns, values, "organism") == "Homo sapiens" and field(columns, values, "target_type") == "SINGLE PROTEIN":
                yield int(field(columns, values, "tid")), field(columns, values, "chembl_id"), field(columns, values, "pref_name")

    insert_records(connection, "INSERT INTO targets VALUES (?, ?, ?)", targets())
    target_ids = {record[0] for record in connection.execute("SELECT tid FROM targets")}
    if not target_ids:
        raise ValueError("No human single-protein targets found")

    def assays():
        for columns, values in reader("assays", ["assay_id", "tid", "confidence_score", "assay_type"]):
            target = field(columns, values, "tid")
            confidence = field(columns, values, "confidence_score")
            if target is not None and confidence is not None and int(target) in target_ids and int(confidence) >= 9:
                yield int(field(columns, values, "assay_id")), int(target), int(confidence), field(columns, values, "assay_type")

    insert_records(connection, "INSERT INTO assays VALUES (?, ?, ?, ?)", assays())
    assay_targets = dict(connection.execute("SELECT assay_id, tid FROM assays"))

    def activities():
        required = ["activity_id", "molregno", "assay_id", "standard_type", "standard_relation", "pchembl_value",
                    "standard_value", "standard_units", "data_validity_comment", "potential_duplicate"]
        for columns, values in reader("activities", required):
            assay_id = int(field(columns, values, "assay_id"))
            if assay_id not in assay_targets:
                continue
            endpoint = field(columns, values, "standard_type")
            if endpoint not in {"IC50", "Ki", "Kd", "EC50"} or field(columns, values, "standard_relation") != "=":
                continue
            potency = field(columns, values, "pchembl_value")
            if potency is None or field(columns, values, "data_validity_comment") not in (None, ""):
                continue
            if field(columns, values, "potential_duplicate") == "1":
                continue
            potency = float(potency)
            if not math.isfinite(potency):
                continue
            yield (int(field(columns, values, "activity_id")), int(field(columns, values, "molregno")), assay_id,
                   assay_targets[assay_id], endpoint, potency, field(columns, values, "standard_value"),
                   field(columns, values, "standard_units"))

    retained_activities = insert_records(connection, "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?, ?)", activities())
    if not retained_activities:
        raise ValueError("No activities survived filtering")
    connection.execute("CREATE INDEX activities_molecule ON activities(molregno)")
    connection.execute("CREATE INDEX activities_target ON activities(tid)")
    connection.commit()
    wanted = {record[0] for record in connection.execute("SELECT DISTINCT molregno FROM activities")}

    def molecules():
        for columns, values in reader("molecule_dictionary", ["molregno", "chembl_id", "pref_name"]):
            molecule = int(field(columns, values, "molregno"))
            if molecule in wanted:
                yield molecule, field(columns, values, "chembl_id"), field(columns, values, "pref_name")

    insert_records(connection, "INSERT INTO molecules(molregno, chembl_id, compound_name) VALUES (?, ?, ?)", molecules())

    def structures():
        for columns, values in reader("compound_structures", ["molregno", "canonical_smiles", "standard_inchi_key"]):
            molecule = int(field(columns, values, "molregno"))
            if molecule in wanted:
                yield field(columns, values, "canonical_smiles"), field(columns, values, "standard_inchi_key"), molecule

    insert_records(connection, "UPDATE molecules SET canonical_smiles=?, standard_inchi_key=? WHERE molregno=?", structures())
    connection.execute("DELETE FROM activities WHERE molregno NOT IN (SELECT molregno FROM molecules WHERE canonical_smiles IS NOT NULL AND canonical_smiles != '')")
    connection.execute("DELETE FROM molecules WHERE NOT EXISTS (SELECT 1 FROM activities WHERE activities.molregno=molecules.molregno)")
    connection.commit()

    components = {}
    for columns, values in reader("target_components", ["tid", "component_id"]):
        target = int(field(columns, values, "tid"))
        if target in target_ids:
            component = int(field(columns, values, "component_id"))
            components.setdefault(component, set()).add(target)

    def accessions():
        for columns, values in reader("component_sequences", ["component_id", "accession"]):
            component = int(field(columns, values, "component_id"))
            accession = field(columns, values, "accession")
            if component in components and accession:
                for target in components[component]:
                    yield target, accession

    insert_records(connection, "INSERT OR IGNORE INTO target_accessions VALUES (?, ?)", accessions())

    def genes():
        for columns, values in reader("component_synonyms", ["component_id", "component_synonym", "syn_type"]):
            component = int(field(columns, values, "component_id"))
            if component in components and field(columns, values, "syn_type") == "GENE_SYMBOL":
                symbol = field(columns, values, "component_synonym")
                if symbol:
                    for target in components[component]:
                        yield target, symbol

    insert_records(connection, "INSERT OR IGNORE INTO target_genes VALUES (?, ?)", genes())
    counts = {table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
              for table in ("targets", "assays", "activities", "molecules", "target_genes", "target_accessions")}
    if not counts["molecules"] or not counts["activities"]:
        raise ValueError("No structured compounds remain")
    counts["activities_removed_for_missing_structures"] = retained_activities - counts["activities"]
    counts["targets_with_activity"] = connection.execute("SELECT count(DISTINCT tid) FROM activities").fetchone()[0]
    connection.execute("PRAGMA optimize")
    connection.commit()
    return counts


def export_csv(connection, query, path):
    cursor = connection.execute(query)
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=1) as stream:
        writer = csv.writer(stream)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)


def extract(root, archive_path, image, output):
    if not archive_path.is_file() or not image.is_file():
        raise FileNotFoundError("Both the ChEMBL archive and PostgreSQL container image must exist")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "chembl_filtered.sqlite").exists():
        raise FileExistsError("Filtered database already exists; use --output with a new directory to rerun")
    dump = extract_dump(archive_path, output / "source")
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output) as temporary:
        working = Path(temporary)
        database = working / "chembl_filtered.sqlite"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("PRAGMA temp_store=FILE")
            def reader(table, required):
                return table_rows(image, dump, table, required)

            counts = populate(connection, reader)
            export_csv(connection, "SELECT * FROM molecules", working / "molecules.csv.gz")
            export_csv(connection, "SELECT activities.*, assays.confidence_score, assays.assay_type FROM activities JOIN assays USING (assay_id)", working / "activities.csv.gz")
            export_csv(connection, "SELECT * FROM targets", working / "targets.csv.gz")
            export_csv(connection, "SELECT * FROM target_genes", working / "target_genes.csv.gz")
            export_csv(connection, "SELECT * FROM target_accessions", working / "target_accessions.csv.gz")
        finally:
            connection.close()
        report = {"source_archive": archive_path.name, "source_archive_bytes": archive_path.stat().st_size,
                  "dump_bytes": dump.stat().st_size, "counts": counts,
                  "filters": {"organism": "Homo sapiens", "target_type": "SINGLE PROTEIN", "minimum_assay_confidence": 9,
                              "endpoints": ["IC50", "Ki", "Kd", "EC50"], "standard_relation": "=",
                              "require_finite_pchembl": True, "exclude_validity_flags": True, "exclude_potential_duplicates": True},
                  "notes": ["SQL from the source archive is parsed as COPY data, not executed.",
                            "Repeated measurements and source activity/assay IDs are preserved; endpoints are not pooled.",
                            "Structures and InChIKeys are ChEMBL source representations, not yet standardized with our RDKit pipeline.",
                            "The full filtered reference is not a training set. Apply frozen scaffold exclusions before training or retrieval.",
                            "Known molecule targets must remain unavailable for held-out compound prediction."]}
        (working / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
        for path in working.iterdir():
            path.replace(output / path.name)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    extract(root, (arguments.archive or root / "chembl_37_postgresql.tar.gz").resolve(),
            (arguments.image or root / "containers" / "postgres16.sif").resolve(),
            (arguments.output or root / "data" / "chembl").resolve())
