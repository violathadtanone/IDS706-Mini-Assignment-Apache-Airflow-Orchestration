"""Small file-based stages; XCom carries paths and counts, never dataframes."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import psycopg2

ROOT = Path('/opt/airflow')
FILES = {
    'hospitals': 'Hospital_General_Information_2.csv',
    'infections': 'Healthcare_Associated_Infections-Hospital.csv',
}
HOSPITAL_COLUMNS = {
    'Facility ID': 'facility_id', 'Facility Name': 'facility_name',
    'State': 'state', 'Hospital Type': 'hospital_type',
    'Hospital overall rating': 'overall_rating',
}
INFECTION_COLUMNS = {
    'Facility ID': 'facility_id', 'Measure ID': 'measure_id',
    'Measure Name': 'measure_name', 'Score': 'score',
    'Compared to National': 'compared_to_national',
    'Start Date': 'start_date', 'End Date': 'end_date',
}


def ingest(kind, run_id):
    # Isolate scheduled runs and retries without putting raw run IDs in paths.
    folder = ROOT / 'output' / hashlib.sha256(run_id.encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'dataset' / FILES[kind]
    destination = folder / f'{kind}_raw.csv'
    shutil.copyfile(source, destination)
    return str(destination)


def clean(path, kind):
    columns = HOSPITAL_COLUMNS if kind == 'hospitals' else INFECTION_COLUMNS
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing {kind} columns: {sorted(missing)}')
    frame = frame[list(columns)].rename(columns=columns)
    for column in frame.columns:
        frame[column] = frame[column].str.strip().replace({'': None, 'Not Available': None})
    if frame.facility_id.isna().any():
        raise ValueError(f'{kind} contains missing Facility IDs')
    if kind == 'hospitals':
        if frame.facility_id.duplicated().any():
            raise ValueError('Hospital Facility IDs must be unique')
        frame['overall_rating'] = pd.to_numeric(frame.overall_rating, errors='coerce')
        if not frame.overall_rating.dropna().between(1, 5).all():
            raise ValueError('Hospital ratings must be between 1 and 5')
    else:
        if frame.measure_id.isna().any():
            raise ValueError('Missing infection measure IDs')
        frame['score'] = pd.to_numeric(frame.score, errors='coerce')
        for column in ['start_date', 'end_date']:
            original = frame[column]
            parsed = pd.to_datetime(original, format='%m/%d/%Y', errors='coerce')
            if (original.notna() & parsed.isna()).any():
                raise ValueError(f'Invalid {column}')
            frame[column] = parsed.dt.strftime('%Y-%m-%d')
        keys = ['facility_id', 'measure_id', 'start_date', 'end_date']
        frame = frame.drop_duplicates()
        if frame.duplicated(keys).any():
            raise ValueError('Conflicting infection records for the same facility/measure/period')
    destination = Path(path).with_name(f'{kind}_clean.csv')
    frame.to_csv(destination, index=False)
    return str(destination)


def merge(hospitals, infections):
    left = pd.read_csv(infections, dtype={'facility_id': str})
    right = pd.read_csv(hospitals, dtype={'facility_id': str})
    merged = left.merge(right, on='facility_id', how='left', validate='many_to_one', indicator=True)
    counts = merged['_merge'].value_counts().to_dict()
    print(f'Facility join coverage: {counts}')
    merged['hospital_matched'] = merged.pop('_merge').eq('both')
    if merged.empty or not merged.hospital_matched.any():
        raise ValueError('Merge produced no matched infection records')
    destination = Path(infections).with_name('merged.csv')
    merged.to_csv(destination, index=False)
    return str(destination)


def connect():
    return psycopg2.connect(os.environ['HEALTHCARE_DATABASE_URL'])


def load_database(path):
    # A transactional refresh makes retries safe and preserves the previous
    # successful snapshot if loading fails. Only this DAG owns this table.
    frame = pd.read_csv(path, dtype={'facility_id': str})
    columns = ['facility_id', 'measure_id', 'measure_name', 'score',
               'compared_to_national', 'start_date', 'end_date', 'facility_name',
               'state', 'hospital_type', 'overall_rating', 'hospital_matched']
    buffer = io.StringIO()
    frame[columns].to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    with connect() as conn, conn.cursor() as cursor:
        cursor.execute('''CREATE TABLE IF NOT EXISTS hospital_infections (
            facility_id TEXT NOT NULL, measure_id TEXT NOT NULL,
            measure_name TEXT, score DOUBLE PRECISION, compared_to_national TEXT,
            start_date DATE, end_date DATE, facility_name TEXT, state TEXT,
            hospital_type TEXT, overall_rating DOUBLE PRECISION,
            hospital_matched BOOLEAN NOT NULL)''')
        cursor.execute('TRUNCATE hospital_infections')
        cursor.copy_expert('COPY hospital_infections FROM STDIN WITH (FORMAT CSV)', buffer)
        cursor.execute('SELECT COUNT(*) FROM hospital_infections')
        count = cursor.fetchone()[0]
        if count != len(frame):
            raise ValueError('Database row count does not match merged data')
    return {'rows': count, 'output_dir': str(Path(path).parent)}


def analyze(loaded):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with connect() as conn, conn.cursor() as cursor:
        cursor.execute('''SELECT measure_id, COUNT(DISTINCT facility_id)
            FROM hospital_infections
            WHERE score IS NOT NULL AND hospital_matched
            GROUP BY measure_id ORDER BY measure_id''')
        rows = cursor.fetchall()
        cursor.execute('''SELECT COUNT(*), COUNT(DISTINCT facility_id),
            COUNT(*) FILTER (WHERE NOT hospital_matched), COUNT(score)
            FROM hospital_infections''')
        total, facilities, unmatched, numeric = cursor.fetchone()
        # Keep scores comparable: other measure IDs include counts and
        # confidence limits, which should not be averaged with this ratio.
        measure_id = 'HAI_1_SIR'
        cursor.execute('''SELECT overall_rating, AVG(score), COUNT(*),
                COUNT(DISTINCT facility_id)
            FROM hospital_infections
            WHERE score IS NOT NULL AND overall_rating IS NOT NULL
                AND hospital_matched = TRUE
                AND overall_rating BETWEEN 1 AND 5 AND measure_id = %s
            GROUP BY overall_rating ORDER BY overall_rating''', (measure_id,))
        rating_rows = cursor.fetchall()
    if not rows or total != loaded['rows']:
        raise ValueError('Database read-back validation failed')
    if not rating_rows:
        raise ValueError('No rated hospitals have numeric HAI_1_SIR scores')
    output = Path(loaded['output_dir'])
    pd.DataFrame(rows, columns=['measure_id', 'reporting_facilities']).to_csv(
        output / 'reporting_by_measure.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([row[0] for row in rows], [row[1] for row in rows])
    ax.set(title='Facilities with numeric infection-measure scores',
           xlabel='Infection measure ID', ylabel='Distinct matched facilities')
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    chart = output / 'reporting_by_measure.png'
    fig.savefig(chart, dpi=150)
    plt.close(fig)

    ratings = pd.DataFrame(rating_rows, columns=[
        'overall_rating', 'average_score', 'record_count', 'facility_count'])
    ratings['overall_rating'] = ratings.overall_rating.astype(int)
    ratings.insert(0, 'measure_id', measure_id)
    ratings.to_csv(output / 'infection_score_by_rating.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ratings.overall_rating, ratings.average_score)
    ax.bar_label(bars, labels=[f'n={n}' for n in ratings.facility_count], padding=3)
    ax.set(title='Central-line bloodstream infection score by hospital rating\n'
                 '(HAI_1_SIR; numeric scores only)',
           xlabel='Hospital overall rating', ylabel='Average infection score (SIR)',
           xticks=[1, 2, 3, 4, 5])
    ax.margins(y=0.15)
    fig.text(0.5, 0.01, 'n = reporting facilities; descriptive comparison only',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rating_chart = output / 'infection_score_by_rating.png'
    fig.savefig(rating_chart, dpi=150)
    plt.close(fig)
    summary = {'rows': total, 'facilities': facilities,
               'unmatched_rows': unmatched, 'numeric_scores': numeric,
               'chart': str(chart), 'rating_chart': str(rating_chart),
               'rating_measure_id': measure_id}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)
    return summary
