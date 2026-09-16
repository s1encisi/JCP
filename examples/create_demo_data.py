"""Generate visibly synthetic import rows; writes to the ignored runtime folder."""
import csv
from pathlib import Path

destination = Path(__file__).resolve().parents[1] / '.runtime' / 'synthetic_import.csv'
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['cu_in','temperature','current','flow','duration','mode'])
    writer.writerows([[38,58,18000,117,4,'three_stage'],
                     [38,60,18000,118,4,'four_stage'],
                     [38,58,17000,117,6,'serial']])
print(f'Synthetic example, not plant observations: {destination}')
