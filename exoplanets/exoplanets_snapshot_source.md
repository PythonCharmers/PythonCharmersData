### Data source

This teaching snapshot was downloaded from the official [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

- Snapshot date: April 7, 2026
- Table: `pscomppars` (Planetary Systems Composite Parameters)
- Source docs: [NASA Exoplanet Archive API queries](https://exoplanetarchive.ipac.caltech.edu/docs/API_queries.html)

The CSV in this folder is intentionally **curated for teaching**.
The query only keeps rows where the fields used by the demo app are already present, so the Streamlit project can focus on app-building rather than data-cleaning.

Selected columns:

- `pl_name`
- `hostname`
- `discoverymethod`
- `disc_year`
- `disc_facility`
- `sy_dist`
- `sy_vmag`
- `st_spectype`
- `st_teff`
- `st_rad`
- `st_mass`
- `st_lum`
- `pl_orbper`
- `pl_orbsmax`
- `pl_rade`
- `pl_bmasse`
- `pl_eqt`
- `pl_insol`

TAP query used:

```sql
select top 180 *
from (
  select
    pl_name,
    hostname,
    discoverymethod,
    disc_year,
    disc_facility,
    sy_dist,
    sy_vmag,
    st_spectype,
    st_teff,
    st_rad,
    st_mass,
    st_lum,
    pl_orbper,
    pl_orbsmax,
    pl_rade,
    pl_bmasse,
    pl_eqt,
    pl_insol
  from pscomppars
  where st_spectype is not null
    and sy_dist is not null
    and sy_vmag is not null
    and st_teff is not null
    and st_rad is not null
    and st_mass is not null
    and st_lum is not null
    and pl_orbper is not null
    and pl_orbsmax is not null
    and pl_rade is not null
    and pl_bmasse is not null
    and pl_eqt is not null
    and pl_insol is not null
  order by disc_year desc, pl_name
) as ranked
```

The `TOP 180` keyword is applied **after** the inner `ORDER BY`, so the snapshot is the 180 most recent rows that have all required fields populated.
A naive `select top 180 ... order by disc_year desc` does not guarantee that result, because ADQL's `TOP` and `ORDER BY` are otherwise independent.

Reproduce the snapshot by submitting the query to the [TAP sync endpoint](https://exoplanetarchive.ipac.caltech.edu/TAP/sync) with `format=csv`.
Note that the live archive grows, so a fresh download will not be byte-identical to the file in this folder.
The catalogue here is intentionally pinned in time so the teaching examples stay reproducible.
