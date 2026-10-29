# OurAirports: large and medium airports

- Source: OurAirports, <https://ourairports.com/data/>
  (file <https://davidmegginson.github.io/ourairports-data/airports.csv>)
- Licence: public domain. OurAirports: "All data is released to the Public
  Domain, and comes with no guarantee of accuracy or fitness for use."
- Attribution (not required, given as a courtesy): airport data from
  OurAirports (https://ourairports.com), released to the public domain.
- Retrieved: 2026-10-29, by `Tools/fetch_ourairports.py` in TrainingMaterials
- Rows: 5280 airports of type large_airport and medium_airport; columns ident, type, name, latitude_deg, longitude_deg, elevation_ft, continent, iso_country, iso_region, municipality, iata_code.
- `continent` codes: AF, AN, AS, EU, NA, OC, SA. "NA" is North America,
  not a missing value; read this file with Polars or `pd.read_parquet`.
