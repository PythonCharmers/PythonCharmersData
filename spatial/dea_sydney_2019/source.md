### Data source

Landsat 8 surface reflectance, cloud mask and land cover for one window of
western Sydney (Penrith east to Parramatta, Windsor/Richmond south to
Liverpool), from [Digital Earth Australia](https://www.dea.ga.gov.au/)
(Geoscience Australia). Built by `Tools/fetch_dea_landsat.py` in the
TrainingMaterials repo.

Products:

- DEA Surface Reflectance NBART (Landsat 8 OLI-TIRS) and DEA Surface
  Reflectance OA (Landsat 8 OLI-TIRS), both part of DEA Surface
  Reflectance (Landsat 8 OLI-TIRS), Collection 3, product `ga_ls8c_ard_3`,
  dataset version 3.1.0 (the DEA product pages list product version
  3.0.0; this scene's `odc:dataset_version` is 3.1.0, as in the file
  names). Scene `ga_ls8c_ard_3-1-0_090083_2019-10-28_final` (WRS-2 path 090 row 083, acquired
  2019-10-28 ~23:50 UTC, i.e. 29 Oct local; Landsat product ID
  LC08_L1TP_090083_20191028_20191114_01_T1). STAC item: https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_ard_3-1-0_090083_2019-10-28_final.stac-item.json
  Docs: https://knowledge.dea.ga.gov.au/data/product/dea-surface-reflectance-nbart-landsat-8-oli-tirs/
  and https://knowledge.dea.ga.gov.au/data/product/dea-surface-reflectance-oa-landsat-8-oli-tirs/
- DEA Land Cover (Landsat), product `ga_ls_landcover_class_cyear_3`,
  version 2.0.0, 2019 calendar-year continental mosaic, level 3.
  Docs: https://knowledge.dea.ga.gov.au/data/product/dea-land-cover-landsat/

### Window and grid

All three rasters share one grid: CRS EPSG:32656 (WGS 84 / UTM 56N,
as delivered by DEA), 30 m pixels, 1200 columns x 1500 rows,
pixel window col_off=5190, row_off=5220 of the
scene's 7621 x 7691 grid. The reflectance and fmask are cut from the
scene without resampling. The northings are negative because DEA, like
USGS, keeps southern-hemisphere Landsat scenes in the north-zone UTM CRS
(EPSG:32656) rather than the south zone (EPSG:32756, which adds a
10,000,000 m false northing).

- Bounds (EPSG:32656): left 283785, bottom -3760785,
  right 319785, top -3715785
- Bounds (lon/lat, EPSG:4326): lon 150.6599 to 151.0585,
  lat -33.9722 to -33.5599

Why this window: the scene is 99.5% clear but mostly Blue Mountains
bushland. Windows from 1000x1000 to 2000x1500 px were scored at a 10-px
stride by the smallest level-3 class count, keeping only those with no
nodata (reflectance, fmask or land cover) and at least 97% fmask == 1.
This 1200 x 1500 window ranked first: it covers suburbs,
Hawkesbury/Nepean farmland, bushland, Prospect Reservoir and the rivers.
Natural aquatic vegetation (124) and natural bare surface (216) stay rare
(see counts below); 124 is concentrated in the Hawkesbury and Georges
River estuaries at the scene's eastern edge, which no full-data window
of this size can reach.

### Files

`landsat8_nbart_20191028.tif`: int16, nodata -999. Values are surface reflectance
scaled by 10,000 (valid range 1 to 10,000; divide by 10,000 for 0-1
reflectance). NBART = nadir BRDF-adjusted reflectance with terrain
illumination correction.

| Band | Description | Wavelength (µm) | Source | URL |
|---|---|---|---|---|
| 1 | blue | 0.45-0.51 | DEA band 2 (nbart_blue) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band02.tif |
| 2 | green | 0.53-0.59 | DEA band 3 (nbart_green) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band03.tif |
| 3 | red | 0.64-0.67 | DEA band 4 (nbart_red) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band04.tif |
| 4 | nir | 0.85-0.88 | DEA band 5 (nbart_nir) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band05.tif |
| 5 | swir_1 | 1.57-1.65 | DEA band 6 (nbart_swir_1) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band06.tif |
| 6 | swir_2 | 2.11-2.29 | DEA band 7 (nbart_swir_2) | https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_nbart_3-1-0_090083_2019-10-28_final_band07.tif |

`fmask_20191028.tif`: uint8 fmask (OA band `oa_fmask`, codes as defined on the
DEA Surface Reflectance OA page above), nodata 0. Source: https://dea-public-data.s3.ap-southeast-2.amazonaws.com/baseline/ga_ls8c_ard_3/090/083/2019/10/28/ga_ls8c_oa_3-1-0_090083_2019-10-28_final_fmask.tif

| Code | Name | Pixels | % |
|---|---|---|---|
| 1 | Clear | 1777393 | 98.74 |
| 2 | Cloud | 2429 | 0.13 |
| 3 | Cloud shadow | 2168 | 0.12 |
| 5 | Water | 18010 | 1.00 |

Fmask labels clear water as 5, not 1, so a "clear observation" mask is
fmask 1 or 5; masking to 1 alone drops most of the water class.

`landcover_level3_2019.tif`: uint8 level-3 class, nodata 255. Reprojected from
the EPSG:3577 (GDA94 / Australian Albers) 30 m mosaic onto the grid above
with nearest-neighbour resampling. Source: https://dea-public-data.s3.ap-southeast-2.amazonaws.com/derivative/ga_ls_landcover_class_cyear_3/2-0-0/continental_mosaics/2019--P1Y/ga_ls_landcover_class_cyear_3_mosaic_2019--P1Y_level3.tif

The Albers and UTM grids are rotated about 10 degrees relative to each
other here, so labels are accurate to about half a pixel and a few per
cent of source pixels are duplicated or dropped. The WGS 84 to GDA94
datum shift (under 2 m) is ignored, as in DEA's own reprojection.

| Code | Name | Pixels | % |
|---|---|---|---|
| 111 | Cultivated Terrestrial Vegetation | 190734 | 10.60 |
| 112 | (Semi-)Natural Terrestrial Vegetation | 1201221 | 66.73 |
| 124 | Natural Aquatic Vegetation | 754 | 0.04 |
| 215 | Artificial Surface | 373810 | 20.77 |
| 216 | Natural Bare Surface | 9356 | 0.52 |
| 220 | Water | 24125 | 1.34 |

### Licence and attribution

Both products are © Commonwealth of Australia (Geoscience Australia) and
released under the [Creative Commons Attribution 4.0 International
Licence](https://creativecommons.org/licenses/by/4.0/). This dataset is a
subset and reprojection, so GA's wording for modified material applies:

> Based on [DEA Surface Reflectance (Landsat 8 OLI-TIRS)](https://knowledge.dea.ga.gov.au/data/product/dea-surface-reflectance-landsat-8-oli-tirs/)
> and [DEA Land Cover (Landsat)](https://knowledge.dea.ga.gov.au/data/product/dea-land-cover-landsat/)
> by Geoscience Australia, which is © Commonwealth of Australia and is
> provided under a [Creative Commons Attribution 4.0 International
> Licence](https://creativecommons.org/licenses/by/4.0/) and is subject to
> the disclaimer of warranties in [section
> 5](https://creativecommons.org/licenses/by/4.0/legalcode#s5) of that
> licence.

(GA attribution guidance:
https://www.ga.gov.au/copyright/copyright-attributing-material-sources-from-geoscience-australias-website)

Landsat level 0 and level 1 data courtesy of the U.S. Geological Survey.

Data citations, verbatim as given by DEA (the first author is Li Fuqin):

- Fuqin, Li., Jupp, D.L.B., Sixsmith, J., Wang, L., Dorj, P., Vincent, A.,
  Alam, I., Hooke, J., Oliver, S., Thankappan, M. 2019. GA Landsat 8
  OLI/TIRS Analysis Ready Data Collection 3. Geoscience Australia,
  Canberra. https://pid.geoscience.gov.au/dataset/ga/132317
- Tissott, B., Mueller, N., Jorand, C., Ebadi, T., Ai, E., Kooymans, C.,
  Harrison, M., Miller, J., Lucas, R., Metternicht, G., Owers, C.J.,
  Clewley, D., Planque, C., Punalekar, S., Chua, S.M.T., Bunting, P.,
  Lewis, B., Lymburner, L., Siggins, A. 2025. DEA Land Cover 30m.
  Geoscience Australia, Canberra. https://doi.org/10.26186/149976

### Rebuild

From the TrainingMaterials repo (anonymous HTTPS, no AWS account needed):

```bash
uv run Tools/fetch_dea_landsat.py --out /Data/spatial/dea_sydney_2019
```

The source URLs pin ARD dataset 3.1.0 ("final") and land cover 2-0-0
through their file names. If DEA withdraws or reprocesses these, the URLs
will stop resolving and the build will fail rather than silently produce
different pixels.
