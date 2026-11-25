# Earth embeddings are decomposable records of the land surface

The lib folder holds the shared code for downloading and reading tiles (tiles, alphaearth), the dictionary (sae), finding families (screen), the list of settings (settings), scoring units (measure), statistics (stats), and paths and constants (config). The experiments folder has one script per setting or analysis, and each script trains or loads a dictionary, measures its units and saves the results in the results folder.

The code writes to data, results and figures folders inside the repository folder. To use other folders, set `TESSERA_INTERP_DATA`, `TESSERA_INTERP_RESULTS` and `TESSERA_INTERP_FIGURES`.

## System requirements

The code was implemented and tested on Ubuntu 24.04 with Python 3.10.13 and the package versions in environment.yml. The specific version of every package is in environment.lock.txt (conda) and requirements.lock.txt (pip). The demo needs under 1 GB of memory. Reproducing the whole paper needs about 1.5 TB of disk for the embeddings and reference maps. We ran it on a server with 256 cores and 1.5 TB of memory.

## Setup

```bash
conda env create -f environment.yml
conda activate tessera
export PYTHONPATH=.    # run from the repository folder
```

A fresh install took about 5 minutes on our machine, most of it was downloading the pip packages.

You also need curl, and unrar for two of the landslide inventories.

## Demo

```bash
python demo/demo.py
```

The demo finds the western US burn family on the Dixie Fire and scores it on the 207 held-out western US fires. Its data are about 14 MB of stored results in demo/data, including the Dixie Fire's yearly mean activations and each fire's per-feature contrasts, derived from Tessera embeddings and MTBS fire perimeters. This takes about 30 seconds on four CPU cores. Expected output, which matches the western US row of Table S1:

```
1. Burn family found on the Dixie Fire: 32 of 1024 features
   same as the paper's family (lib.config.BURN_FAMILY): True
2. Held-out western US fires: 207, peaking in the year after the fire: 184 (0.889)
3. Observed rate 0.889, 95% interval [0.845, 0.927] (tile bootstrap)
   Random families: median 0.140, 95% interval [0.024, 0.406], share at or above the observed rate 0.000
   Random live features outside the family: median 0.092
```

## Using the code on your own data

To find a feature family for your own event, you need its tile, its year, and two boolean masks on the tile's 10 m grid for the disturbed and undisturbed pixels. `lib.tiles.tile_for_point(lon, lat)` specifies the tile, and `lib.tiles.tile_grid(tile)` gives the grid to rasterize your polygons onto. Then:

```python
from lib.sae import load_sae, group_means
from lib.screen import anchored_family, peak_outcome
from lib.tiles import download_tile, load_tile

sae = load_sae()               # the Dixie dictionary, or load_sae(path=...) for your own
traj = {}                      # disturbed minus undisturbed activation of every feature, by year since the event
for year in range(2017, 2026):
    download_tile(tile, year)
    m = group_means(sae, load_tile(tile, year), {"disturbed": disturbed, "undisturbed": undisturbed})
    traj[year - event_year] = m["disturbed"] - m["undisturbed"]

family = anchored_family(traj, window=(1,))                      # features that peak in the expected window, Y+1
outcome = peak_outcome({"my_event": traj}, family, window=(1,))   # {unit: (peak offset, dated in the window)}
```

On the Dixie Fire (lib.config.FIRES, with its dNBR burn mask) this recovers the paper's 32-feature burn family. To screen on several events, combine their trajectories with `lib.screen.median_contrasts`, then score other events with `peak_outcome`. To train a dictionary on your own tiles, sample pixels with `lib.tiles.sample_pixels` and train with `lib.sae.train_and_save`. To add a setting to the paper's pipeline, write an experiment script like those in experiments/, register it in lib/settings.py and add its units in pipeline/units.py.

## Data

Tessera embeddings have 128 numbers per 10 m pixel per year. They download tile by tile the first time they are needed (lib/tiles.py). AlphaEarth Foundations embeddings have 64 numbers per pixel and are read from Google's public storage (lib/alphaearth.py).

Reference maps download with `python experiments/fetch.py reference`. Add product names to fetch only some of them, or add --list to see them all. This covers MTBS, NBAC, FORWIND, the European Forest Disturbance Atlas, Hansen Global Forest Change, the RER2023, RLBJ and RLZX landslide inventories, USPVDB, and the Natural Earth land outlines used for the Figure 3 map. The scripts that need the remaining sources download them themselves. Geoscience Australia bushfire boundaries come through se_australia_inventory.py, and the global mining land use dataset comes through mines.py. The USDA Cropland Data Layer, the JRC EU crop map, ESA WorldCover, the Copernicus 30 m elevation model and Sentinel-2 images from Microsoft Planetary Computer are fetched by the scripts that use them.

Two sources need an account. RADD deforestation alerts need a Global Forest Watch API key in `GFW_API_KEY`, which amazon_arc_radd.py uses. JRC Global Surface Water needs Earth Engine, which new_water_labels.py uses. Install earthengine-api and put a project name in `EE_PROJECT`.

## Reproducing the paper

Each experiment lists its steps at the top of its file. Run them in order, for example

```bash
python experiments/landslides.py select
python experiments/landslides.py fetch
python experiments/landslides.py train
python experiments/landslides.py screen
```

Then build the paper's numbers, tables and figures from the results folder:

```bash
python pipeline/units.py          # gather every unit's measurements -> results/units/
python pipeline/events.py         # each unit's outcome -> paper_events.csv
python pipeline/nulls.py          # random-family baselines and 95% intervals -> all_nulls.json
python pipeline/anchor_sweep.py   # how the results change with the family threshold -> anchor_sweep.json
python pipeline/prep_fig1_encode.py
python pipeline/prep_fig2_data.py
python pipeline/prep_figS3_data.py
for f in figures/paper_*.py; do python "$f"; done
python pipeline/paper_numbers.py  # every number the paper quotes -> paper_numbers.json, paper/numbers.tex
python pipeline/tables.py         # supplementary tables
```

## Settings

A setting is one row of the paper's results table. It is made up of a kind of change, the reference map that dates it, and the years in which its family should peak, which we call the expected window. Some maps record when the event happened (event-dated). Others record the first yearly image that shows it, which can be a year later (detection-dated).

Every setting is listed in lib/settings.py. A Dictionary there is one trained dictionary, and several settings can share it. Each unit's outcome comes from the outcomes function in lib/measure.py, which uses peak_outcome from lib/screen.py.

| setting | experiment | saved results |
|---|---|---|
| Western US fire | western_us_fire | western_us_fire.jsonl, western_us_fire_perfeat.jsonl |
| SE US wildfire, prescribed fire | se_us_fire | se_us_fire.json/.jsonl |
| Boreal Canada fire | boreal_fire | boreal.json/.jsonl |
| SE Australia fire | se_australia_fire (fire list from se_australia_inventory) | se_australia_fire.json/.jsonl |
| Europe harvest, wind/bark beetle, fire | europe_atlas | europe_atlas.json/.jsonl |
| Windthrow (Friederike, Xavier, Vaia) | windthrow | windthrow.json/.jsonl |
| Amazon arc clearing | amazon_arc | amazon_arc.json/.jsonl |
| Africa, East Asia forest loss | forest_loss africa, forest_loss eastasia | forest_loss_africa.json/.jsonl, forest_loss_eastasia.json/.jsonl |
| Rainfall landslides (Emilia-Romagna, Beijing, Zixing) | landslides | landslides.json/.jsonl |
| New permanent water | new_water (labels from new_water_labels) | new_water.json/.jsonl |
| Crop atlas, crop rotation | crop_atlas, crop_rotation | crop_atlas.json, crop_rotation.json |
| Mining land, solar farms (supplement) | mines, solar | mines.json, solar.json |

Other analyses:

| analysis | experiment | saved results |
|---|---|---|
| The Dixie dictionary and its burn family | dixie_dictionary | the checkpoint, feature_stats.json |
| AlphaEarth Dixie dictionary (its burn family is used in the western US AlphaEarth run) | alphaearth_dixie | alphaearth_stats.json |
| Seed check of the burn family | dixie_seeds | dixie_seeds.json |
| Specificity of the burn family on other disturbances | specificity (tables from europe_agents_dixie, europe_fire_dixie) | specificity.json |
| Precursor test (Europe wind/bark beetle) | precursor | precursor.json |
| Sub-annual timing with RADD alerts (Amazon arc) | amazon_arc_radd | amazon_arc_radd.json |

## Figures

| figure | script | reads | made by |
|---|---|---|---|
| 1 | paper_fig1_decompose.py | fig1_dixie_feats.npy, fire_trajectories.json, the Dixie tile | prep_fig1_encode, fire_trajectories, dixie_dictionary |
| 2 | paper_fig2_sparsity.py | fig2_data.json/.npz | prep_fig2_data (iowa_dictionary) |
| 3 | paper_fig3_scale.py | paper_events.csv, all_nulls.json | events, nulls |
| 4 | paper_fig4_timing.py | paper_events.csv | events |
| 5 | paper_fig5_boundaries.py | ablation_results.json, se_us_fire.json/.jsonl, se_us_fire_extra.jsonl, europe_atlas.json/.jsonl | causal_ablation, se_us_fire, europe_atlas |
| 6 | paper_fig6_crossmodel.py | raw_retention.json, crop_aef.json, western_us_fire.jsonl, western_us_fire_aef.jsonl | raw_retention, crop_atlas, western_us_fire |
| S1 | paper_figS1_purity.py | feature_purity*.json | feature_purity |
| S2 | paper_figS2_capacity.py | m8192.json, alphaearth_capacity.json | burn_capacity, alphaearth_capacity |
| S3 | paper_figS3_invariance.py | figS3_data.npz | prep_figS3_data (fire_invariance, rice_invariance, wheat_invariance) |
