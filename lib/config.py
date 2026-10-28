"""Paths, data sources and constants that every script shares.

Data lives under DATA_DIR. The default is data/, and $TESSERA_INTERP_DATA moves it. It holds:
    embeddings/    GeoTessera tile downloads (npy arrays, scales, landmask tiffs). $TESSERA_INTERP_EMBEDDINGS moves it.
    alphaearth/    AlphaEarth tile-years on GeoTessera tile grids
    groundtruth/   reference products, and reference rasters on tile grids
    intermediate/  pixel samples kept instead of whole tiles
    checkpoints/   dictionary checkpoints
Results go to RESULTS_DIR and figures to FIG_DIR (results/ and figures/ by default). $TESSERA_INTERP_RESULTS and
$TESSERA_INTERP_FIGURES move them.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TESSERA_INTERP_DATA", REPO_ROOT / "data"))
EMB_DIR = Path(os.environ.get("TESSERA_INTERP_EMBEDDINGS", DATA_DIR / "embeddings"))
GT_DIR = DATA_DIR / "groundtruth"
CKPT_DIR = DATA_DIR / "checkpoints"
RESULTS_DIR = Path(os.environ.get("TESSERA_INTERP_RESULTS", REPO_ROOT / "results"))
FIG_DIR = Path(os.environ.get("TESSERA_INTERP_FIGURES", REPO_ROOT / "figures"))

# Tessera v1 embeddings on Source Cooperative. This is TESSERA_MIRROR_URL in geotessera 0.10 and later.
# Build URLs with lib.tiles.embedding_url and landmask_url.
GEOTESSERA_BASE_URL = "https://data.source.coop/tessera/tessera"
GEOTESSERA_VERSION = "v1"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

FIRES = {
    "camp2018": dict(tile="grid_-121.55_39.85", fire_year=2018,
                     years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
                     pre="2018-06-01/2018-10-31", post="2019-04-15/2019-06-15"),
    "northcx2020": dict(tile="grid_-121.35_39.65", fire_year=2020,
                        years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
                        pre="2020-06-01/2020-08-15", post="2020-10-20/2020-12-05"),
    "dixie2021": dict(tile="grid_-121.25_40.05", fire_year=2021,
                      years=[2019, 2020, 2021, 2022, 2023, 2024],
                      pre="2021-06-01/2021-07-10", post="2021-10-20/2021-11-30"),
    "caldor2021": dict(tile="grid_-120.55_38.65", fire_year=2021,
                       years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
                       pre="2021-06-01/2021-08-10", post="2021-10-25/2021-12-10"),
    "mosquito2022": dict(tile="grid_-120.75_39.05", fire_year=2022,
                         years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
                         pre="2022-06-01/2022-09-01", post="2022-10-25/2022-12-10"),
}
DIXIE = "dixie2021"
DNBR_BURN_THRESHOLD = 0.27
DIXIE_SAE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

IOWA_TILE = "grid_-93.55_42.05"
IOWA_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
CDL_YEARS = [2018, 2019, 2020, 2021]
CDL_CORN, CDL_SOY = 1, 5

SAE_DIM, SAE_LATENTS, SAE_K = 128, 1024, 16

# The Dixie dictionary, trained on the Dixie Fire tile
DIXIE_CKPT = CKPT_DIR / "sae_k16_m1024.pt"

# The burn family, used by the western US fire setting. lib.screen.anchored_family picks it from the Dixie Fire's
# burned-minus-unburned trajectory in the Dixie dictionary, with the year after the fire as the expected window.
BURN_FAMILY = [2, 115, 161, 186, 192, 216, 231, 243, 302, 318, 320, 384, 430, 462, 468, 490,
                493, 502, 518, 566, 598, 638, 644, 655, 657, 663, 683, 693, 932, 952, 979, 1012]

def dnbr_path(fire):
    return GT_DIR / f"dnbr_{fire}.npy"

# Hansen Global Forest Change release, used by the Amazon arc and forest-loss settings
HANSEN_VERSION = "GFC-2023-v1.11"
# the Hansen granules that cover the Amazon arc
ARC_GRANULES = ["00N_050W", "00N_060W", "00N_070W", "00N_080W",
                "10S_050W", "10S_060W", "10S_070W", "10S_080W"]

# Earth Engine project for the JRC surface water labels
EE_PROJECT = os.environ.get("EE_PROJECT")


def workers(n):
    """A worker count: n, but never more than this machine's CPUs."""
    return max(1, min(n, os.cpu_count() or 1))
