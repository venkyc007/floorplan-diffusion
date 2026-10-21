"""Downloading and loading the ResPlan dataset.

ResPlan (github.com/m-agour/ResPlan, CC BY 4.0) is one big pickle file:
a Python list of ~17,000 dictionaries, one per apartment. Each dictionary
holds shapely shapes ('wall', 'door', 'bedroom', ...) plus numbers
('id', 'area', ...). Coordinates are NOT metres: every plan is rescaled so
its longest side is ~256 units. Only 'area' (m^2) tells the real size.

Run this file directly to download + unzip the dataset:
    python -m src.stage1.data.resplan
"""

import pickle
import subprocess
import zipfile

import shapely

from src.utils import load_config, project_path

REPO_URL = "https://github.com/m-agour/ResPlan.git"


def download(raw_dir):
    """Clone the ResPlan repo into raw_dir/resplan and unzip ResPlan.pkl.

    We use `git clone` because raw.githubusercontent.com (normal file
    downloads) is blocked on some networks, while github.com works.
    """
    repo_dir = raw_dir / "resplan"
    unzip_dir = raw_dir / "resplan_unzipped"
    if not repo_dir.exists():
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                       check=True)
    if not (unzip_dir / "ResPlan.pkl").exists():
        print("Unzipping ResPlan.zip ...")
        with zipfile.ZipFile(repo_dir / "ResPlan.zip") as z:
            z.extractall(unzip_dir)
    print("ResPlan ready in", unzip_dir)


class _SafeUnpickler(pickle.Unpickler):
    """A pickle loader that refuses anything except the expected data types.

    Normal pickle.load() can run ANY code hidden inside a file. The ResPlan
    file only needs the three things below (numpy numbers and shapely
    shapes), so we allow exactly those and block everything else.
    """

    ALLOWED = {
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "scalar"),   # same thing, older numpy name
        ("numpy", "dtype"),
        ("shapely.io", "from_wkb"),
        ("shapely.wkb", "loads"),              # same thing, older shapely
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Blocked unexpected type {module}.{name}")
        if module.startswith("numpy"):
            return super().find_class(module, name)
        return shapely.from_wkb


def load_plans(pickle_path):
    """Load ResPlan.pkl and return the list of plan dictionaries."""
    with open(pickle_path, "rb") as f:
        return _SafeUnpickler(f).load()


def get_polygons(plan, field):
    """Return a plain list of Polygons for one field of a plan.

    Handles every form the data comes in: missing (None), empty,
    a single Polygon, a MultiPolygon, or a mixed GeometryCollection.
    """
    geom = plan.get(field)
    if geom is None or geom.is_empty:
        return []
    return [g for g in shapely.get_parts(geom)
            if g.geom_type == "Polygon" and not g.is_empty]


if __name__ == "__main__":
    cfg = load_config("stage1/data")
    download(project_path(cfg["paths"]["raw_dir"]))
