# AI/ML-enhanced RWDS analysis notebook-style script
# Run with: `python rwds_offline_notebook.py`
# Cells separated by comments. Designed to run offline-first with an end-to-end pipeline
# and a minimal smoke-test fallback when geospatial/ML dependencies are unavailable.

# ==== Cell: Imports and basic utilities ====
import os
import sys
import json
import math
import csv
import random
import itertools
import warnings
from pathlib import Path
from datetime import datetime
import importlib.util
from typing import Dict, List, Tuple

# ==== Cell: Module availability helpers ====
REQUIRED_LIBS = [
    "geopandas",
    "shapely",
    "pyproj",
    "networkx",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "pyarrow",
]


def module_available(name: str) -> bool:
    """Return True if a module can be imported."""
    return importlib.util.find_spec(name) is not None


def availability_report():
    return {name: module_available(name) for name in REQUIRED_LIBS}


HAS_TQDM = module_available("tqdm")
if HAS_TQDM:
    from tqdm import tqdm  # type: ignore
else:
    def tqdm(x, **kwargs):
        return x

# ==== Cell: Paths and configuration ====
SEED = 42
random.seed(SEED)

CANDIDATE_WORKDIRS = ["./", "/workspace", "/mnt/data", "/mnt", "/home/sandbox", "/home/coder"]
BASE_DIR = Path(next((str(p) for p in CANDIDATE_WORKDIRS if Path(p).exists()), "./")).resolve()
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
for d in [DATA_DIR, OUT_DIR, CACHE_DIR, OUT_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

CRS_TARGET = "EPSG:5181"
WALK_SPEED_KMPH = 4.5
WALK_SPEED_MPS = WALK_SPEED_KMPH * 1000 / 3600
NODE_BUFFER_METERS = 50
ISO_BREAKS_MINUTES = [5, 10, 15, 20, 30]
RWDS_BREAK = 15
TWO_SFCA_CUTOFF = 30
BANDS = [(0, 10, 1.0), (10, 20, 0.6), (20, 30, 0.3)]
LAMBDA_QUALITY = 0.75

APT_GPKG = DATA_DIR / "apartments.gpkg"
POI_CSV = DATA_DIR / "poi.csv"
POI_GPKG = DATA_DIR / "poi.gpkg"
GRAPH_GRAPHML = DATA_DIR / "graph.graphml"
NODES_GPKG = DATA_DIR / "nodes.gpkg"
EDGES_GPKG = DATA_DIR / "edges.gpkg"
ADMIN_BOUNDARY = DATA_DIR / "boundary.gpkg"
SATISFACTION_CSV = DATA_DIR / "satisfaction.csv"
IMAGE_DIR = DATA_DIR / "street_images"

CATEGORY_TAXONOMY = [
    "medical",
    "transit",
    "public",
    "retail_grocery",
    "food",
    "leisure_green",
    "childcare_edu",
]

OSM_CATEGORY_TAGS = {
    "medical": [("amenity", "clinic"), ("amenity", "hospital"), ("amenity", "doctors"), ("healthcare", "*")],
    "transit": [("public_transport", "platform"), ("railway", "station"), ("amenity", "bus_station")],
    "public": [("amenity", "townhall"), ("amenity", "library"), ("amenity", "community_centre")],
    "retail_grocery": [("shop", "supermarket"), ("shop", "convenience"), ("amenity", "marketplace")],
    "food": [("amenity", "restaurant"), ("amenity", "cafe"), ("amenity", "fast_food")],
    "leisure_green": [("leisure", "park"), ("landuse", "grass"), ("leisure", "garden")],
    "childcare_edu": [("amenity", "kindergarten"), ("amenity", "school"), ("amenity", "childcare")],
}

# ==== Cell: Run mode detection ====

def detect_run_modes():
    files_present = any([APT_GPKG.exists(), POI_CSV.exists(), POI_GPKG.exists(), GRAPH_GRAPHML.exists()])
    smoke = not files_present
    offline = True
    online = False
    if os.environ.get("ONLINE_MODE", "0") == "1":
        online = True
        offline = False
    if os.environ.get("OFFLINE_MODE", "1") == "0":
        offline = False
    return smoke, offline, online


SMOKE_TEST, OFFLINE_MODE, ONLINE_MODE = detect_run_modes()

# ==== Cell: Preflight reporting ====

def preflight(missing_packages: list[str]):
    env_info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "cache_dir": str(CACHE_DIR),
        "missing_packages": missing_packages,
        "smoke_test": SMOKE_TEST,
        "offline_mode": OFFLINE_MODE,
        "online_mode": ONLINE_MODE,
        "crs_target": CRS_TARGET,
    }
    existing = [
        str(p)
        for p in [APT_GPKG, POI_CSV, POI_GPKG, GRAPH_GRAPHML, NODES_GPKG, EDGES_GPKG]
        if p.exists()
    ]
    print("==== Preflight Report ====")
    print(json.dumps(env_info, indent=2))
    print("Detected files:", existing)
    print("==========================")
    preflight_payload = {"env": env_info, "detected_files": existing}
    with (OUT_DIR / "preflight_report.json").open("w") as f:
        json.dump(preflight_payload, f, indent=2)
    return preflight_payload


availability = availability_report()
missing_packages = [k for k, v in availability.items() if not v]
preflight_report = preflight(missing_packages)

# ==== Cell: Dependency hints ====

def print_install_hints(missing: list[str]):
    if not missing:
        return
    print("Missing packages detected. Please install using pip or conda:")
    pip_cmd = "pip install " + " ".join(missing)
    conda_cmd = "conda install -c conda-forge " + " ".join(missing)
    print("Pip:", pip_cmd)
    print("Conda:", conda_cmd)


print_install_hints(missing_packages)

# ==== Cell: Minimal fallback utilities (pure Python) ====

def dijkstra(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str, float]], origin_id: str) -> Dict[str, float]:
    """Simple Dijkstra on a dictionary graph returning travel time seconds."""
    adj = {n: [] for n in nodes}
    for u, v, length in edges:
        t = length / WALK_SPEED_MPS
        adj[u].append((v, t))
        adj[v].append((u, t))
    import heapq

    dist = {n: math.inf for n in nodes}
    dist[origin_id] = 0.0
    heap = [(0.0, origin_id)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent


def generate_synthetic_data(num_apts: int = 6, num_pois: int = 18):
    """Generate synthetic apartments, POIs, a grid network, and optional satisfaction."""
    apartments = []
    for i in range(num_apts):
        x = random.uniform(200, 1800)
        y = random.uniform(200, 1800)
        apartments.append({"apt_id": f"A{i+1}", "x": x, "y": y, "households": random.randint(1, 4)})

    categories = CATEGORY_TAXONOMY[:-1]
    pois = []
    for i in range(num_pois):
        x = random.uniform(100, 1900)
        y = random.uniform(100, 1900)
        cat = random.choice(categories)
        pois.append({"poi_id": f"P{i+1}", "category": cat, "x": x, "y": y})

    # grid network 4x4
    nodes = {}
    edges = []
    step = 700
    node_id = 0
    for i in range(4):
        for j in range(4):
            node_id += 1
            nodes[str(node_id)] = (100 + i * step, 100 + j * step)
    for i in range(4):
        for j in range(4):
            nid = i * 4 + j + 1
            if i < 3:
                nid2 = (i + 1) * 4 + j + 1
                edges.append((str(nid), str(nid2), step))
            if j < 3:
                nid2 = i * 4 + (j + 1) + 1
                edges.append((str(nid), str(nid2), step))

    satisfaction = []
    for apt in apartments:
        satisfaction.append({"apt_id": apt["apt_id"], "satisfaction_score": round(random.uniform(3, 5), 2)})

    return apartments, pois, nodes, edges, satisfaction


def nearest_node(nodes: Dict[str, Tuple[float, float]], x: float, y: float):
    best = None
    best_d = math.inf
    for nid, (nx, ny) in nodes.items():
        d = math.hypot(nx - x, ny - y)
        if d < best_d:
            best_d = d
            best = nid
    return best


def compute_isochrones(apartments, nodes, edges, breaks_minutes):
    iso = []
    for apt in apartments:
        origin = nearest_node(nodes, apt["x"], apt["y"])
        dist = dijkstra(nodes, edges, origin)
        for br in breaks_minutes:
            cutoff = br * 60
            reachable = [nid for nid, t in dist.items() if t <= cutoff]
            iso.append({"apt_id": apt["apt_id"], "minutes": br, "nodes": reachable})
    return iso


def compute_metrics(apartments, pois, iso):
    metrics = []
    for record in iso:
        apt_id = record["apt_id"]
        minutes = record["minutes"]
        apt = next(a for a in apartments if a["apt_id"] == apt_id)
        counts = {cat: 0 for cat in CATEGORY_TAXONOMY}
        for poi in pois:
            d = math.hypot(poi["x"] - apt["x"], poi["y"] - apt["y"])
            if d <= minutes * WALK_SPEED_MPS * 60:
                counts[poi["category"]] += 1
        richness = sum(1 for v in counts.values() if v > 0)
        shannon = entropy(list(counts.values()))
        total = sum(counts.values()) or 1
        norm_entropy = shannon / math.log(len(CATEGORY_TAXONOMY)) if total else 0
        hhi = sum((c / total) ** 2 for c in counts.values())
        metrics.append(
            {
                "apt_id": apt_id,
                "minutes": minutes,
                "total_poi": total,
                "richness": richness,
                "shannon": shannon,
                "entropy_norm": norm_entropy,
                "hhi": hhi,
                **{f"cnt_{k}": v for k, v in counts.items()},
            }
        )
    return metrics


def compute_2sfca(apartments, pois, bands=BANDS):
    demand = {apt["apt_id"]: apt.get("households", 1) for apt in apartments}
    supply_weight = {poi["poi_id"]: 1 for poi in pois}

    ratios = {}
    for poi in pois:
        denom = 0.0
        for apt in apartments:
            d = math.hypot(poi["x"] - apt["x"], poi["y"] - apt["y"])
            minutes = d / WALK_SPEED_MPS / 60
            w = 0.0
            for lo, hi, wt in bands:
                if lo <= minutes < hi:
                    w = wt
                    break
            if minutes <= TWO_SFCA_CUTOFF:
                denom += demand[apt["apt_id"]] * (w or 1.0)
        ratios[poi["poi_id"]] = 0 if denom == 0 else supply_weight[poi["poi_id"]] / denom

    access = []
    for apt in apartments:
        acc = 0.0
        cat_acc = {cat: 0.0 for cat in CATEGORY_TAXONOMY}
        for poi in pois:
            d = math.hypot(poi["x"] - apt["x"], poi["y"] - apt["y"])
            minutes = d / WALK_SPEED_MPS / 60
            w = 0.0
            for lo, hi, wt in bands:
                if lo <= minutes < hi:
                    w = wt
                    break
            if minutes <= TWO_SFCA_CUTOFF:
                acc += ratios[poi["poi_id"]] * (w or 1.0)
                cat_acc[poi["category"]] += ratios[poi["poi_id"]] * (w or 1.0)
        access.append({"apt_id": apt["apt_id"], "access_2sfca": acc, **{f"access_{c}": v for c, v in cat_acc.items()}})
    return access


def compute_rwds(metrics, access):
    by_apt = {}
    for m in metrics:
        if m["minutes"] != RWDS_BREAK:
            continue
        by_apt[m["apt_id"]] = m
    results = []
    for acc in access:
        apt_id = acc["apt_id"]
        m = by_apt.get(apt_id)
        if not m:
            continue
        opp = m["total_poi"]
        ent = m["entropy_norm"]
        rwds1 = opp * ent
        rwds2 = sum(acc.get(f"access_{c}", 0) for c in CATEGORY_TAXONOMY)
        results.append(
            {
                "apt_id": apt_id,
                "rwds1": rwds1,
                "rwds2": rwds2,
                "entropy_norm": ent,
                "total_poi": opp,
                "access_2sfca": acc["access_2sfca"],
            }
        )
    return results


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ==== Cell: Full pipeline (requires dependencies) ====

def build_edge_quality(edges_gdf):
    def score_row(row):
        quality = 0.5
        tags = [str(row.get(col, "")).lower() for col in ["sidewalk", "highway", "surface", "lit"]]
        if "footway" in tags or "residential" in tags:
            quality += 0.1
        if "no" in tags[0]:
            quality -= 0.1
        if "unpaved" in tags[2] or "gravel" in tags[2]:
            quality -= 0.1
        quality = max(0.0, min(1.0, quality))
        return quality

    if "quality" not in edges_gdf.columns:
        edges_gdf["quality"] = edges_gdf.apply(score_row, axis=1)
    edges_gdf["quality"] = edges_gdf["quality"].fillna(0.5)
    edges_gdf["travel_time"] = edges_gdf["length_m"] / WALK_SPEED_MPS
    edges_gdf["travel_time_adj"] = edges_gdf["travel_time"] * (1 + LAMBDA_QUALITY * (1 - edges_gdf["quality"]))
    return edges_gdf


def load_apartments(gpd, pd):
    if APT_GPKG.exists():
        gdf = gpd.read_file(APT_GPKG, layer="apartments")
        if gdf.crs is None:
            raise ValueError("Apartments GPKG missing CRS; please set it.")
        gdf = gdf.to_crs(CRS_TARGET)
    else:
        raise FileNotFoundError("Apartments file not found and SMOKE_TEST is False.")
    if "apt_id" not in gdf.columns:
        raise ValueError("apartments layer must have apt_id column")
    return gdf


def load_pois(gpd, pd):
    if POI_CSV.exists():
        df = pd.read_csv(POI_CSV)
        if {"lon", "lat"}.issubset(df.columns):
            geometry = gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326")
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(CRS_TARGET)
        else:
            raise ValueError("poi.csv must contain lon and lat columns")
    elif POI_GPKG.exists():
        gdf = gpd.read_file(POI_GPKG)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf = gdf.to_crs(CRS_TARGET)
    else:
        raise FileNotFoundError("POI file not found and SMOKE_TEST is False.")
    if "category" not in gdf.columns:
        raise ValueError("POIs must have category column")
    return gdf


def load_network(gpd, nx):
    if GRAPH_GRAPHML.exists():
        G = nx.read_graphml(GRAPH_GRAPHML)
    elif NODES_GPKG.exists() and EDGES_GPKG.exists():
        nodes = gpd.read_file(NODES_GPKG)
        edges = gpd.read_file(EDGES_GPKG)
        if nodes.crs is None or edges.crs is None:
            raise ValueError("Network files require CRS")
        nodes = nodes.to_crs(CRS_TARGET)
        edges = edges.to_crs(CRS_TARGET)
        G = nx.Graph()
        for _, row in nodes.iterrows():
            G.add_node(str(row["node_id"]), x=row.geometry.x, y=row.geometry.y)
        for _, row in edges.iterrows():
            G.add_edge(str(row["u"]), str(row["v"]), length_m=row["length_m"])
    elif ONLINE_MODE and not OFFLINE_MODE:
        import osmnx as ox

        place = ""
        if ADMIN_BOUNDARY.exists():
            boundary = gpd.read_file(ADMIN_BOUNDARY)
            boundary = boundary.to_crs("EPSG:4326")
            centroid = boundary.geometry.unary_union.centroid
            place = f"{centroid.y}, {centroid.x}"
        G = ox.graph_from_point(
            (centroid.y if place else 37.5665, centroid.x if place else 126.9780),
            dist=2000,
            network_type="walk",
        )
        G = ox.project_graph(G, to_crs=CRS_TARGET)
    else:
        raise FileNotFoundError("Network files missing and offline mode enforced")
    return G


def graph_to_edges_nodes(gpd, pd, nx, G):
    from shapely.geometry import LineString

    nodes = []
    edges = []
    for nid, data in G.nodes(data=True):
        nodes.append({"node_id": str(nid), "geometry": gpd.points_from_xy([data.get("x")], [data.get("y")], crs=CRS_TARGET)[0]})
    for u, v, data in G.edges(data=True):
        length = data.get("length", data.get("length_m", math.hypot(G.nodes[u]["x"] - G.nodes[v]["x"], G.nodes[u]["y"] - G.nodes[v]["y"])))
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        edges.append({"u": str(u), "v": str(v), "length_m": length, "geometry": geom})
    nodes_gdf = gpd.GeoDataFrame(pd.DataFrame(nodes), geometry="geometry", crs=CRS_TARGET)
    edges_gdf = gpd.GeoDataFrame(pd.DataFrame(edges), geometry="geometry", crs=CRS_TARGET)
    return nodes_gdf, edges_gdf


def enrich_apartments_geometry(gpd, apt_gdf):
    if apt_gdf.geometry.is_empty.any():
        apt_gdf["geometry"] = apt_gdf.centroid
    if apt_gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        apt_gdf["geometry"] = apt_gdf.centroid
    return apt_gdf


def compute_isochrones_full(gpd, nx, nodes_gdf, edges_gdf, apt_points):
    edges_gdf = build_edge_quality(edges_gdf)
    G = nx.Graph()
    for _, row in nodes_gdf.iterrows():
        G.add_node(str(row["node_id"]), x=row.geometry.x, y=row.geometry.y)
    for _, row in edges_gdf.iterrows():
        G.add_edge(str(row["u"]), str(row["v"]), travel_time=row["travel_time"], travel_time_adj=row["travel_time_adj"], geometry=row.geometry)

    iso_rows = []
    iso_rows_q = []
    for _, apt in apt_points.iterrows():
        origin = min(G.nodes, key=lambda n: math.hypot(G.nodes[n]["x"] - apt.geometry.x, G.nodes[n]["y"] - apt.geometry.y))
        for minutes in ISO_BREAKS_MINUTES:
            cutoff = minutes * 60
            res = nx.single_source_dijkstra_path_length(G, origin, weight="travel_time", cutoff=cutoff)
            reach_nodes = [n for n, t in res.items() if t <= cutoff]
            buffers = nodes_gdf[nodes_gdf["node_id"].astype(str).isin(reach_nodes)].buffer(NODE_BUFFER_METERS)
            iso_rows.append({"apt_id": apt["apt_id"], "minutes": minutes, "geometry": buffers.unary_union})

            res_q = nx.single_source_dijkstra_path_length(G, origin, weight="travel_time_adj", cutoff=cutoff)
            reach_nodes_q = [n for n, t in res_q.items() if t <= cutoff]
            buffers_q = nodes_gdf[nodes_gdf["node_id"].astype(str).isin(reach_nodes_q)].buffer(NODE_BUFFER_METERS)
            iso_rows_q.append({"apt_id": apt["apt_id"], "minutes": minutes, "geometry": buffers_q.unary_union})
    iso_gdf = gpd.GeoDataFrame(iso_rows, geometry="geometry", crs=CRS_TARGET)
    iso_q_gdf = gpd.GeoDataFrame(iso_rows_q, geometry="geometry", crs=CRS_TARGET)
    return iso_gdf, iso_q_gdf


def metrics_from_isochrones(gpd, pd, poi_gdf, iso_gdf):
    poi_gdf = poi_gdf.to_crs(CRS_TARGET)
    metrics = []
    for minutes in sorted(iso_gdf["minutes"].unique()):
        subset_iso = iso_gdf[iso_gdf["minutes"] == minutes]
        for _, iso in subset_iso.iterrows():
            inside = gpd.sjoin(poi_gdf, gpd.GeoDataFrame([iso], geometry="geometry", crs=CRS_TARGET), predicate="within")
            counts = {cat: int((inside["category"] == cat).sum()) for cat in CATEGORY_TAXONOMY}
            richness = sum(1 for v in counts.values() if v > 0)
            shannon = entropy(list(counts.values()))
            total = sum(counts.values()) or 1
            norm_entropy = shannon / math.log(len(CATEGORY_TAXONOMY)) if total else 0
            hhi = sum((c / total) ** 2 for c in counts.values())
            metrics.append(
                {
                    "apt_id": iso["apt_id"],
                    "minutes": minutes,
                    "total_poi": total,
                    "richness": richness,
                    "shannon": shannon,
                    "entropy_norm": norm_entropy,
                    "hhi": hhi,
                    **{f"cnt_{k}": v for k, v in counts.items()},
                }
            )
    return pd.DataFrame(metrics)


def compute_2sfca_full(pd, gpd, apt_points, poi_gdf):
    demand = apt_points[["apt_id", "households"]].copy()
    demand["weight"] = demand["households"].fillna(1)
    poi = poi_gdf.copy()
    poi["supply"] = 1

    tree = gpd.GeoSeries(apt_points.geometry).sindex

    ratios = {}
    for idx, poi_row in poi.iterrows():
        nearby_idx = list(tree.query(poi_row.geometry.buffer(WALK_SPEED_MPS * TWO_SFCA_CUTOFF * 60)))
        denom = 0.0
        for ridx in nearby_idx:
            apt_row = apt_points.iloc[ridx]
            dist_m = poi_row.geometry.distance(apt_row.geometry)
            minutes = dist_m / WALK_SPEED_MPS / 60
            w = 0.0
            for lo, hi, wt in BANDS:
                if lo <= minutes < hi:
                    w = wt
                    break
            if minutes <= TWO_SFCA_CUTOFF:
                denom += apt_row["households"] * (w or 1.0)
        ratios[poi_row.name] = 0 if denom == 0 else poi_row["supply"] / denom

    access_rows = []
    for _, apt_row in apt_points.iterrows():
        acc = 0.0
        cat_acc = {cat: 0.0 for cat in CATEGORY_TAXONOMY}
        for idx, poi_row in poi.iterrows():
            dist_m = poi_row.geometry.distance(apt_row.geometry)
            minutes = dist_m / WALK_SPEED_MPS / 60
            w = 0.0
            for lo, hi, wt in BANDS:
                if lo <= minutes < hi:
                    w = wt
                    break
            if minutes <= TWO_SFCA_CUTOFF:
                acc += ratios[idx] * (w or 1.0)
                cat_acc[poi_row["category"]] += ratios[idx] * (w or 1.0)
        access_rows.append({"apt_id": apt_row["apt_id"], "access_2sfca": acc, **{f"access_{c}": v for c, v in cat_acc.items()}})
    return pd.DataFrame(access_rows)


def compute_rwds_full(metrics_df, access_df):
    m15 = metrics_df[metrics_df["minutes"] == RWDS_BREAK].set_index("apt_id")
    access_df = access_df.set_index("apt_id")
    rows = []
    for apt_id in m15.index.intersection(access_df.index):
        m = m15.loc[apt_id]
        acc = access_df.loc[apt_id]
        opp = m["total_poi"]
        ent = m["entropy_norm"]
        rwds1 = opp * ent
        rwds2 = acc[[c for c in acc.index if c.startswith("access_")]].sum()
        rows.append({"apt_id": apt_id, "rwds1": rwds1, "rwds2": rwds2, "entropy_norm": ent, "total_poi": opp, "access_2sfca": acc["access_2sfca"]})
    return rows


def clustering(pd, np, sklearn, features_df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    feature_cols = [c for c in features_df.columns if c not in ["apt_id"]]
    X = features_df[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    best_k = None
    best_score = -1
    best_labels = None
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    features_df["cluster"] = best_labels
    silhouette = {"best_k": best_k, "silhouette": best_score}
    return features_df, silhouette


def satisfaction_modeling(pd, np, sklearn, features_df, satisfaction_df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    merged = features_df.merge(satisfaction_df, on="apt_id", how="inner")
    target = merged["satisfaction_score"]
    feature_cols = [c for c in merged.columns if c not in ["apt_id", "satisfaction_score", "cluster"]]
    X = merged[feature_cols].fillna(0)
    results = {}

    def run_model(cols, name):
        model = LinearRegression()
        Xsub = X[cols]
        model.fit(Xsub, target)
        preds = model.predict(Xsub)
        results[name] = {
            "r2": r2_score(target, preds),
            "rmse": math.sqrt(mean_squared_error(target, preds)),
            "features": cols,
        }

    base_cols = [c for c in feature_cols if "cnt_" in c or c.startswith("richness") or c.startswith("entropy")]
    access_cols = base_cols + [c for c in feature_cols if "access" in c]
    rwds_cols = access_cols + [c for c in feature_cols if c.startswith("rwds") or c.startswith("quality")]

    run_model(base_cols, "model_baseline")
    run_model(access_cols, "model_2sfca")
    run_model(rwds_cols, "model_rwds")
    return results


def save_figures(plt, metrics_df, features_df, silhouette_info):
    fig1, ax1 = plt.subplots()
    rwds1_vals = features_df.get("rwds1")
    if rwds1_vals is not None:
        ax1.hist(rwds1_vals.dropna(), bins=10, color="skyblue")
        ax1.set_title("RWDS1 Distribution")
        fig1.savefig(OUT_DIR / "figures" / "rwds1_hist.png")
        plt.close(fig1)

    fig2, ax2 = plt.subplots()
    cnt_cols = [c for c in metrics_df.columns if c.startswith("cnt_") and metrics_df["minutes"].eq(RWDS_BREAK).any()]
    subset = metrics_df[metrics_df["minutes"] == RWDS_BREAK]
    if cnt_cols:
        subset[cnt_cols].sum().plot(kind="bar", ax=ax2, color="orange")
        ax2.set_title("POI counts by category (15 min)")
        fig2.savefig(OUT_DIR / "figures" / "poi_counts.png")
        plt.close(fig2)

    if silhouette_info:
        fig3, ax3 = plt.subplots()
        ax3.bar([silhouette_info["best_k"]], [silhouette_info["silhouette"]], color="green")
        ax3.set_title("Best Silhouette")
        fig3.savefig(OUT_DIR / "figures" / "silhouette.png")
        plt.close(fig3)


# ==== Cell: Full pipeline orchestration ====

def run_full_pipeline():
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import networkx as nx
    import matplotlib.pyplot as plt

    print("Running full RWDS pipeline...")
    apt_gdf = load_apartments(gpd, pd)
    apt_gdf = enrich_apartments_geometry(gpd, apt_gdf)
    apt_points = apt_gdf.copy()
    apt_points["households"] = apt_points.get("households", 1).fillna(1)
    poi_gdf = load_pois(gpd, pd)

    G = load_network(gpd, nx)
    nodes_gdf, edges_gdf = graph_to_edges_nodes(gpd, pd, nx, G)
    edges_gdf = build_edge_quality(edges_gdf)

    iso_gdf, iso_q_gdf = compute_isochrones_full(gpd, nx, nodes_gdf, edges_gdf, apt_points)
    metrics_df = metrics_from_isochrones(gpd, pd, poi_gdf, iso_gdf)
    access_df = compute_2sfca_full(pd, gpd, apt_points, poi_gdf)
    rwds_rows = compute_rwds_full(metrics_df, access_df)
    features_df = pd.DataFrame(rwds_rows).merge(access_df, on="apt_id", how="left")
    metrics_15 = metrics_df[metrics_df["minutes"] == RWDS_BREAK]
    features_df = features_df.merge(metrics_15[["apt_id", "entropy_norm", "richness"] + [c for c in metrics_15.columns if c.startswith("cnt_")]], on="apt_id", how="left")

    silhouette_info = None
    try:
        import sklearn  # noqa: F401
        features_df, silhouette_info = clustering(pd, np, sklearn, features_df)
    except Exception as exc:  # pragma: no cover
        print("Clustering skipped:", exc)

    if SATISFACTION_CSV.exists():
        sat_df = pd.read_csv(SATISFACTION_CSV)
        if "apt_id" not in sat_df.columns:
            sat_df.rename(columns={sat_df.columns[0]: "apt_id"}, inplace=True)
        model_results = satisfaction_modeling(pd, np, None, features_df, sat_df)
        with (OUT_DIR / "model_results.json").open("w") as f:
            json.dump(model_results, f, indent=2)
    else:
        model_results = {}

    outputs_gpkg = OUT_DIR / "outputs.gpkg"
    apt_points.to_file(outputs_gpkg, layer="apt_points", driver="GPKG")
    poi_gdf.to_file(outputs_gpkg, layer="poi_points", driver="GPKG")
    iso_gdf.to_file(outputs_gpkg, layer="isochrones_time", driver="GPKG")
    iso_q_gdf.to_file(outputs_gpkg, layer="isochrones_quality", driver="GPKG")
    edges_gdf.to_file(outputs_gpkg, layer="edge_quality", driver="GPKG")

    metrics_df.to_csv(OUT_DIR / "apt_metrics_long.csv", index=False)
    metrics_df[metrics_df["minutes"] == RWDS_BREAK].to_csv(OUT_DIR / "apt_metrics.csv", index=False)
    metrics_df.to_parquet(OUT_DIR / "apt_metrics.parquet", index=False)
    access_df.to_csv(OUT_DIR / "access_2sfca.csv", index=False)
    pd.DataFrame(rwds_rows).to_csv(OUT_DIR / "rwds.csv", index=False)
    features_df.to_csv(OUT_DIR / "cluster_results.csv", index=False)

    save_figures(plt, metrics_df, features_df, silhouette_info)

    summary = {
        "apartments": len(apt_points),
        "pois": len(poi_gdf),
        "edges": len(edges_gdf),
        "isochrones": len(iso_gdf),
        "outputs_gpkg": str(outputs_gpkg),
    }
    with (OUT_DIR / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Run summary:")
    print(json.dumps(summary, indent=2))


# ==== Cell: Minimal smoke pipeline (no external deps) ====

def run_minimal_smoke(reason: str = "auto"):
    print("Running minimal SMOKE_TEST pipeline (no external GIS/ML packages available or data missing)...")
    apartments, pois, nodes, edges, satisfaction = generate_synthetic_data()
    iso = compute_isochrones(apartments, nodes, edges, ISO_BREAKS_MINUTES)
    metrics = compute_metrics(apartments, pois, iso)
    access = compute_2sfca(apartments, pois)
    rwds = compute_rwds(metrics, access)

    save_csv(OUT_DIR / "apt_metrics.csv", metrics)
    save_csv(OUT_DIR / "access_2sfca.csv", access)
    save_csv(OUT_DIR / "rwds.csv", rwds)

    summary = {
        "apartments": len(apartments),
        "pois": len(pois),
        "metrics_rows": len(metrics),
        "access_rows": len(access),
        "rwds_rows": len(rwds),
        "mode": "minimal_smoke",
        "reason": reason,
        "missing_packages": missing_packages,
    }
    with (OUT_DIR / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Run summary:")
    print(json.dumps(summary, indent=2))


# ==== Cell: Entry point ====
if __name__ == "__main__":
    try:
        if SMOKE_TEST or missing_packages:
            reason = "smoke_flag" if SMOKE_TEST else "missing_packages"
            run_minimal_smoke(reason=reason)
        else:
            run_full_pipeline()
    except Exception as exc:  # pragma: no cover
        import traceback

        print("Pipeline failed with an exception:")
        traceback.print_exc()
        tb = traceback.TracebackException.from_exception(exc)
        if tb.stack:
            last = tb.stack[-1]
            print(f"Failing at {last.filename}:{last.lineno} in {last.name}")
        raise
