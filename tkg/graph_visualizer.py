#!/usr/bin/env python3
"""
Unified Personal Temporal Universe Visualization + Loopable Fly-through PNG Export (multiple trajectories)

- Builds the 3D Plotly figure from your SQLite KG schema (same as your provided code).
- Exports N PNG frames by moving the Plotly 3D camera along a smooth, LOOPABLE trajectory
  (frame 0 == last frame if closed_loop=True) so you can stitch into a video.

Install:
  pip install plotly kaleido numpy

Examples:
  # Write HTML only
  python unified_temporal_universe_flythrough.py --db ../data/output/kg.db --output out.html

  # Write HTML + frames using default trajectory
  python unified_temporal_universe_flythrough.py \
    --db ../data/output/kg.db --output out.html \
    --frames-dir frames --n-frames 240

  # Choose a trajectory
  python unified_temporal_universe_flythrough.py \
    --db ../data/output/kg.db --output out.html \
    --frames-dir frames --n-frames 240 \
    --trajectory topdown_spiral

  # Override trajectory params (JSON)
  python unified_temporal_universe_flythrough.py \
    --db ../data/output/kg.db --output out.html \
    --frames-dir frames --n-frames 240 \
    --trajectory dolly_through \
    --traj-kwargs '{"axis":"y","far":3.4,"near":0.30,"side_sway":0.22}'

Make video (ffmpeg):
  ffmpeg -y -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p flythrough.mp4

Looping note:
- Default is closed-loop: first and last frames are identical (perfect loop, but 1-frame "hold").
- If you want no duplicated last frame, use --open-loop and/or drop the last frame in encoding.
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go


# =========================
# Logging
# =========================

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# =========================
# Color palettes (yours)
# =========================

INNER_COLORS = {
    "project": "#3B82F6",
    "interest": "#10B981",
    "task": "#F59E0B",
    "desire": "#EC4899",
    "plan": "#8B5CF6",
    "problem": "#EF4444",
    "unclassified": "#6B7280",
}

INNER_COLORS_LIGHT = {
    "project": "#93C5FD",
    "interest": "#6EE7B7",
    "task": "#FCD34D",
    "desire": "#F9A8D4",
    "plan": "#C4B5FD",
    "problem": "#FCA5A5",
    "unclassified": "#9CA3AF",
}

CATEGORY_ICONS = {
    "project": "📁",
    "interest": "✨",
    "task": "☑️",
    "desire": "💫",
    "plan": "🗓️",
    "problem": "⚠️",
    "unclassified": "📌",
}

OUTER_COLORS = {
    "PERSON": "#FF6B9D",
    "ORG": "#4ECDC4",
    "LOCATION": "#95E1D3",
    "EMAIL": "#FFA07A",
    "URL": "#87CEEB",
    "DOI": "#DDA0DD",
    "UUID": "#F0E68C",
    "HASH_HEX": "#FFD700",
    "IP_ADDRESS": "#FF7F50",
    "PHONE": "#DA70D6",
    "FILEPATH": "#98FB98",
    "BARE_DOMAIN": "#B0E0E6",
    "OTHER": "#C0C0C0",
}


# =========================
# Config / data structures
# =========================

@dataclass
class UnifiedNebulaConfig:
    # Spatial zones
    self_radius: float = 0.0
    inner_zone_min: float = 0.3
    inner_zone_max: float = 1.2
    transition_zone: float = 1.5
    outer_zone_min: float = 1.5
    outer_zone_max: float = 2.5

    # Shared settings
    z_span: float = 15.0
    background_color: str = "#000008"
    grid_color: str = "#1a1a2e"
    text_color: str = "#E0E0E0"

    # Inner zone
    inner_particle_min: float = 4.0
    inner_particle_max: float = 25.0
    inner_opacity: float = 0.8
    inner_max_topics_per_cluster: int = 20
    inner_min_salience: float = 0.0
    show_cluster_connections: bool = True
    cluster_connection_opacity: float = 0.15

    # Outer zone
    outer_particle_min: float = 2.0
    outer_particle_max: float = 10.0
    outer_opacity: float = 0.6
    outer_max_entities: int = 10_000
    outer_edge_sample_rate: float = 0.08
    outer_min_confidence: float = 0.3
    outer_edge_opacity: float = 0.12
    outer_edge_width: float = 0.5

    # Data limits
    max_windows: int = 36

    # Time filtering
    time_filter: bool = True
    time_start: str = "2023-01-01T00:00:00Z"
    time_end: str = "2026-01-01T00:00:00Z"

    # Visual enhancements
    show_zone_boundaries: bool = True
    show_time_rings: bool = True
    show_time_spine: bool = True

    # SELF marker
    self_size: float = 25.0
    self_color: str = "#FFFFFF"

    # Random seed
    seed: int = 42


@dataclass
class TopicParticle:
    topic_id: str
    label: str
    category: str
    salience: float
    assertion_count: int
    window_label: str
    window_index: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    size: float = 5.0
    color: str = "#FFFFFF"


@dataclass
class EntityParticle:
    entity_id: str
    entity_type: str
    canonical_name: str
    mention_count: int
    first_seen_at: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    size: float = 3.0
    color: str = "#FFFFFF"


@dataclass
class ClusterCloud:
    cluster_id: str
    category: str
    window_index: int
    center_x: float
    center_y: float
    center_z: float
    particles: List[TopicParticle] = field(default_factory=list)


# =========================
# Camera trajectories
# =========================

CameraDict = Dict[str, Dict[str, float]]
TrajectoryFn = Callable[..., CameraDict]


def ease_in_out_cycle(t: float) -> float:
    """0 -> 1 -> 0 smoothly over one loop (perfectly periodic)."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t)


def traj_orbit_time(
    t: float,
    turns: float = 1.0,
    orbit_radius: float = 2.1,
    zoom_amplitude: float = 0.12,
    eye_z_base: float = 0.95,
    eye_z_amplitude: float = 0.35,
    center_z_amplitude: float = 0.55,
    center_xy_amplitude: float = 0.06,
) -> CameraDict:
    """Smooth orbit + subtle zoom, with a periodic bottom->top->bottom center.z tour."""
    theta = 2.0 * math.pi * turns * t
    r = orbit_radius * (1.0 + zoom_amplitude * math.sin(2.0 * math.pi * t))
    eye = {
        "x": r * math.cos(theta),
        "y": r * math.sin(theta),
        "z": eye_z_base + eye_z_amplitude * math.sin(2.0 * math.pi * t),
    }
    center = {
        "x": center_xy_amplitude * math.cos(theta + math.pi / 3.0),
        "y": center_xy_amplitude * math.sin(theta + math.pi / 3.0),
        "z": -center_z_amplitude * math.cos(2.0 * math.pi * t),
    }
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_helix_time(
    t: float,
    turns: float = 1.25,
    orbit_radius: float = 2.2,
    eye_z_low: float = 0.55,
    eye_z_high: float = 1.25,
    center_z_amp: float = 0.65,
    center_xy_amp: float = 0.06,
) -> CameraDict:
    """Helical orbit while the camera height rises/falls; center.z tours bottom->top->bottom."""
    s = ease_in_out_cycle(t)
    theta = 2 * math.pi * turns * t
    eye = {
        "x": orbit_radius * math.cos(theta),
        "y": orbit_radius * math.sin(theta),
        "z": eye_z_low + (eye_z_high - eye_z_low) * s,
    }
    center = {
        "x": center_xy_amp * math.cos(theta + math.pi / 6),
        "y": center_xy_amp * math.sin(theta + math.pi / 6),
        "z": -center_z_amp + 2 * center_z_amp * s,
    }
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_topdown_spiral(
    t: float,
    turns: float = 1.75,
    r_far: float = 2.7,
    r_near: float = 1.4,
    eye_z_far: float = 1.8,
    eye_z_near: float = 0.55,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Starts above & far, spirals inward and down toward the core, then returns."""
    s = ease_in_out_cycle(t)
    theta = 2 * math.pi * turns * t
    r = r_far + (r_near - r_far) * s
    ez = eye_z_far + (eye_z_near - eye_z_far) * s
    eye = {"x": r * math.cos(theta), "y": r * math.sin(theta), "z": ez}
    center = {"x": 0.0, "y": 0.0, "z": -center_z_amp * math.cos(2.0 * math.pi * t)}
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_dolly_through(
    t: float,
    axis: str = "y",          # "x", "y", or "diag"
    far: float = 3.0,
    near: float = 0.35,
    side_sway: float = 0.25,
    eye_z: float = 0.9,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Line-like fly-through: far -> near -> far with slight sway for parallax."""
    s = ease_in_out_cycle(t)
    d = far + (near - far) * s
    sway = side_sway * math.sin(2.0 * math.pi * t)

    if axis == "y":
        eye = {"x": sway, "y": d, "z": eye_z}
    elif axis == "x":
        eye = {"x": d, "y": sway, "z": eye_z}
    else:  # "diag"
        eye = {"x": d / math.sqrt(2), "y": d / math.sqrt(2), "z": eye_z}

    center = {"x": 0.0, "y": 0.0, "z": -center_z_amp * math.cos(2.0 * math.pi * t)}
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_figure8(
    t: float,
    amp_x: float = 2.2,
    amp_y: float = 2.2,
    eye_z_base: float = 0.95,
    eye_z_amp: float = 0.25,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Figure-8 sweep (Lissajous-ish) that looks great in outreach loops."""
    x = amp_x * math.sin(2.0 * math.pi * t)
    y = amp_y * math.sin(4.0 * math.pi * t) / 2.0
    z = eye_z_base + eye_z_amp * math.sin(2.0 * math.pi * t + math.pi / 4)
    eye = {"x": x, "y": y, "z": z}
    center = {"x": 0.0, "y": 0.0, "z": -center_z_amp * math.cos(2.0 * math.pi * t)}
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_corkscrew_dollyzoom(
    t: float,
    turns: float = 0.9,
    r_mid: float = 2.2,
    r_amp: float = 0.65,
    eye_z_base: float = 0.95,
    eye_z_amp: float = 0.30,
    center_xy_amp: float = 0.08,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Orbit while zooming strongly in/out for dramatic depth."""
    theta = 2.0 * math.pi * turns * t
    r = r_mid + r_amp * math.sin(2.0 * math.pi * t)
    eye = {
        "x": r * math.cos(theta),
        "y": r * math.sin(theta),
        "z": eye_z_base + eye_z_amp * math.cos(2.0 * math.pi * t),
    }
    center = {
        "x": center_xy_amp * math.cos(theta + math.pi / 2),
        "y": center_xy_amp * math.sin(theta + math.pi / 2),
        "z": -center_z_amp * math.cos(2.0 * math.pi * t),
    }
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_under_over(
    t: float,
    turns: float = 1.0,
    orbit_radius: float = 2.2,
    eye_z_base: float = 0.8,
    flip_depth: float = 1.2,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Orbit with camera going above and below the scene plane (reduce flip_depth if too intense)."""
    theta = 2.0 * math.pi * turns * t
    ez = eye_z_base + flip_depth * math.sin(2.0 * math.pi * t)
    eye = {"x": orbit_radius * math.cos(theta), "y": orbit_radius * math.sin(theta), "z": ez}
    center = {"x": 0.0, "y": 0.0, "z": -center_z_amp * math.cos(2.0 * math.pi * t)}
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


def traj_sector_tour(
    t: float,
    turns: float = 1.0,
    orbit_radius: float = 2.25,
    eye_z_base: float = 0.95,
    eye_z_amp: float = 0.22,
    slow_factor: float = 0.35,
    center_z_amp: float = 0.55,
) -> CameraDict:
    """Gentle orbit that eases/lingers at certain angles (nice for readability)."""
    theta = 2.0 * math.pi * turns * (t + slow_factor * math.sin(2.0 * math.pi * t) / (2.0 * math.pi))
    eye = {
        "x": orbit_radius * math.cos(theta),
        "y": orbit_radius * math.sin(theta),
        "z": eye_z_base + eye_z_amp * math.sin(2.0 * math.pi * t),
    }
    center = {"x": 0.0, "y": 0.0, "z": -center_z_amp * math.cos(2.0 * math.pi * t)}
    up = {"x": 0, "y": 0, "z": 1}
    return {"eye": eye, "center": center, "up": up}


TRAJECTORIES: Dict[str, Dict[str, Any]] = {
    # The original default-ish one
    "orbit_time": {
        "fn": traj_orbit_time,
        "defaults": dict(turns=1.0, orbit_radius=2.1, zoom_amplitude=0.12,
                         eye_z_base=0.95, eye_z_amplitude=0.35,
                         center_z_amplitude=0.55, center_xy_amplitude=0.06),
        "desc": "Orbit + subtle zoom, center tours bottom->top->bottom (smooth default).",
    },
    "helix_time": {
        "fn": traj_helix_time,
        "defaults": dict(turns=1.25, orbit_radius=2.2, eye_z_low=0.55, eye_z_high=1.25,
                         center_z_amp=0.65, center_xy_amp=0.06),
        "desc": "Helical orbit with camera height rise/fall; center tours bottom->top->bottom.",
    },
    "topdown_spiral": {
        "fn": traj_topdown_spiral,
        "defaults": dict(turns=1.75, r_far=2.7, r_near=1.4, eye_z_far=1.8, eye_z_near=0.55,
                         center_z_amp=0.55),
        "desc": "Top-down spiral: far/above -> closer/down -> return (very 'fly-in' feel).",
    },
    "dolly_through": {
        "fn": traj_dolly_through,
        "defaults": dict(axis="y", far=3.0, near=0.35, side_sway=0.25, eye_z=0.9, center_z_amp=0.55),
        "desc": "Dolly pass: far -> near -> far along an axis with parallax sway.",
    },
    "figure8": {
        "fn": traj_figure8,
        "defaults": dict(amp_x=2.2, amp_y=2.2, eye_z_base=0.95, eye_z_amp=0.25, center_z_amp=0.55),
        "desc": "Figure-8 sweep (outreach-friendly loop).",
    },
    "corkscrew_dollyzoom": {
        "fn": traj_corkscrew_dollyzoom,
        "defaults": dict(turns=0.9, r_mid=2.2, r_amp=0.65, eye_z_base=0.95, eye_z_amp=0.30,
                         center_xy_amp=0.08, center_z_amp=0.55),
        "desc": "Corkscrew orbit with strong in/out zoom for dramatic depth.",
    },
    "under_over": {
        "fn": traj_under_over,
        "defaults": dict(turns=1.0, orbit_radius=2.2, eye_z_base=0.8, flip_depth=1.2, center_z_amp=0.55),
        "desc": "Under-and-over orbit: camera goes above/below plane (reduce flip_depth if intense).",
    },
    "sector_tour": {
        "fn": traj_sector_tour,
        "defaults": dict(turns=1.0, orbit_radius=2.25, eye_z_base=0.95, eye_z_amp=0.22,
                         slow_factor=0.35, center_z_amp=0.55),
        "desc": "Slow/lingering orbit (good for readability).",
    },
}


def get_trajectory(name: str, overrides: Optional[Dict[str, Any]] = None) -> Callable[[float], CameraDict]:
    if name not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory '{name}'. Choices: {sorted(TRAJECTORIES.keys())}")
    spec = TRAJECTORIES[name]
    fn: TrajectoryFn = spec["fn"]
    params = dict(spec["defaults"])
    if overrides:
        params.update(overrides)

    def cam(t: float) -> CameraDict:
        return fn(t, **params)

    return cam


# =========================
# Visualizer (your code + exporter)
# =========================

class UnifiedTemporalNebula:
    def __init__(self, db_path: Path, config: Optional[UnifiedNebulaConfig] = None):
        self.db_path = Path(db_path)
        self.config = config or UnifiedNebulaConfig()
        self.log = get_logger(self.__class__.__name__)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Inner
        self.profile_label: str = "Personal Profile"
        self.date_range: Tuple[str, str] = ("", "")
        self.windows: List[Dict[str, Any]] = []
        self.inner_categories: List[str] = []
        self.topic_particles: List[TopicParticle] = []
        self.clusters: List[ClusterCloud] = []

        # Outer
        self.entities: List[EntityParticle] = []
        self.entity_types: List[str] = []
        self.edges: List[Tuple[str, str, float]] = []

        # Time bounds for outer
        self.min_time: Optional[datetime] = None
        self.max_time: Optional[datetime] = None
        self.min_timestamp: float = 0.0
        self.max_timestamp: float = 1.0

        np.random.seed(self.config.seed)

    def close(self):
        self.conn.close()

    @staticmethod
    def _parse_timestamp(ts_str: Optional[str]) -> float:
        if not ts_str:
            return 0.0
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return float(dt.timestamp())
        except (ValueError, TypeError):
            return 0.0

    def _get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        node = dict(row)
        if node.get("metadata_json"):
            node["metadata"] = json.loads(node["metadata_json"])
        else:
            node["metadata"] = {}
        return node

    def _get_edges_from(self, src_id: str, edge_type: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM graph_edges WHERE src_node_id = ? AND edge_type = ?",
            (src_id, edge_type),
        ).fetchall()
        edges: List[Dict[str, Any]] = []
        for row in rows:
            edge = dict(row)
            if edge.get("metadata_json"):
                edge["metadata"] = json.loads(edge["metadata_json"])
            else:
                edge["metadata"] = {}
            edges.append(edge)
        return edges

    # -------- Inner load --------

    def _find_profile(self) -> Optional[str]:
        profile = self.conn.execute(
            "SELECT node_id FROM graph_nodes WHERE node_type = 'TemporalProfile' LIMIT 1"
        ).fetchone()
        if profile:
            return profile["node_id"]

        edge = self.conn.execute(
            "SELECT dst_node_id FROM graph_edges WHERE edge_type = 'HAS_PROFILE' LIMIT 1"
        ).fetchone()
        if edge:
            return edge["dst_node_id"]
        return None

    def _load_inner_zone_data(self) -> bool:
        self.log.info("Loading inner zone data (topics/clusters)...")

        profile_id = self._find_profile()
        if not profile_id:
            self.log.warning("No temporal profile found - inner zone will be empty")
            return False

        profile_node = self._get_node(profile_id)
        if not profile_node:
            return False

        meta = profile_node.get("metadata", {})
        self.profile_label = profile_node.get("label", "Profile")
        self.date_range = tuple(meta.get("date_range", ["", ""]))

        window_edges = self._get_edges_from(profile_id, "HAS_WINDOW")
        window_edges.sort(key=lambda e: e.get("metadata", {}).get("window_index", 0))

        categories_seen: Set[str] = set()

        for we in window_edges[: self.config.max_windows]:
            window_node = self._get_node(we["dst_node_id"])
            if not window_node:
                continue

            wm = window_node.get("metadata", {})
            window_idx = we.get("metadata", {}).get("window_index", len(self.windows))

            window_data = {
                "id": window_node["node_id"],
                "label": window_node.get("label", ""),
                "index": window_idx,
                "start_utc": wm.get("window_start_utc", ""),
                "end_utc": wm.get("window_end_utc", ""),
                "clusters": [],
            }

            if window_data["start_utc"]:
                try:
                    dt = datetime.fromisoformat(window_data["start_utc"].replace("Z", "+00:00"))
                    self.min_time = dt if self.min_time is None or dt < self.min_time else self.min_time
                    self.max_time = dt if self.max_time is None or dt > self.max_time else self.max_time
                except ValueError:
                    pass

            cluster_edges = self._get_edges_from(window_node["node_id"], "HAS_CLUSTER")

            for ce in cluster_edges:
                cluster_node = self._get_node(ce["dst_node_id"])
                if not cluster_node:
                    continue

                cm = cluster_node.get("metadata", {})
                category = cm.get("category_key", "unclassified")
                categories_seen.add(category)

                cluster_data = {
                    "id": cluster_node["node_id"],
                    "category": category,
                    "member_count": cm.get("member_count", 0),
                    "avg_salience": cm.get("avg_salience", 0.0),
                    "topics": [],
                }

                member_edges = self._get_edges_from(cluster_node["node_id"], "CLUSTER_CONTAINS")
                member_edges.sort(key=lambda e: e.get("metadata", {}).get("rank", 999))

                for me in member_edges[: self.config.inner_max_topics_per_cluster]:
                    em = me.get("metadata", {})
                    salience = em.get("salience", 0.0)
                    if salience < self.config.inner_min_salience:
                        continue

                    topic_node = self._get_node(me["dst_node_id"])
                    if not topic_node:
                        continue

                    tm = topic_node.get("metadata", {})
                    cluster_data["topics"].append(
                        {
                            "id": topic_node["node_id"],
                            "label": topic_node.get("label", "")[:50],
                            "salience": salience,
                            "assertions": tm.get("assertion_count", tm.get("mention_count", 1)),
                        }
                    )

                if cluster_data["topics"]:
                    window_data["clusters"].append(cluster_data)

            self.windows.append(window_data)

        self.inner_categories = sorted(categories_seen)
        total_topics = sum(len(c["topics"]) for w in self.windows for c in w["clusters"])
        self.log.info(f"  Loaded {len(self.windows)} windows, {total_topics} topics")
        return total_topics > 0

    # -------- Outer load --------

    def _load_outer_zone_data(self) -> bool:
        self.log.info("Loading outer zone data (entities/relationships)...")

        if self.config.time_filter:
            query = """
                SELECT
                    e.entity_id,
                    e.entity_type,
                    e.canonical_name,
                    e.first_seen_at_utc,
                    e.mention_count
                FROM entities e
                WHERE e.status = 'active'
                  AND e.first_seen_at_utc >= ?
                  AND e.first_seen_at_utc < ?
                ORDER BY e.first_seen_at_utc
                LIMIT ?
            """
            rows = self.conn.execute(
                query, (self.config.time_start, self.config.time_end, self.config.outer_max_entities)
            ).fetchall()
        else:
            query = """
                SELECT
                    e.entity_id,
                    e.entity_type,
                    e.canonical_name,
                    e.first_seen_at_utc,
                    e.mention_count
                FROM entities e
                WHERE e.status = 'active'
                ORDER BY e.first_seen_at_utc
                LIMIT ?
            """
            rows = self.conn.execute(query, (self.config.outer_max_entities,)).fetchall()

        entities_raw = [dict(r) for r in rows]
        entity_types_seen: Set[str] = set()

        for e in entities_raw:
            entity_type = str(e.get("entity_type") or "OTHER")
            entity_types_seen.add(entity_type)

            ts = self._parse_timestamp(e.get("first_seen_at_utc"))
            if self.min_timestamp == 0.0 or ts < self.min_timestamp:
                self.min_timestamp = ts
            if ts > self.max_timestamp:
                self.max_timestamp = ts

            self.entities.append(
                EntityParticle(
                    entity_id=str(e["entity_id"]),
                    entity_type=entity_type,
                    canonical_name=str(e.get("canonical_name") or ""),
                    mention_count=int(e.get("mention_count") or 0),
                    first_seen_at=str(e.get("first_seen_at_utc") or ""),
                    color=OUTER_COLORS.get(entity_type, "#FFFFFF"),
                )
            )

        self.entity_types = sorted(entity_types_seen)

        edge_query = """
            SELECT
                a.subject_entity_id,
                a.object_entity_id,
                a.confidence_final
            FROM assertions a
            JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
            WHERE a.object_entity_id IS NOT NULL
              AND at.status = 'active'
              AND a.confidence_final > ?
        """
        edge_rows = self.conn.execute(edge_query, (self.config.outer_min_confidence,)).fetchall()
        edges_raw = [(str(r[0]), str(r[1]), float(r[2])) for r in edge_rows]

        if len(edges_raw) > 5000 and 0.0 < self.config.outer_edge_sample_rate < 1.0:
            sample_size = max(1, int(len(edges_raw) * self.config.outer_edge_sample_rate))
            idx = np.random.choice(len(edges_raw), sample_size, replace=False)
            self.edges = [edges_raw[i] for i in idx]
        else:
            self.edges = edges_raw

        self.log.info(f"  Loaded {len(self.entities)} entities, {len(self.edges)} edges")
        return len(self.entities) > 0

    # -------- Layout compute --------

    def _compute_inner_zone_positions(self):
        self.log.info("Computing inner zone layout...")
        if not self.windows or not self.inner_categories:
            return

        num_windows = len(self.windows)
        num_categories = len(self.inner_categories)

        angle_per_category = (2 * math.pi) / max(num_categories, 1)
        category_angles = {cat: i * angle_per_category for i, cat in enumerate(self.inner_categories)}
        z_per_window = self.config.z_span / max(num_windows, 1)

        for window in self.windows:
            window_idx = window["index"]
            window_label = window["label"]
            base_z = window_idx * z_per_window

            for cluster in window["clusters"]:
                category = cluster["category"]
                base_angle = category_angles.get(category, 0.0)

                cluster_r = (self.config.inner_zone_min + self.config.inner_zone_max) / 2
                cloud = ClusterCloud(
                    cluster_id=cluster["id"],
                    category=category,
                    window_index=window_idx,
                    center_x=cluster_r * math.cos(base_angle),
                    center_y=cluster_r * math.sin(base_angle),
                    center_z=base_z + z_per_window / 2,
                )

                for topic in cluster["topics"]:
                    salience = float(topic["salience"])
                    base_radius = self.config.inner_zone_max - salience * (self.config.inner_zone_max - self.config.inner_zone_min)
                    r = float(np.clip(base_radius + np.random.normal(0, 0.08),
                                      self.config.inner_zone_min, self.config.inner_zone_max))
                    angle = base_angle + float(np.random.normal(0, angle_per_category / 8))
                    z = base_z + float(np.random.uniform(0, z_per_window) + np.random.normal(0, 0.2))

                    x = r * math.cos(angle)
                    y = r * math.sin(angle)

                    size = self.config.inner_particle_min + salience * (self.config.inner_particle_max - self.config.inner_particle_min)
                    color = INNER_COLORS.get(category, "#FFFFFF")

                    particle = TopicParticle(
                        topic_id=topic["id"],
                        label=topic["label"],
                        category=category,
                        salience=salience,
                        assertion_count=int(topic["assertions"]),
                        window_label=window_label,
                        window_index=int(window_idx),
                        x=float(x), y=float(y), z=float(z),
                        size=float(size),
                        color=color,
                    )
                    self.topic_particles.append(particle)
                    cloud.particles.append(particle)

                self.clusters.append(cloud)

        self.log.info(f"  Positioned {len(self.topic_particles)} topic particles")

    def _compute_outer_zone_positions(self):
        self.log.info("Computing outer zone layout...")
        if not self.entities or not self.entity_types:
            return

        time_range = self.max_timestamp - self.min_timestamp
        if time_range <= 0:
            time_range = 1.0
        z_scale = self.config.z_span / time_range

        num_types = len(self.entity_types)
        angle_per_type = (2 * math.pi) / max(num_types, 1)
        type_angles = {t: i * angle_per_type for i, t in enumerate(self.entity_types)}

        for entity in self.entities:
            t = self._parse_timestamp(entity.first_seen_at)
            z = (t - self.min_timestamp) * z_scale

            base_theta = type_angles.get(entity.entity_type, 0.0)
            theta = base_theta + float(np.random.normal(0, angle_per_type / 10))

            mentions = float(entity.mention_count)
            base_radius = self.config.outer_zone_max - (mentions / 100.0) * (self.config.outer_zone_max - self.config.outer_zone_min)
            base_radius = float(np.clip(base_radius, self.config.outer_zone_min, self.config.outer_zone_max))
            r = base_radius + float(np.random.normal(0, 0.1))

            entity.x = r * float(np.cos(theta)) + float(np.random.normal(0, 0.05))
            entity.y = r * float(np.sin(theta)) + float(np.random.normal(0, 0.05))
            entity.z = float(z + np.random.normal(0, 0.02 * z_scale))

            entity.size = self.config.outer_particle_min + min(1.0, mentions / 50.0) * (self.config.outer_particle_max - self.config.outer_particle_min)

        self.log.info(f"  Positioned {len(self.entities)} entity particles")

    def compute_all_positions(self):
        self._compute_inner_zone_positions()
        self._compute_outer_zone_positions()

    # -------- Figure building --------

    def _add_self_marker(self, fig: go.Figure):
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                name="⬤ SELF",
                marker=dict(
                    size=self.config.self_size,
                    color=self.config.self_color,
                    symbol="diamond",
                    line=dict(width=3, color="#60A5FA"),
                    opacity=1.0,
                ),
                text=["SELF"],
                textposition="middle center",
                textfont=dict(size=12, color="#1E293B", family="Arial Black"),
                hoverinfo="name",
                showlegend=True,
                legendgroup="infrastructure",
                legendgrouptitle=dict(text="Core"),
            )
        )

    def _add_time_spine(self, fig: go.Figure):
        if not self.config.show_time_spine:
            return
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, self.config.z_span],
                mode="lines",
                name="Time Axis",
                line=dict(color="rgba(96, 165, 250, 0.25)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def _add_time_rings(self, fig: go.Figure):
        if not self.config.show_time_rings or not self.windows:
            return
        num_windows = len(self.windows)
        z_per_window = self.config.z_span / max(num_windows, 1)
        theta = np.linspace(0, 2 * np.pi, 80)
        step = max(1, num_windows // 8)

        for i in range(0, num_windows, step):
            z = i * z_per_window + z_per_window / 2
            r_inner = self.config.inner_zone_max
            x_inner = r_inner * np.cos(theta)
            y_inner = r_inner * np.sin(theta)
            z_inner = np.full_like(theta, z)

            fig.add_trace(
                go.Scatter3d(
                    x=x_inner, y=y_inner, z=z_inner,
                    mode="lines",
                    line=dict(color="rgba(100, 116, 139, 0.1)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def _add_zone_boundaries(self, fig: go.Figure):
        if not self.config.show_zone_boundaries:
            return
        theta = np.linspace(0, 2 * np.pi, 120)
        z_mid = self.config.z_span / 2

        r_inner = self.config.inner_zone_max
        fig.add_trace(
            go.Scatter3d(
                x=r_inner * np.cos(theta),
                y=r_inner * np.sin(theta),
                z=np.full_like(theta, z_mid),
                mode="lines",
                name="Inner Zone",
                line=dict(color="rgba(59, 130, 246, 0.2)", width=2, dash="dot"),
                hovertext="Personal Topics Zone",
                hoverinfo="text",
                showlegend=False,
            )
        )

        r_outer = self.config.outer_zone_min
        fig.add_trace(
            go.Scatter3d(
                x=r_outer * np.cos(theta),
                y=r_outer * np.sin(theta),
                z=np.full_like(theta, z_mid),
                mode="lines",
                name="Outer Zone",
                line=dict(color="rgba(16, 185, 129, 0.2)", width=2, dash="dot"),
                hovertext="Knowledge Graph Zone",
                hoverinfo="text",
                showlegend=False,
            )
        )

    def _add_inner_zone_traces(self, fig: go.Figure):
        self.log.info("Building inner zone traces...")
        for category in self.inner_categories:
            cat_particles = [p for p in self.topic_particles if p.category == category]
            if not cat_particles:
                continue

            color = INNER_COLORS.get(category, "#FFFFFF")
            color_light = INNER_COLORS_LIGHT.get(category, color)
            icon = CATEGORY_ICONS.get(category, "•")

            fig.add_trace(
                go.Scatter3d(
                    x=[p.x for p in cat_particles],
                    y=[p.y for p in cat_particles],
                    z=[p.z for p in cat_particles],
                    mode="markers",
                    name=f"{icon} {category.title()}",
                    marker=dict(
                        size=[p.size for p in cat_particles],
                        color=color,
                        opacity=self.config.inner_opacity,
                        line=dict(width=1.5, color=color_light),
                        symbol="circle",
                    ),
                    text=[
                        f"<b>{p.label}</b><br>"
                        f"Category: {icon} {p.category.title()}<br>"
                        f"Period: {p.window_label}<br>"
                        f"Salience: {p.salience:.1%}<br>"
                        f"Assertions: {p.assertion_count}"
                        for p in cat_particles
                    ],
                    hoverinfo="text",
                    showlegend=True,
                    legendgroup="inner",
                    legendgrouptitle=dict(text="Personal Topics"),
                )
            )

    def _add_cluster_connections(self, fig: go.Figure):
        if not self.config.show_cluster_connections:
            return
        self.log.info("Building cluster connections...")
        for cloud in self.clusters:
            if len(cloud.particles) < 2:
                continue

            color = INNER_COLORS.get(cloud.category, "#FFFFFF")
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba = f"rgba({r}, {g}, {b}, {self.config.cluster_connection_opacity})"

            edge_x: List[Optional[float]] = []
            edge_y: List[Optional[float]] = []
            edge_z: List[Optional[float]] = []

            for p in cloud.particles[:8]:
                edge_x.extend([cloud.center_x, p.x, None])
                edge_y.extend([cloud.center_y, p.y, None])
                edge_z.extend([cloud.center_z, p.z, None])

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode="lines",
                    line=dict(color=rgba, width=1.0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def _add_outer_zone_traces(self, fig: go.Figure):
        self.log.info("Building outer zone traces...")
        for entity_type in self.entity_types:
            type_entities = [e for e in self.entities if e.entity_type == entity_type]
            if not type_entities:
                continue

            color = OUTER_COLORS.get(entity_type, "#FFFFFF")
            fig.add_trace(
                go.Scatter3d(
                    x=[e.x for e in type_entities],
                    y=[e.y for e in type_entities],
                    z=[e.z for e in type_entities],
                    mode="markers",
                    name=f"▪ {entity_type}",
                    marker=dict(
                        size=[e.size for e in type_entities],
                        color=color,
                        opacity=self.config.outer_opacity,
                        line=dict(width=0.5, color=color),
                        symbol="circle",
                    ),
                    text=[
                        f"<b>{e.canonical_name[:40]}</b><br>"
                        f"Type: {e.entity_type}<br>"
                        f"First Seen: {e.first_seen_at[:10]}<br>"
                        f"Mentions: {e.mention_count}"
                        for e in type_entities
                    ],
                    hoverinfo="text",
                    showlegend=True,
                    legendgroup="outer",
                    legendgrouptitle=dict(text="Knowledge Graph"),
                )
            )

    def _add_relationship_edges(self, fig: go.Figure):
        self.log.info("Building relationship edges...")
        entity_lookup = {e.entity_id: e for e in self.entities}

        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        edge_z: List[Optional[float]] = []

        for src_id, dst_id, _conf in self.edges:
            src = entity_lookup.get(src_id)
            dst = entity_lookup.get(dst_id)
            if not src or not dst:
                continue
            edge_x.extend([src.x, dst.x, None])
            edge_y.extend([src.y, dst.y, None])
            edge_z.extend([src.z, dst.z, None])

        rgba_color = f"rgba(150, 150, 200, {self.config.outer_edge_opacity})"
        fig.add_trace(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                name="Connections",
                line=dict(color=rgba_color, width=self.config.outer_edge_width),
                hoverinfo="skip",
                showlegend=True,
                legendgroup="outer",
            )
        )

    def _apply_layout(self, fig: go.Figure):
        start_date = self.date_range[0][:10] if self.date_range[0] else "?"
        end_date = self.date_range[1][:10] if self.date_range[1] else "?"

        title_text = (
            f"<b>Personal Temporal Universe</b><br>"
            f"<sub>Inner Core: {len(self.topic_particles)} topics · "
            f"Outer Layer: {len(self.entities)}+ entities · "
            f"{start_date} → {end_date}</sub>"
        )

        if self.windows:
            num_windows = len(self.windows)
            z_per_window = self.config.z_span / max(num_windows, 1)
            step = max(1, num_windows // 6)
            tick_vals = []
            tick_labels = []
            for i in range(0, num_windows, step):
                z = i * z_per_window + z_per_window / 2
                tick_vals.append(z)
                tick_labels.append(self.windows[i]["label"][:10])
        else:
            tick_vals = [0, self.config.z_span]
            tick_labels = ["Start", "End"]

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=22, color=self.config.text_color), x=0.5, xanchor="center"),
            scene=dict(
                xaxis=dict(
                    title="", showgrid=True, gridcolor=self.config.grid_color,
                    showbackground=True, backgroundcolor=self.config.background_color,
                    zerolinecolor=self.config.grid_color, showticklabels=False, showspikes=False
                ),
                yaxis=dict(
                    title="", showgrid=True, gridcolor=self.config.grid_color,
                    showbackground=True, backgroundcolor=self.config.background_color,
                    zerolinecolor=self.config.grid_color, showticklabels=False, showspikes=False
                ),
                zaxis=dict(
                    title=dict(text="Time →", font=dict(color=self.config.text_color)),
                    showgrid=True, gridcolor=self.config.grid_color,
                    showbackground=True, backgroundcolor=self.config.background_color,
                    zerolinecolor=self.config.grid_color,
                    tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
                    tickfont=dict(size=10, color=self.config.text_color),
                    visible=False,
                ),
                bgcolor=self.config.background_color,
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.0), up=dict(x=0, y=0, z=1)),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1.5),
            ),
            paper_bgcolor=self.config.background_color,
            plot_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor=self.config.grid_color,
                borderwidth=1,
                font=dict(size=11),
                itemsizing="constant",
                grouptitlefont=dict(size=13, family="Arial", color="#60A5FA"),
                tracegroupgap=10,
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            hovermode="closest",
        )

    # -------- Public API --------

    def build_figure(self) -> go.Figure:
        self.log.info("\n" + "=" * 70)
        self.log.info("UNIFIED TEMPORAL UNIVERSE - Building Visualization")
        self.log.info("=" * 70)

        inner_loaded = self._load_inner_zone_data()
        outer_loaded = self._load_outer_zone_data()
        if not inner_loaded and not outer_loaded:
            raise ValueError("No data loaded for either zone!")

        self.compute_all_positions()

        fig = go.Figure()
        self._add_time_spine(fig)
        self._add_time_rings(fig)
        self._add_zone_boundaries(fig)

        self._add_relationship_edges(fig)
        self._add_cluster_connections(fig)
        self._add_outer_zone_traces(fig)
        self._add_inner_zone_traces(fig)
        self._add_self_marker(fig)

        self._apply_layout(fig)
        self.log.info("Figure complete!")
        return fig

    def visualize(self, output_path: Path, show: bool = False) -> bool:
        try:
            fig = self.build_figure()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
            self.log.info(f"\nVisualization saved to: {output_path}")
            self._print_statistics()
            if show:
                fig.show()
            return True
        except Exception as e:
            self.log.error(f"Visualization failed: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return False

    def export_flythrough_frames(
        self,
        frames_dir: Path,
        *,
        n_frames: int = 240,
        closed_loop: bool = True,
        width: int = 1920,
        height: int = 1080,
        scale: float = 2.0,
        hide_legend: bool = True,
        trajectory: str = "orbit_time",
        traj_kwargs: Optional[Dict[str, Any]] = None,
        file_prefix: str = "frame_",
        digits: int = 5,
    ) -> Path:
        self.log.info("\n" + "=" * 70)
        self.log.info("FLY-THROUGH EXPORT - Rendering PNG frames")
        self.log.info("=" * 70)

        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")

        # Ensure kaleido exists (Plotly uses it for static export)
        try:
            import kaleido  # noqa: F401
        except Exception as e:
            raise RuntimeError("Static PNG export requires 'kaleido'. Install: pip install kaleido") from e

        cam_fn = get_trajectory(trajectory, traj_kwargs)

        fig = self.build_figure()
        if hide_legend:
            fig.update_layout(showlegend=False)

        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        denom = (n_frames - 1) if (closed_loop and n_frames > 1) else n_frames

        for i in range(n_frames):
            t = 0.0 if denom == 0 else (i / denom)
            fig.update_layout(scene_camera=cam_fn(t))

            outpath = frames_dir / f"{file_prefix}{i:0{digits}d}.png"
            fig.write_image(str(outpath), width=width, height=height, scale=scale, engine="kaleido")

            if (i == 0) or ((i + 1) % max(1, n_frames // 10) == 0) or (i == n_frames - 1):
                self.log.info(f"  Rendered {i+1}/{n_frames}: {outpath.name}")

        self.log.info(f"\nFrames written to: {frames_dir.resolve()}")
        self.log.info("=" * 70 + "\n")
        return frames_dir

    def _print_statistics(self):
        self.log.info("\n" + "=" * 70)
        self.log.info("VISUALIZATION STATISTICS")
        self.log.info("=" * 70)

        if self.profile_label:
            self.log.info(f"Profile: {self.profile_label}")
        if self.date_range[0]:
            self.log.info(f"Time Range: {self.date_range[0][:10]} to {self.date_range[1][:10]}")

        self.log.info("\nINNER ZONE (Personal Topics):")
        self.log.info(f"  Time Windows: {len(self.windows)}")
        self.log.info(f"  Semantic Clusters: {len(self.clusters)}")
        self.log.info(f"  Topic Particles: {len(self.topic_particles)}")

        if self.topic_particles:
            cat_counts = defaultdict(int)
            for p in self.topic_particles:
                cat_counts[p.category] += 1
            self.log.info("  By Category:")
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                icon = CATEGORY_ICONS.get(cat, "•")
                pct = 100.0 * count / len(self.topic_particles)
                self.log.info(f"    {icon} {cat}: {count} ({pct:.1f}%)")

        self.log.info("\nOUTER ZONE (Knowledge Graph):")
        self.log.info(f"  Entity Particles: {len(self.entities)}")
        self.log.info(f"  Relationship Edges: {len(self.edges)}")

        if self.entities:
            type_counts = defaultdict(int)
            for e in self.entities:
                type_counts[e.entity_type] += 1
            self.log.info("  By Entity Type:")
            for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
                pct = 100.0 * count / len(self.entities)
                self.log.info(f"    ▪ {etype}: {count} ({pct:.1f}%)")

        self.log.info("\nSPATIAL LAYOUT:")
        self.log.info(f"  Inner Zone Radius: {self.config.inner_zone_min:.2f} - {self.config.inner_zone_max:.2f}")
        self.log.info(f"  Outer Zone Radius: {self.config.outer_zone_min:.2f} - {self.config.outer_zone_max:.2f}")
        self.log.info(f"  Time Span (Z-axis): 0 - {self.config.z_span:.1f}")
        self.log.info("=" * 70 + "\n")


# =========================
# CLI
# =========================

def run(
    db: Path,
    output_html: Path,
    *,
    # Inner
    max_windows: int = 36,
    max_topics_per_cluster: int = 20,
    min_salience: float = 0.0,
    show_cluster_connections: bool = True,
    # Outer
    max_entities: int = 10_000,
    edge_sample_rate: float = 0.08,
    min_confidence: float = 0.3,
    # Shared
    z_span: float = 15.0,
    time_filter: bool = True,
    time_start: str = "2023-01-01T00:00:00Z",
    time_end: str = "2026-01-01T00:00:00Z",
    # Visual
    show_zone_boundaries: bool = True,
    show_time_rings: bool = True,
    seed: int = 42,
    show: bool = False,
    # Frames export
    frames_dir: Optional[Path] = None,
    n_frames: int = 240,
    closed_loop: bool = True,
    frame_width: int = 1920,
    frame_height: int = 1080,
    frame_scale: float = 2.0,
    hide_legend: bool = True,
    trajectory: str = "orbit_time",
    traj_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    config = UnifiedNebulaConfig(
        inner_max_topics_per_cluster=max_topics_per_cluster,
        inner_min_salience=min_salience,
        show_cluster_connections=show_cluster_connections,
        outer_max_entities=max_entities,
        outer_edge_sample_rate=edge_sample_rate,
        outer_min_confidence=min_confidence,
        max_windows=max_windows,
        z_span=z_span,
        time_filter=time_filter,
        time_start=time_start,
        time_end=time_end,
        show_zone_boundaries=show_zone_boundaries,
        show_time_rings=show_time_rings,
        seed=seed,
    )

    viz = UnifiedTemporalNebula(db, config)
    try:
        ok = viz.visualize(output_html, show=show)
        if frames_dir is not None:
            viz.export_flythrough_frames(
                frames_dir,
                n_frames=n_frames,
                closed_loop=closed_loop,
                width=frame_width,
                height=frame_height,
                scale=frame_scale,
                hide_legend=hide_legend,
                trajectory=trajectory,
                traj_kwargs=traj_kwargs,
            )
        return ok
    finally:
        viz.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Unified Temporal Universe + loopable fly-through PNG export (multi-trajectory)"
    )

    parser.add_argument("--db", type=Path, default=Path("../data/output/kg.db"))
    parser.add_argument("--output", type=Path, default=Path("../data/figures/unified_temporal_universe.html"))

    # Inner
    parser.add_argument("--max-windows", type=int, default=36)
    parser.add_argument("--max-topics", type=int, default=20)
    parser.add_argument("--min-salience", type=float, default=0.0)
    parser.add_argument("--no-cluster-connections", action="store_true")

    # Outer
    parser.add_argument("--max-entities", type=int, default=10_000)
    parser.add_argument("--edge-sample-rate", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.3)

    # Shared
    parser.add_argument("--z-span", type=float, default=15.0)
    parser.add_argument("--time-start", type=str, default="2023-01-01T00:00:00Z")
    parser.add_argument("--time-end", type=str, default="2026-01-01T00:00:00Z")
    parser.add_argument("--no-time-filter", dest="time_filter", action="store_false")
    parser.set_defaults(time_filter=True)

    # Visual
    parser.add_argument("--no-zone-boundaries", action="store_true")
    parser.add_argument("--no-time-rings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")

    # Frames export
    parser.add_argument("--frames-dir", type=Path, default=Path("../data/snapshots"), help="If set, export PNG frames here.")
    parser.add_argument("--n-frames", type=int, default=240)
    parser.add_argument("--open-loop", action="store_true", help="Do NOT duplicate last frame (no identical end frame).")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--frame-scale", type=float, default=2.0)
    parser.add_argument("--keep-legend", action="store_true", help="Keep legend visible in frames (default hides).")

    # Trajectory selection + overrides
    parser.add_argument(
        "--trajectory",
        type=str,
        default="orbit_time",
        choices=sorted(TRAJECTORIES.keys()),
        help="Camera path to use for fly-through frames.",
    )
    parser.add_argument(
        "--traj-kwargs",
        type=str,
        default=None,
        help="JSON dict to override trajectory parameters, e.g. '{\"turns\":1.5,\"orbit_radius\":2.6}'.",
    )
    parser.add_argument(
        "--list-trajectories",
        action="store_true",
        help="Print available trajectories and exit.",
    )

    args = parser.parse_args()

    if args.list_trajectories:
        print("Available trajectories:")
        for name in sorted(TRAJECTORIES.keys()):
            desc = TRAJECTORIES[name]["desc"]
            defaults = TRAJECTORIES[name]["defaults"]
            print(f" - {name}: {desc}")
            print(f"   defaults: {defaults}")
        raise SystemExit(0)

    traj_kwargs = None
    if args.traj_kwargs:
        try:
            traj_kwargs = json.loads(args.traj_kwargs)
            if not isinstance(traj_kwargs, dict):
                raise ValueError("traj-kwargs must be a JSON object/dict")
        except Exception as e:
            raise SystemExit(f"Invalid --traj-kwargs JSON: {e}")

    for traj_name, traj_kwargs in TRAJECTORIES.items():
        logging.info(f"Trajectory {traj_name} ...")
        logging.info(f"--------------------------")

        success = run(
            db=args.db,
            output_html=args.output,
            max_windows=args.max_windows,
            max_topics_per_cluster=args.max_topics,
            min_salience=args.min_salience,
            show_cluster_connections=not args.no_cluster_connections,
            max_entities=args.max_entities,
            edge_sample_rate=args.edge_sample_rate,
            min_confidence=args.min_confidence,
            z_span=args.z_span,
            time_filter=args.time_filter,
            time_start=args.time_start,
            time_end=args.time_end,
            show_zone_boundaries=not args.no_zone_boundaries,
            show_time_rings=not args.no_time_rings,
            seed=args.seed,
            show=args.show,
            frames_dir=args.frames_dir / traj_name,
            n_frames=args.n_frames,
            closed_loop=not args.open_loop,
            frame_width=args.frame_width,
            frame_height=args.frame_height,#
            frame_scale=args.frame_scale,
            hide_legend=not args.keep_legend,
            trajectory=traj_name,
            traj_kwargs=None,
        )
