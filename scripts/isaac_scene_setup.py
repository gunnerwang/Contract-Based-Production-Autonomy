#!/usr/bin/env python3
"""Isaac Sim CBPA production cell scene — headless setup and verification server.

Launches Isaac Sim in headless mode, creates a simple production cell with:
  - Two robot arms (Franka Panda) for assembly tasks
  - A conveyor belt
  - Acoustic and force sensors
  - A human operator placeholder (capsule + fatigue model)

Then runs a lightweight HTTP server (port 8211) so the CBPA dashboard
can connect via the IsaacBridge remote mode.

Usage:
    conda activate isaac
    python scripts/isaac_scene_setup.py              # headless
    python scripts/isaac_scene_setup.py --live-ui     # with viewport
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Accept NVIDIA EULA
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("isaac_cbpa")

# ── Parse args ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CBPA Isaac Sim scene")
parser.add_argument("--live-ui", action="store_true", help="Show viewport UI")
parser.add_argument("--port", type=int, default=8211, help="REST API port")
parser.add_argument("--multi-cell", action="store_true",
                    help="Build two-cell factory layout (Cell A + Cell B + AGV)")
args = parser.parse_args()

# ── Launch Isaac Sim ─────────────────────────────────────────────────
from isaacsim import SimulationApp  # noqa: E402

config = {
    "headless": not args.live_ui,
    "width": 1920,
    "height": 1080,
    "anti_aliasing": 3,          # 0=off, 1=FXAA, 2=DLAA, 3=TAA
    "renderer": "PathTracing",    # RTX Interactive — best visual quality
}
sim_app = SimulationApp(config)

# Now import Omniverse modules (available after SimulationApp init)
import omni.usd  # noqa: E402
try:
    from isaacsim.core.api import World  # noqa: E402
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder  # noqa: E402
except ImportError:
    from omni.isaac.core import World  # noqa: E402
    from omni.isaac.core.objects import VisualCuboid, VisualCylinder  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
import numpy as np  # noqa: E402

# Improve render quality when viewport is visible
if args.live_ui:
    try:
        import carb
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 1)            # 1 sample/pixel/frame (fast convergence)
        settings.set("/rtx/pathtracing/totalSpp", 16)      # low accumulation (less lag)
        settings.set("/rtx/pathtracing/maxBounces", 2)     # fewer bounces = faster
        settings.set("/rtx/pathtracing/clampSpp", True)
        settings.set("/rtx/post/aa/op", 3)                 # TAA to smooth remaining noise
        settings.set("/rtx/post/dlss/execMode", 1)         # DLSS Performance for upscaling
        settings.set("/rtx/ambientOcclusion/enabled", True)
        logger.info("Render quality set to RTX Interactive (Path Tracing, optimized)")
    except Exception as e:
        logger.debug("Could not set render quality: %s", e)

logger.info("Isaac Sim launched successfully")

# ── Build the CBPA production cell scene ─────────────────────────────
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Assembly stations along the X axis:
#   x=-1.2  Entry (spawn)
#   x=-0.6  Station 1: R1 pick-and-place
#   x= 0.0  Station 2: Human insertion + inspection
#   x= 0.6  Station 3: R2 screwdriving
#   x= 1.2  Exit (done)
STATIONS_X = [-1.2, -0.6, 0.0, 0.6, 1.2]
STATION_NAMES = ["Entry", "R1 Pick-Place", "Human Insert", "R2 Screw", "Exit"]

# Resolve Isaac Sim assets root (HTTPS S3 URL — no Nucleus needed)
_ASSETS_ROOT = ""
_S3_ASSETS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
try:
    from isaacsim.core.utils.nucleus import get_assets_root_path
    _ASSETS_ROOT = get_assets_root_path() or ""
    logger.info(f"Isaac assets root: {_ASSETS_ROOT}")
except Exception:
    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path as _g2
        _ASSETS_ROOT = _g2() or ""
    except Exception:
        pass
if not _ASSETS_ROOT:
    _ASSETS_ROOT = _S3_ASSETS_ROOT
    logger.info(f"Nucleus not found — using public S3 asset root: {_ASSETS_ROOT}")

# Asset paths relative to root
_FRANKA_USD = _ASSETS_ROOT + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
_HUMAN_USD  = _ASSETS_ROOT + "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"

# Helper: add a USD reference (with cuboid fallback)
_PREC_D = UsdGeom.XformOp.PrecisionDouble   # match existing attrs in Isaac USD files

def _add_usd_asset(prim_path: str, usd_path: str, position: list, orientation=None, scale=None):
    """Try to load a USD asset from the Isaac assets server; fall back to cuboid."""
    try:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
        except ImportError:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        prim = world.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(precision=_PREC_D).Set(Gf.Vec3d(*position))
            if orientation:
                xform.AddRotateXYZOp(precision=_PREC_D).Set(Gf.Vec3d(*orientation))
            if scale:
                xform.AddScaleOp(precision=_PREC_D).Set(Gf.Vec3d(*scale))
            logger.info(f"Loaded USD asset: {usd_path} -> {prim_path}")
            return True
    except Exception as e:
        logger.warning(f"USD asset load failed ({usd_path}): {e}")
    return False

# ── Conveyor belt assembly ───────────────────────────────────────────
world.scene.add(VisualCuboid(
    prim_path="/World/CBPACell/Conveyor/Belt",
    name="conveyor_belt",
    position=[0.0, 0.0, 0.302],
    scale=[2.96, 0.26, 0.018],
    color=np.array([0.18, 0.18, 0.18]),
))
world.scene.add(VisualCuboid(
    prim_path="/World/CBPACell/Conveyor/Frame",
    name="conveyor_frame",
    position=[0.0, 0.0, 0.298],
    scale=[3.02, 0.32, 0.026],
    color=np.array([0.55, 0.55, 0.58]),
))
for _side, _sy in [("RailL", 0.155), ("RailR", -0.155)]:
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Conveyor/{_side}",
        name=f"conveyor_{_side.lower()}",
        position=[0.0, _sy, 0.33],
        scale=[2.98, 0.012, 0.04],
        color=np.array([0.60, 0.62, 0.65]),
    ))
for _li, _lx in enumerate([-0.95, 0.0, 0.95]):
    for _lsi, _ly in enumerate([-0.14, 0.14]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/CBPACell/Conveyor/Leg_{_li}_{_lsi}",
            name=f"conveyor_leg_{_li}_{_lsi}",
            position=[_lx, _ly, 0.145],
            scale=[0.05, 0.05, 0.29],
            color=np.array([0.50, 0.50, 0.52]),
        ))
for _rname, _rx in [("RollerL", -1.49), ("RollerR", 1.49)]:
    try:
        world.scene.add(VisualCylinder(
            prim_path=f"/World/CBPACell/Conveyor/{_rname}",
            name=f"conveyor_{_rname.lower()}",
            position=[_rx, 0.0, 0.302],
            radius=0.025, height=0.30,
            color=np.array([0.65, 0.65, 0.68]),
        ))
    except Exception:
        pass

# ── Workstation tables ────────────────────────────────────────────────
for _ti, _tx in enumerate([STATIONS_X[1], STATIONS_X[3]]):
    # Table top (steel surface)
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Table_{_ti}/Top",
        name=f"table_{_ti}_top",
        position=[_tx, 0.40, 0.288],
        scale=[0.55, 0.50, 0.024],
        color=np.array([0.72, 0.72, 0.75]),
    ))
    for _lj, (_ldx, _ldy) in enumerate([(-0.24, -0.22), (-0.24, 0.22),
                                          ( 0.24, -0.22), ( 0.24, 0.22)]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/CBPACell/Table_{_ti}/Leg_{_lj}",
            name=f"table_{_ti}_leg_{_lj}",
            position=[_tx + _ldx, 0.40 + _ldy, 0.142],
            scale=[0.04, 0.04, 0.284],
            color=np.array([0.50, 0.50, 0.52]),
        ))
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Table_{_ti}/Shelf",
        name=f"table_{_ti}_shelf",
        position=[_tx, 0.40, 0.10],
        scale=[0.50, 0.44, 0.016],
        color=np.array([0.60, 0.60, 0.62]),
    ))
    # Parts tray: always cuboid (small accent piece)
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Table_{_ti}/Tray",
        name=f"table_{_ti}_tray",
        position=[_tx - 0.20, 0.40, 0.306],
        scale=[0.14, 0.10, 0.020],
        color=np.array([0.20, 0.30, 0.50]),
    ))

# Robot R1 (Franka Panda at pick-and-place station)
# No scale — Franka USD is already in meters.
# Orientation: rotate to face the conveyor (−Y direction from y=0.4 toward y=0).
# Franka default forward is +X; 90° Z → faces +Y (away); −90° Z → faces −Y (toward conveyor).
_r1_ok = _add_usd_asset(
    "/World/CBPACell/Robot_R1", _FRANKA_USD,
    position=[STATIONS_X[1], 0.4, 0.30],
    orientation=[0, 0, -90],
)
if _r1_ok:
    try:
        _bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        _rng = _bb.ComputeWorldBound(world.stage.GetPrimAtPath("/World/CBPACell/Robot_R1")).GetRange()
        logger.info(f"R1 bbox min={_rng.GetMin()}, max={_rng.GetMax()}")
    except Exception:
        pass
if not _r1_ok and not world.stage.GetPrimAtPath("/World/CBPACell/Robot_R1").IsValid():
    world.scene.add(VisualCuboid(
        prim_path="/World/CBPACell/Robot_R1_fallback", name="robot_r1",
        position=[STATIONS_X[1], 0.4, 0.55],
        scale=[0.15, 0.15, 0.5], color=np.array([0.3, 0.3, 0.8]),
    ))

# Robot R2 (Franka Panda at screwdriving station)
_r2_ok = _add_usd_asset(
    "/World/CBPACell/Robot_R2", _FRANKA_USD,
    position=[STATIONS_X[3], 0.4, 0.30],
    orientation=[0, 0, -90],
)
if _r2_ok:
    try:
        _bb2 = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        _rng2 = _bb2.ComputeWorldBound(world.stage.GetPrimAtPath("/World/CBPACell/Robot_R2")).GetRange()
        logger.info(f"R2 bbox min={_rng2.GetMin()}, max={_rng2.GetMax()}")
    except Exception:
        pass
if not _r2_ok and not world.stage.GetPrimAtPath("/World/CBPACell/Robot_R2").IsValid():
    world.scene.add(VisualCuboid(
        prim_path="/World/CBPACell/Robot_R2_fallback", name="robot_r2",
        position=[STATIONS_X[3], 0.4, 0.55],
        scale=[0.15, 0.15, 0.5], color=np.array([0.3, 0.3, 0.8]),
    ))

# ── EE tracking ─────────────────────────────────────────────────────────
# When the Franka USD loaded we read the actual hand prim's world position
# each frame so carried products sit exactly in the gripper.  When the USD
# didn't load we fall back to the computed world-space EE trajectory.
_franka_loaded = {"R1": _r1_ok, "R2": _r2_ok}

# Candidate prim suffixes for the hand (varies between Isaac Sim versions)
_HAND_PRIM_SUFFIXES = ["panda_hand", "panda_link8", "panda_link7"]
_hand_prims: dict[str, "Usd.Prim"] = {}   # populated lazily

def _resolve_hand_prim(rname: str):
    """Find and cache the Franka hand prim for world-position readback."""
    if rname in _hand_prims:
        return _hand_prims[rname]
    for suffix in _HAND_PRIM_SUFFIXES:
        # Try both flat and nested paths
        for path in [f"/World/CBPACell/Robot_{rname}/{suffix}",
                     f"/World/CBPACell/Robot_{rname}/panda/{suffix}"]:
            prim = world.stage.GetPrimAtPath(path)
            if prim.IsValid():
                _hand_prims[rname] = prim
                logger.info(f"{rname}: hand prim found at {path}")
                return prim
    # Walk descendants as last resort
    root = world.stage.GetPrimAtPath(f"/World/CBPACell/Robot_{rname}")
    if root.IsValid():
        from pxr import Usd
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if "hand" in name.lower() or name == "panda_link8":
                _hand_prims[rname] = prim
                logger.info(f"{rname}: hand prim found at {prim.GetPath()}")
                return prim
    _hand_prims[rname] = None
    return None

def _read_hand_world_pos(rname: str):
    """Return [x, y, z] world position of the Franka hand prim, or None."""
    prim = _resolve_hand_prim(rname)
    if prim is None or not prim.IsValid():
        return None
    try:
        xformable = UsdGeom.Xformable(prim)
        # ComputeLocalToWorldTransform gives the accumulated world matrix
        from pxr import Usd
        mtx = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mtx.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])
    except Exception:
        return None


def _attach_tool(robot_root: str, tool_type: str) -> None:
    """Attach end-effector tool geometry as children of panda_hand.

    Geometry is defined in panda_hand's LOCAL coordinate frame (Z = distal /
    away from flange) so it follows the robot's keyframe animation exactly.

    tool_type:
        'coupling'    – coupling ring only; default parallel fingers remain visible
        'screwdriver' – hide fingers, add electric-screwdriver bit
        'probe'       – hide fingers, add circuit-test needle probe
        'suction'     – hide fingers, add 2×2 vacuum suction pad
    """
    from pxr import Usd as _Usd, Vt as _Vt

    stage = world.stage

    # ── 1. Find the panda_hand prim ──────────────────────────────────
    hand_path: str | None = None
    for _suf in ["panda_hand", "panda_link8"]:
        for _p in [f"{robot_root}/{_suf}", f"{robot_root}/panda/{_suf}"]:
            if stage.GetPrimAtPath(_p).IsValid():
                hand_path = _p
                break
        if hand_path:
            break
    if hand_path is None:
        root_prim = stage.GetPrimAtPath(robot_root)
        if root_prim.IsValid():
            for _prim in _Usd.PrimRange(root_prim):
                _n = _prim.GetName()
                if "hand" in _n.lower() or _n == "panda_link8":
                    hand_path = str(_prim.GetPath())
                    break
    if hand_path is None:
        logger.warning("_attach_tool: panda_hand not found under %s", robot_root)
        return

    # ── 2. Hide default finger prims (screwdriver / probe / suction) ─
    if tool_type != "coupling":
        _hand_prim = stage.GetPrimAtPath(hand_path)
        if _hand_prim.IsValid():
            for _fp in _Usd.PrimRange(_hand_prim):
                if "finger" in _fp.GetName().lower():
                    UsdGeom.Imageable(_fp).MakeInvisible()
            logger.info("_attach_tool: fingers hidden on %s", robot_root)

    # ── 3. Tool geometry helpers (LOCAL coords relative to panda_hand) ──
    # panda_hand local Z → points distally (away from wrist, toward tool tip).
    # We lay all geometry along +Z starting just past the mounting flange.
    _pfx = robot_root.lstrip("/").replace("/", "_").lower() + "_tool"
    _tool_xf = f"{hand_path}/Tool"
    stage.DefinePrim(_tool_xf, "Xform")

    def _cyl(name, lz, radius, height, rgb, lx=0.0, ly=0.0):
        """Cylinder along Z, centred at local (lx, ly, lz)."""
        _p = stage.DefinePrim(f"{_tool_xf}/{name}", "Cylinder")
        _c = UsdGeom.Cylinder(_p)
        _c.GetRadiusAttr().Set(radius)
        _c.GetHeightAttr().Set(height)
        _c.GetAxisAttr().Set("Z")
        _c.GetDisplayColorAttr().Set(_Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        UsdGeom.Xformable(_p).AddTranslateOp().Set(Gf.Vec3d(lx, ly, lz))

    def _box(name, lx, ly, lz, W, D, H, rgb):
        """Box of size W×D×H (full extents), centred at local (lx,ly,lz)."""
        _p = stage.DefinePrim(f"{_tool_xf}/{name}", "Cube")
        _b = UsdGeom.Cube(_p)
        _b.GetSizeAttr().Set(1.0)
        _b.GetDisplayColorAttr().Set(_Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        _xf = UsdGeom.Xformable(_p)
        _xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, lz))
        _xf.AddScaleOp().Set(Gf.Vec3d(W, D, H))

    # ── 4. Build the requested tool ──────────────────────────────────
    if tool_type == "coupling":
        # Visible coupling ring just above the Franka mounting flange;
        # the existing parallel fingers remain (already in the USD).
        _cyl("Coupling", 0.010, 0.040, 0.014, (0.52, 0.52, 0.54))
        _cyl("CouplingRim", 0.018, 0.046, 0.006, (0.35, 0.35, 0.37))

    elif tool_type == "screwdriver":
        # Quick-change coupling disk
        _cyl("Coupling",  0.012, 0.036, 0.018, (0.55, 0.55, 0.57))
        # Electric-screwdriver motor body (wider cylinder)
        _cyl("Motor",     0.052, 0.022, 0.060, (0.25, 0.25, 0.27))
        # Drive shaft
        _cyl("Shaft",     0.096, 0.008, 0.040, (0.42, 0.42, 0.44))
        # Gold hex bit
        _cyl("HexBit",    0.123, 0.005, 0.018, (0.78, 0.62, 0.18))

    elif tool_type == "probe":
        # Probe housing / body
        _cyl("Body",      0.016, 0.018, 0.028, (0.62, 0.62, 0.64))
        # Coaxial cable connector ring
        _cyl("Connector", 0.036, 0.012, 0.008, (0.35, 0.35, 0.37))
        # Slender probe shaft
        _cyl("Shaft",     0.077, 0.005, 0.064, (0.72, 0.72, 0.74))
        # Red insulated tip (marks measurement point)
        _cyl("Tip",       0.114, 0.003, 0.010, (0.88, 0.15, 0.12))

    elif tool_type == "suction":
        # Flat mounting / tool-change plate
        _box("Plate",    0.0, 0.0, 0.014,  0.090, 0.090, 0.014, (0.44, 0.44, 0.46))
        # Vacuum manifold block
        _box("Manifold", 0.0, 0.0, 0.029,  0.062, 0.062, 0.012, (0.30, 0.30, 0.32))
        # 2×2 suction cups at ±20 mm
        for _si, (_sx, _sy) in enumerate([
            (-0.020, -0.020), (-0.020, +0.020),
            (+0.020, -0.020), (+0.020, +0.020),
        ]):
            _cyl(f"Cup{_si}", 0.046, 0.010, 0.018, (0.15, 0.15, 0.17), lx=_sx, ly=_sy)
            _cyl(f"Rim{_si}", 0.058, 0.013, 0.005, (0.08, 0.08, 0.10), lx=_sx, ly=_sy)

    logger.info("_attach_tool: '%s' attached to %s", tool_type, robot_root)


# ── Cell A end-effector tooling ──────────────────────────────────────────
# R1 pick-and-place: keep default parallel fingers, add visible coupling ring.
# R2 screwdriving:   hide fingers, attach electric-screwdriver bit.
if _r1_ok:
    _attach_tool("/World/CBPACell/Robot_R1", "coupling")
if _r2_ok:
    _attach_tool("/World/CBPACell/Robot_R2", "screwdriver")

# Camera — overview of the production cell
if args.live_ui:
    try:
        from pxr import UsdGeom as _UsdGeom, Sdf
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.DefinePrim("/World/CBPACell/OverviewCam", "Camera")
        cam = _UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(18.0)  # wide angle
        xform = _UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -2.5, 2.5))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(55.0, 0.0, 0.0))  # tilt down
        # Set as active viewport camera
        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport:
            viewport.set_active_camera("/World/CBPACell/OverviewCam")
            logger.info("Viewport camera set to OverviewCam")
    except Exception as e:
        logger.debug("Could not set up camera: %s", e)

# Product cubes — pool of units that cycle through the line.
# VisualCuboid (no physics) so we control positions directly without jitter.
# Conveyor: pos z=0.3, scale z=0.05 → top surface at z=0.325
# Product: scale 0.04 → half-height 0.02 → resting centre z=0.345
_CONVEYOR_TOP_Z = 0.325
_PRODUCT_HALF   = 0.015                           # 30 mm tall housing, half = 15 mm
_PRODUCT_REST_Z = _CONVEYOR_TOP_Z + _PRODUCT_HALF # 0.340

# Electronic housing colours — subtle variation across units
_PRODUCT_COLORS = [
    np.array([0.22, 0.22, 0.24]),   # dark charcoal
    np.array([0.28, 0.28, 0.32]),   # medium gray
    np.array([0.24, 0.24, 0.26]),   # near-black
    np.array([0.30, 0.30, 0.34]),   # steel gray
]

NUM_PRODUCTS = 4
products = []
# Per-product smooth positions (float x, y, z) for lerp-based sliding
_product_pos = []
for i in range(NUM_PRODUCTS):
    start_station = (i + 1) % len(STATIONS_X)
    sx = STATIONS_X[start_station]
    cube = world.scene.add(
        VisualCuboid(
            prim_path=f"/World/CBPACell/Product_{i}",
            name=f"product_{i}",
            position=[sx, 0.0, _PRODUCT_REST_Z],
            scale=[0.10, 0.07, 0.030],           # flat rectangular housing (10×7×3 cm)
            color=_PRODUCT_COLORS[i % len(_PRODUCT_COLORS)],
        )
    )
    products.append(cube)
    _product_pos.append(np.array([sx, 0.0, _PRODUCT_REST_Z]))

# ── Robot mounting pedestals ────────────────────────────────────────
# Thick steel plinths that the Franka arms stand on (improves realism)
for _rname, _rx in [("R1", STATIONS_X[1]), ("R2", STATIONS_X[3])]:
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Pedestal_{_rname}",
        name=f"pedestal_{_rname.lower()}",
        position=[_rx, 0.40, 0.148],
        scale=[0.22, 0.22, 0.30],
        color=np.array([0.30, 0.30, 0.32]),  # dark steel
    ))

# ── Safety fencing around robot work envelopes ───────────────────────
_FENCE_COLOR  = np.array([0.95, 0.75, 0.05])  # industrial yellow
_RAIL_COLOR   = np.array([0.88, 0.70, 0.04])
_FENCE_H      = 1.40   # post height
_RAIL_H_LOW   = 0.55   # lower horizontal rail centre z
_RAIL_H_HIGH  = 1.05   # upper horizontal rail centre z

for _rname, _rx in [("R1", STATIONS_X[1]), ("R2", STATIONS_X[3])]:
    # Safety fence: open U-shape so the conveyor belt passes through freely.
    # Only back posts (y=0.90) — no front posts that would clip the conveyor
    # (conveyor frame extends to y=+0.16; front posts at y=0.02 would pass through it).
    # Side rails start at y=0.22 (just past the conveyor edge at y=0.16).
    _fence_back_posts = [(_rx - 0.65, 0.90), (_rx + 0.65, 0.90)]
    for _pi, (_px, _py) in enumerate(_fence_back_posts):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/CBPACell/Fence_{_rname}/Post_{_pi}",
            name=f"fence_{_rname.lower()}_post_{_pi}",
            position=[_px, _py, _FENCE_H / 2],
            scale=[0.06, 0.06, _FENCE_H],
            color=_FENCE_COLOR,
        ))
    # Back rail connecting the two rear posts
    for _rz in [_RAIL_H_LOW, _RAIL_H_HIGH]:
        world.scene.add(VisualCuboid(
            prim_path=f"/World/CBPACell/Fence_{_rname}/BackRail_{int(_rz*100)}",
            name=f"fence_{_rname.lower()}_backrail_{int(_rz*100)}",
            position=[_rx, 0.90, _rz],
            scale=[1.30, 0.04, 0.05],
            color=_RAIL_COLOR,
        ))
    # Side rails: start at y=0.22 (conveyor clears at y=0.16) → back post at y=0.90
    # centre_y = (0.22 + 0.90) / 2 = 0.56, length = 0.68
    for _sx2, _side in [(_rx - 0.65, "L"), (_rx + 0.65, "R")]:
        for _rz in [_RAIL_H_LOW, _RAIL_H_HIGH]:
            world.scene.add(VisualCuboid(
                prim_path=f"/World/CBPACell/Fence_{_rname}/SideRail_{_side}_{int(_rz*100)}",
                name=f"fence_{_rname.lower()}_side_{_side}_{int(_rz*100)}",
                position=[_sx2, 0.56, _rz],
                scale=[0.04, 0.68, 0.05],
                color=_RAIL_COLOR,
            ))

# ── Parts bins at entry and exit ────────────────────────────────────
# Each bin is built from 6 pieces: base + 4 walls + top rim strip.
# Dimensions: 28 cm wide (X) × 22 cm deep (Y) × 20 cm tall (Z), wall 1.5 cm thick.
def _make_bin(root, cx, cy, color):
    W, D, H, T = 0.28, 0.22, 0.20, 0.015   # width, depth, height, wall thickness
    bz = H / 2                               # bin centre z (base sits at z=0)
    dark = color * 0.65                      # slightly darker shade for rim/base
    # Derive a scene-unique name prefix from the full prim path (avoids
    # collisions when the same bin type (BinIn/BinOut) is built for multiple cells)
    _pfx = root.lstrip("/").replace("/", "_").lower()
    # Base
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Base", name=f"{_pfx}_base",
        position=[cx, cy, T / 2],
        scale=[W, D, T], color=dark,
    ))
    # Front wall  (−Y face)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/WallFront", name=f"{_pfx}_wfront",
        position=[cx, cy - D/2 + T/2, bz],
        scale=[W, T, H], color=color,
    ))
    # Back wall  (+Y face)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/WallBack", name=f"{_pfx}_wback",
        position=[cx, cy + D/2 - T/2, bz],
        scale=[W, T, H], color=color,
    ))
    # Left wall  (−X face)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/WallLeft", name=f"{_pfx}_wleft",
        position=[cx - W/2 + T/2, cy, bz],
        scale=[T, D - 2*T, H], color=color,
    ))
    # Right wall  (+X face)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/WallRight", name=f"{_pfx}_wright",
        position=[cx + W/2 - T/2, cy, bz],
        scale=[T, D - 2*T, H], color=color,
    ))
    # Top rim (slightly wider/taller lip around the opening)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Rim", name=f"{_pfx}_rim",
        position=[cx, cy, H + T/2],
        scale=[W + 0.008, D + 0.008, T * 0.8], color=dark,
    ))
    # White label on the front face
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Label", name=f"{_pfx}_label",
        position=[cx, cy - D/2 + T + 0.001, bz * 0.9],
        scale=[W * 0.55, 0.003, H * 0.35], color=np.array([0.95, 0.95, 0.97]),
    ))

_make_bin("/World/CBPACell/BinIn",  -1.45, -0.25, np.array([0.22, 0.40, 0.65]))  # blue — raw parts
_make_bin("/World/CBPACell/BinOut",  1.45, -0.25, np.array([0.18, 0.55, 0.28]))  # green — finished

# ── Workshop equipment helpers ────────────────────────────────────────
# Realistic tooling and machines placed in each cell.
# All names are derived from the full prim path so multi-cell reuse is safe.

def _make_tool_trolley(root, cx, cy):
    """Mobile 3-drawer tool chest with handle bar and wheels."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Body", name=f"{_pfx}_body",
        position=[cx, cy, 0.425], scale=[0.45, 0.40, 0.85],
        color=np.array([0.25, 0.25, 0.27]),
    ))
    for _di, _dz in enumerate([0.20, 0.42, 0.62]):
        world.scene.add(VisualCuboid(
            prim_path=f"{root}/Drawer_{_di}", name=f"{_pfx}_drawer_{_di}",
            position=[cx, cy - 0.201, _dz], scale=[0.40, 0.005, 0.16],
            color=np.array([0.32, 0.32, 0.34]),
        ))
        world.scene.add(VisualCuboid(
            prim_path=f"{root}/DHandle_{_di}", name=f"{_pfx}_dhandle_{_di}",
            position=[cx, cy - 0.207, _dz], scale=[0.08, 0.005, 0.015],
            color=np.array([0.60, 0.60, 0.62]),
        ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/PushBar", name=f"{_pfx}_pushbar",
        position=[cx, cy - 0.22, 0.88], scale=[0.40, 0.025, 0.03],
        color=np.array([0.50, 0.50, 0.52]),
    ))
    for _wi, _wox in enumerate([-0.15, 0.15]):
        world.scene.add(VisualCuboid(
            prim_path=f"{root}/Wheel_{_wi}", name=f"{_pfx}_wheel_{_wi}",
            position=[cx + _wox, cy, 0.03], scale=[0.06, 0.38, 0.06],
            color=np.array([0.15, 0.15, 0.15]),
        ))


def _make_andon_tower(root, cx, cy):
    """3-light andon status tower (green/yellow/red) on a mounting post."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Post", name=f"{_pfx}_post",
        position=[cx, cy, 0.70], scale=[0.04, 0.04, 1.40],
        color=np.array([0.40, 0.40, 0.42]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Base", name=f"{_pfx}_base",
        position=[cx, cy, 0.02], scale=[0.12, 0.12, 0.04],
        color=np.array([0.35, 0.35, 0.37]),
    ))
    for _li, (_lz, _lcolor) in enumerate([
        (1.05, np.array([0.10, 0.75, 0.15])),   # green  — running
        (1.22, np.array([0.90, 0.75, 0.05])),   # yellow — caution
        (1.39, np.array([0.85, 0.12, 0.08])),   # red    — fault
    ]):
        try:
            world.scene.add(VisualCylinder(
                prim_path=f"{root}/Light_{_li}", name=f"{_pfx}_light_{_li}",
                position=[cx, cy, _lz], radius=0.045, height=0.13,
                color=_lcolor,
            ))
        except Exception:
            world.scene.add(VisualCuboid(
                prim_path=f"{root}/Light_{_li}", name=f"{_pfx}_light_{_li}",
                position=[cx, cy, _lz], scale=[0.09, 0.09, 0.13], color=_lcolor,
            ))


def _make_hmi_terminal(root, cx, cy):
    """HMI workstation: control cabinet + monitor arm + keyboard shelf."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Cabinet", name=f"{_pfx}_cabinet",
        position=[cx, cy, 0.60], scale=[0.50, 0.28, 1.20],
        color=np.array([0.72, 0.72, 0.74]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/DoorPanel", name=f"{_pfx}_doorpanel",
        position=[cx, cy - 0.141, 0.60], scale=[0.44, 0.005, 1.10],
        color=np.array([0.65, 0.65, 0.67]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/MonitorArm", name=f"{_pfx}_monarm",
        position=[cx, cy - 0.18, 1.35], scale=[0.04, 0.18, 0.04],
        color=np.array([0.35, 0.35, 0.37]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Screen", name=f"{_pfx}_screen",
        position=[cx, cy - 0.28, 1.42], scale=[0.38, 0.03, 0.24],
        color=np.array([0.08, 0.12, 0.18]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/LEDs", name=f"{_pfx}_leds",
        position=[cx, cy - 0.141, 1.00], scale=[0.20, 0.005, 0.025],
        color=np.array([0.10, 0.70, 0.15]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/KbShelf", name=f"{_pfx}_kbshelf",
        position=[cx, cy - 0.22, 0.95], scale=[0.38, 0.18, 0.015],
        color=np.array([0.55, 0.55, 0.57]),
    ))


def _make_elec_cabinet(root, cx, cy):
    """Electrical/control panel cabinet (wall-mount style)."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Body", name=f"{_pfx}_body",
        position=[cx, cy, 0.90], scale=[0.60, 0.28, 1.80],
        color=np.array([0.70, 0.72, 0.70]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Door", name=f"{_pfx}_door",
        position=[cx, cy - 0.141, 0.90], scale=[0.55, 0.005, 1.70],
        color=np.array([0.78, 0.78, 0.76]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Handle", name=f"{_pfx}_handle",
        position=[cx + 0.22, cy - 0.148, 0.90], scale=[0.015, 0.012, 0.12],
        color=np.array([0.50, 0.50, 0.52]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/WarnLabel", name=f"{_pfx}_warnlabel",
        position=[cx, cy - 0.142, 1.35], scale=[0.10, 0.003, 0.10],
        color=np.array([0.95, 0.80, 0.05]),
    ))


def _make_part_fixture(root, cx, cy, conveyor_z):
    """Part locating jig: low tray with 3 alignment pegs, sits on conveyor."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Tray", name=f"{_pfx}_tray",
        position=[cx, cy, conveyor_z + 0.008], scale=[0.22, 0.16, 0.012],
        color=np.array([0.50, 0.48, 0.46]),
    ))
    for _pi, _pxo in enumerate([-0.06, 0.0, 0.06]):
        try:
            world.scene.add(VisualCylinder(
                prim_path=f"{root}/Peg_{_pi}", name=f"{_pfx}_peg_{_pi}",
                position=[cx + _pxo, cy - 0.04, conveyor_z + 0.024],
                radius=0.006, height=0.022, color=np.array([0.70, 0.40, 0.10]),
            ))
        except Exception:
            world.scene.add(VisualCuboid(
                prim_path=f"{root}/Peg_{_pi}", name=f"{_pfx}_peg_{_pi}",
                position=[cx + _pxo, cy - 0.04, conveyor_z + 0.024],
                scale=[0.012, 0.012, 0.022], color=np.array([0.70, 0.40, 0.10]),
            ))


def _make_vision_camera(root, cx, cy):
    """Overhead vision inspection camera on a vertical post."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Post", name=f"{_pfx}_post",
        position=[cx, cy, 0.90], scale=[0.025, 0.025, 1.80],
        color=np.array([0.35, 0.35, 0.37]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Body", name=f"{_pfx}_body",
        position=[cx, cy + 0.04, 1.82], scale=[0.09, 0.07, 0.06],
        color=np.array([0.12, 0.12, 0.14]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Lens", name=f"{_pfx}_lens",
        position=[cx, cy - 0.005, 1.82], scale=[0.035, 0.014, 0.035],
        color=np.array([0.06, 0.06, 0.22]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Ring", name=f"{_pfx}_ring",
        position=[cx, cy - 0.006, 1.82], scale=[0.05, 0.004, 0.05],
        color=np.array([0.50, 0.50, 0.52]),
    ))


def _make_test_stand(root, cx, cy, conveyor_z):
    """Test & measurement stand with probe arms and dial gauge (Cell B)."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    # Stand base
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Base", name=f"{_pfx}_base",
        position=[cx, cy, conveyor_z + 0.010], scale=[0.30, 0.20, 0.016],
        color=np.array([0.35, 0.35, 0.38]),
    ))
    # Upright column
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Column", name=f"{_pfx}_column",
        position=[cx - 0.10, cy, conveyor_z + 0.17], scale=[0.025, 0.025, 0.34],
        color=np.array([0.42, 0.42, 0.44]),
    ))
    # Horizontal arm
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Arm", name=f"{_pfx}_arm",
        position=[cx, cy, conveyor_z + 0.34], scale=[0.20, 0.020, 0.020],
        color=np.array([0.42, 0.42, 0.44]),
    ))
    # Probe tip
    try:
        world.scene.add(VisualCylinder(
            prim_path=f"{root}/Probe", name=f"{_pfx}_probe",
            position=[cx + 0.06, cy, conveyor_z + 0.30],
            radius=0.005, height=0.08, color=np.array([0.75, 0.75, 0.78]),
        ))
    except Exception:
        world.scene.add(VisualCuboid(
            prim_path=f"{root}/Probe", name=f"{_pfx}_probe",
            position=[cx + 0.06, cy, conveyor_z + 0.30],
            scale=[0.010, 0.010, 0.08], color=np.array([0.75, 0.75, 0.78]),
        ))
    # Dial gauge body (small round display)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Gauge", name=f"{_pfx}_gauge",
        position=[cx - 0.08, cy - 0.015, conveyor_z + 0.34],
        scale=[0.05, 0.015, 0.05], color=np.array([0.90, 0.90, 0.92]),
    ))


def _make_pack_station(root, cx, cy):
    """Packing workstation: table with tape dispenser and label printer."""
    _pfx = root.lstrip("/").replace("/", "_").lower()
    # Table top
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Top", name=f"{_pfx}_top",
        position=[cx, cy, 0.88], scale=[0.70, 0.50, 0.025],
        color=np.array([0.82, 0.75, 0.62]),  # wood tone
    ))
    # Table legs (2 front, shared apron)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/Apron", name=f"{_pfx}_apron",
        position=[cx, cy, 0.46], scale=[0.68, 0.04, 0.88],
        color=np.array([0.55, 0.48, 0.38]),
    ))
    # Tape dispenser (gray box)
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/TapeDisp", name=f"{_pfx}_tapedisp",
        position=[cx - 0.22, cy - 0.18, 0.91], scale=[0.08, 0.06, 0.06],
        color=np.array([0.40, 0.40, 0.42]),
    ))
    # Label printer
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/LabelPrinter", name=f"{_pfx}_printer",
        position=[cx + 0.18, cy - 0.15, 0.91], scale=[0.16, 0.12, 0.08],
        color=np.array([0.88, 0.88, 0.90]),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{root}/PrinterSlot", name=f"{_pfx}_printslot",
        position=[cx + 0.18, cy - 0.215, 0.915], scale=[0.10, 0.005, 0.015],
        color=np.array([0.20, 0.20, 0.22]),
    ))


# ── Cell A — workshop equipment ───────────────────────────────────────
# Tool trolley beside operator (south of conveyor)
# cy=-1.20: north face at y=-1.00, giving H1 (y=-0.45) ~0.55 m clearance
_make_tool_trolley("/World/CBPACell/ToolTrolley",
                   STATIONS_X[2] - 0.50, -1.20)

# Andon status towers behind robot safety fences
_make_andon_tower("/World/CBPACell/AndonR1", STATIONS_X[1], 1.05)
_make_andon_tower("/World/CBPACell/AndonR2", STATIONS_X[3], 1.05)

# HMI terminal at exit end of line
_make_hmi_terminal("/World/CBPACell/HMI",
                   STATIONS_X[4] + 0.40, 0.20)

# Electrical cabinet at entry end (back wall)
_make_elec_cabinet("/World/CBPACell/ElecCabinet",
                   STATIONS_X[0] - 0.55, 0.80)

# Part fixture/jig at the human insertion + inspection station
_make_part_fixture("/World/CBPACell/PartFixture",
                   STATIONS_X[2], 0.0, _CONVEYOR_TOP_Z)

# Overhead vision inspection camera above station 2
_make_vision_camera("/World/CBPACell/InspCamera", STATIONS_X[2], 0.18)

# ── Overhead light fixtures ──────────────────────────────────────────
# Three industrial fluorescent-style fixtures above the production line
for _fi, _fx in enumerate([STATIONS_X[1], STATIONS_X[2], STATIONS_X[3]]):
    # Housing
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Light_{_fi}/Housing",
        name=f"light_{_fi}_housing",
        position=[_fx, -0.10, 2.18],
        scale=[0.60, 0.14, 0.06],
        color=np.array([0.80, 0.80, 0.82]),
    ))
    # Diffuser panel (bright white)
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/Light_{_fi}/Diffuser",
        name=f"light_{_fi}_diffuser",
        position=[_fx, -0.10, 2.15],
        scale=[0.56, 0.12, 0.008],
        color=np.array([0.98, 0.98, 1.00]),
    ))

# ── Floor safety markings (yellow painted lanes) ─────────────────────
# Robot hazard zones — yellow stripes in front of each robot station
for _fi, _fx in enumerate([STATIONS_X[1], STATIONS_X[3]]):
    world.scene.add(VisualCuboid(
        prim_path=f"/World/CBPACell/FloorMark_{_fi}",
        name=f"floor_mark_{_fi}",
        position=[_fx, 0.10, 0.002],
        scale=[1.20, 0.80, 0.004],
        color=np.array([0.95, 0.82, 0.10]),  # yellow floor marking
    ))

# Human operator at the insertion/inspection station.
# Operator stands in front of the conveyor (y<0) facing toward it (+Y).
# Operator at the insertion/inspection station, facing the conveyor (+Y).
# Try 0° first — the bbox log below will confirm the facing direction.
_operator_is_model = _add_usd_asset(
    "/World/CBPACell/Operator", _HUMAN_USD,
    position=[STATIONS_X[2], -0.45, 0.0],
    orientation=[0, 0, 180],   # flip to face +Y (toward conveyor)
    # No scale — the character file is already in meters.
)
# Log the character's bounding box to determine default facing
if _operator_is_model:
    try:
        _op_bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        _op_bound = _op_bb.ComputeWorldBound(world.stage.GetPrimAtPath("/World/CBPACell/Operator"))
        _op_rng = _op_bound.GetRange()
        logger.info(f"Operator bbox min={_op_rng.GetMin()}, max={_op_rng.GetMax()}, "
                     f"size=({_op_rng.GetMax()[0]-_op_rng.GetMin()[0]:.3f}, "
                     f"{_op_rng.GetMax()[1]-_op_rng.GetMin()[1]:.3f}, "
                     f"{_op_rng.GetMax()[2]-_op_rng.GetMin()[2]:.3f})")
    except Exception as e:
        logger.debug(f"Operator bbox: {e}")
if not _operator_is_model and not world.stage.GetPrimAtPath("/World/CBPACell/Operator").IsValid():
    world.scene.add(
        VisualCuboid(
            prim_path="/World/CBPACell/Operator_fallback",
            name="operator",
            position=[STATIONS_X[2], -0.5, 0.85],
            scale=[0.25, 0.2, 1.7],
            color=np.array([0.3, 0.8, 0.3]),
        )
    )

# Add a dome light for ambient illumination
try:
    stage = omni.usd.get_context().get_stage()
    from pxr import UsdLux
    dome = UsdLux.DomeLight.Define(stage, "/World/CBPACell/DomeLight")
    dome.GetIntensityAttr().Set(500.0)
    dome.GetColorAttr().Set(Gf.Vec3f(0.95, 0.95, 1.0))  # slightly cool white

    # Directional light for shadows
    dist = UsdLux.DistantLight.Define(stage, "/World/CBPACell/KeyLight")
    dist.GetIntensityAttr().Set(3000.0)
    dist.GetAngleAttr().Set(1.0)  # soft shadows
    xform = UsdGeom.Xformable(dist.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))
except Exception as e:
    logger.debug(f"Lighting setup: {e}")

logger.info("CBPA production cell scene created")

# ── Multi-cell factory extension ────────────────────────────────────
# When --multi-cell is set, add Cell B, AGV corridor, buffer zone,
# and supervisor station.  Cell A stays at its existing y=0 position.
# Cell B is placed at y=-2.0 (mirrored layout facing +Y toward AGV corridor).
_MULTI_CELL = args.multi_cell
_CELL_B_Y_OFFSET = -4.0   # Cell B conveyor centre y
# Increased from -2.0: gives ~1.9 m clear passage on each side of the AGV
# corridor (Cell A south edge ≈ y=-1.05, Cell B north fence ≈ y=-3.35,
# AGV lane centre ≈ y=-2.0).

# Cell B state (only used in multi-cell mode)
_cellB_products = []
_cellB_product_pos = []
_cellB_product_station = []
_cellB_units_completed = 0
_cellB_steps_per_cycle = 55  # slightly different pace than Cell A

# AGV state
_agv_pos = np.array([-1.5, -1.0, 0.10])
_agv_dir = 1       # +1 = moving right, -1 = moving left
_AGV_SPEED = 0.008  # world-units per step
_AGV_LEFT  = -1.5
_AGV_RIGHT =  1.5
_agv_carrying = False

if _MULTI_CELL:
    _BY = _CELL_B_Y_OFFSET
    logger.info(f"Building Cell B at y_offset={_BY}")

    # ── Cell B conveyor ─────────────────────────────────────────────
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/CellB/Conveyor/Belt",
        name="cellB_conveyor_belt",
        position=[0.0, _BY, 0.302],
        scale=[2.96, 0.26, 0.018],
        color=np.array([0.18, 0.18, 0.18]),
    ))
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/CellB/Conveyor/Frame",
        name="cellB_conveyor_frame",
        position=[0.0, _BY, 0.298],
        scale=[3.02, 0.32, 0.026],
        color=np.array([0.55, 0.55, 0.58]),
    ))
    for _side, _sy in [("RailL", 0.155), ("RailR", -0.155)]:
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/Conveyor/{_side}",
            name=f"cellB_conveyor_{_side.lower()}",
            position=[0.0, _BY + _sy, 0.33],
            scale=[2.98, 0.012, 0.04],
            color=np.array([0.60, 0.62, 0.65]),
        ))
    for _li, _lx in enumerate([-0.95, 0.0, 0.95]):
        for _lsi, _ly in enumerate([-0.14, 0.14]):
            world.scene.add(VisualCuboid(
                prim_path=f"/World/Factory/CellB/Conveyor/Leg_{_li}_{_lsi}",
                name=f"cellB_leg_{_li}_{_lsi}",
                position=[_lx, _BY + _ly, 0.145],
                scale=[0.05, 0.05, 0.29],
                color=np.array([0.50, 0.50, 0.52]),
            ))

    # ── Cell B robots (R3 and R4) ───────────────────────────────────
    _r3_ok = _add_usd_asset(
        "/World/Factory/CellB/Robot_R3", _FRANKA_USD,
        position=[STATIONS_X[1], _BY + 0.4, 0.30],
        orientation=[0, 0, -90],
    )
    if not _r3_ok:
        world.scene.add(VisualCuboid(
            prim_path="/World/Factory/CellB/Robot_R3_fallback",
            name="robot_r3",
            position=[STATIONS_X[1], _BY + 0.4, 0.55],
            scale=[0.15, 0.15, 0.5],
            color=np.array([0.3, 0.3, 0.8]),
        ))

    _r4_ok = _add_usd_asset(
        "/World/Factory/CellB/Robot_R4", _FRANKA_USD,
        position=[STATIONS_X[3], _BY + 0.4, 0.30],
        orientation=[0, 0, -90],
    )
    if not _r4_ok:
        world.scene.add(VisualCuboid(
            prim_path="/World/Factory/CellB/Robot_R4_fallback",
            name="robot_r4",
            position=[STATIONS_X[3], _BY + 0.4, 0.55],
            scale=[0.15, 0.15, 0.5],
            color=np.array([0.3, 0.3, 0.8]),
        ))

    # ── Cell B end-effector tooling ──────────────────────────────────
    # R3 functional testing: needle probe (circuit-board test).
    # R4 packaging:          2×2 vacuum suction pad.
    if _r3_ok:
        _attach_tool("/World/Factory/CellB/Robot_R3", "probe")
    if _r4_ok:
        _attach_tool("/World/Factory/CellB/Robot_R4", "suction")

    # ── Cell B pedestals ────────────────────────────────────────────
    for _rname, _rx in [("R3", STATIONS_X[1]), ("R4", STATIONS_X[3])]:
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/Pedestal_{_rname}",
            name=f"pedestal_{_rname.lower()}",
            position=[_rx, _BY + 0.40, 0.148],
            scale=[0.22, 0.22, 0.30],
            color=np.array([0.30, 0.30, 0.32]),
        ))

    # ── Cell B workstation tables ───────────────────────────────────
    for _ti, _tx in enumerate([STATIONS_X[1], STATIONS_X[3]]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/Table_{_ti}/Top",
            name=f"cellB_table_{_ti}_top",
            position=[_tx, _BY + 0.40, 0.288],
            scale=[0.55, 0.50, 0.024],
            color=np.array([0.72, 0.72, 0.75]),
        ))
        for _lj, (_ldx, _ldy) in enumerate([(-0.24, -0.22), (-0.24, 0.22),
                                              ( 0.24, -0.22), ( 0.24, 0.22)]):
            world.scene.add(VisualCuboid(
                prim_path=f"/World/Factory/CellB/Table_{_ti}/Leg_{_lj}",
                name=f"cellB_table_{_ti}_leg_{_lj}",
                position=[_tx + _ldx, _BY + 0.40 + _ldy, 0.142],
                scale=[0.04, 0.04, 0.284],
                color=np.array([0.50, 0.50, 0.52]),
            ))

    # ── Corridor boundary constants (defined early, used throughout) ────
    _CB_FENCE_BACK = 0.65   # how far north of Cell B conveyor the back fence sits
    _bol_ca_y = -0.35                         # Cell A south edge marker
    _bol_cb_y = _BY + _CB_FENCE_BACK + 0.25  # Cell B north edge marker (just outside fence)
    # Buffer handoff shelf sits near Cell A side so Cell A can drop parts
    # without the AGV having to travel far into Cell A territory.
    _buf_y  = _bol_ca_y - 0.50     # ≈ -0.85  (just inside corridor, Cell A side)
    # Material supply cart sits near Cell B side of the corridor.
    _cart_y = _bol_cb_y + 0.50     # ≈ -2.60  (just inside corridor, Cell B side)

    # ── Cell B safety fencing ───────────────────────────────────────
    # Open U-shape fence: back posts + back rail + side rails.
    # No front posts — conveyor passes freely through the open south end.
    # Side rails start at _BY+0.22 (conveyor frame clears at _BY+0.16).
    for _rname, _rx in [("R3", STATIONS_X[1]), ("R4", STATIONS_X[3])]:
        _fence_back_posts = [
            (_rx - 0.65, _BY + _CB_FENCE_BACK),
            (_rx + 0.65, _BY + _CB_FENCE_BACK),
        ]
        for _pi, (_px, _py) in enumerate(_fence_back_posts):
            world.scene.add(VisualCuboid(
                prim_path=f"/World/Factory/CellB/Fence_{_rname}/Post_{_pi}",
                name=f"cellB_fence_{_rname.lower()}_post_{_pi}",
                position=[_px, _py, _FENCE_H / 2],
                scale=[0.06, 0.06, _FENCE_H],
                color=_FENCE_COLOR,
            ))
        for _rz in [_RAIL_H_LOW, _RAIL_H_HIGH]:
            world.scene.add(VisualCuboid(
                prim_path=f"/World/Factory/CellB/Fence_{_rname}/BackRail_{int(_rz*100)}",
                name=f"cellB_fence_{_rname.lower()}_backrail_{int(_rz*100)}",
                position=[_rx, _BY + _CB_FENCE_BACK, _rz],
                scale=[1.30, 0.04, 0.05],
                color=_RAIL_COLOR,
            ))
        # Side rails: _BY+0.22 (past conveyor) → _BY+_CB_FENCE_BACK
        # centre_y = _BY + (0.22 + _CB_FENCE_BACK) / 2 = _BY + 0.435, length = 0.43
        _sr_cy = _BY + (0.22 + _CB_FENCE_BACK) / 2
        _sr_len = _CB_FENCE_BACK - 0.22
        for _sx2, _side in [(_rx - 0.65, "L"), (_rx + 0.65, "R")]:
            for _rz in [_RAIL_H_LOW, _RAIL_H_HIGH]:
                world.scene.add(VisualCuboid(
                    prim_path=f"/World/Factory/CellB/Fence_{_rname}/SideRail_{_side}_{int(_rz*100)}",
                    name=f"cellB_fence_{_rname.lower()}_side_{_side}_{int(_rz*100)}",
                    position=[_sx2, _sr_cy, _rz],
                    scale=[0.04, _sr_len, 0.05],
                    color=_RAIL_COLOR,
                ))

    # ── Cell B bins ─────────────────────────────────────────────────
    _make_bin("/World/Factory/CellB/BinIn",  -1.45, _BY - 0.25, np.array([0.22, 0.40, 0.65]))
    _make_bin("/World/Factory/CellB/BinOut",  1.45, _BY - 0.25, np.array([0.18, 0.55, 0.28]))

    # ── Cell B products (4 units) ───────────────────────────────────
    _CELLB_PRODUCT_COLORS = [
        np.array([0.25, 0.20, 0.30]),   # dark plum
        np.array([0.30, 0.25, 0.35]),   # lavender gray
        np.array([0.22, 0.22, 0.28]),   # blue-black
        np.array([0.28, 0.28, 0.34]),   # slate
    ]
    for i in range(4):
        start_station = (i + 1) % len(STATIONS_X)
        sx = STATIONS_X[start_station]
        cube = world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Factory/CellB/Product_{i}",
                name=f"cellB_product_{i}",
                position=[sx, _BY, _PRODUCT_REST_Z],
                scale=[0.10, 0.07, 0.030],
                color=_CELLB_PRODUCT_COLORS[i % len(_CELLB_PRODUCT_COLORS)],
            )
        )
        _cellB_products.append(cube)
        _cellB_product_pos.append(np.array([sx, _BY, _PRODUCT_REST_Z]))
        _cellB_product_station.append(i % len(STATIONS_X))

    # ── Cell B operator (H2) ────────────────────────────────────────
    _h2_is_model = _add_usd_asset(
        "/World/Factory/CellB/Operator_H2", _HUMAN_USD,
        position=[STATIONS_X[2], _BY - 0.45, 0.0],
        orientation=[0, 0, 180],
    )
    if not _h2_is_model and not world.stage.GetPrimAtPath("/World/Factory/CellB/Operator_H2").IsValid():
        world.scene.add(VisualCuboid(
            prim_path="/World/Factory/CellB/Operator_H2_fallback",
            name="operator_h2",
            position=[STATIONS_X[2], _BY - 0.5, 0.85],
            scale=[0.25, 0.2, 1.7],
            color=np.array([0.8, 0.6, 0.3]),  # orange-ish to distinguish from H1
        ))

    # ── Cell B overhead lights ──────────────────────────────────────
    for _fi, _fx in enumerate([STATIONS_X[1], STATIONS_X[2], STATIONS_X[3]]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/Light_{_fi}/Housing",
            name=f"cellB_light_{_fi}_housing",
            position=[_fx, _BY - 0.10, 2.18],
            scale=[0.60, 0.14, 0.06],
            color=np.array([0.80, 0.80, 0.82]),
        ))
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/Light_{_fi}/Diffuser",
            name=f"cellB_light_{_fi}_diffuser",
            position=[_fx, _BY - 0.10, 2.15],
            scale=[0.56, 0.12, 0.008],
            color=np.array([0.98, 0.98, 1.00]),
        ))

    # ── Cell B floor markings ───────────────────────────────────────
    for _fi, _fx in enumerate([STATIONS_X[1], STATIONS_X[3]]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/CellB/FloorMark_{_fi}",
            name=f"cellB_floor_mark_{_fi}",
            position=[_fx, _BY + 0.10, 0.002],
            scale=[1.20, 0.80, 0.004],
            color=np.array([0.95, 0.82, 0.10]),
        ))

    # ── AGV corridor ────────────────────────────────────────────────
    # AGV track (painted lane on floor between cells)
    _agv_y = _BY / 2   # halfway between Cell A (y=0) and Cell B (y=-2)
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/AGV/Track",
        name="agv_track",
        position=[0.0, _agv_y, 0.001],
        scale=[3.5, 0.5, 0.003],
        color=np.array([0.35, 0.55, 0.35]),  # green lane
    ))

    # AGV vehicle (flat mobile platform)
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/AGV/Vehicle",
        name="agv_vehicle",
        position=[_agv_pos[0], _agv_y, 0.08],
        scale=[0.35, 0.25, 0.12],
        color=np.array([0.20, 0.45, 0.70]),  # blue AGV
    ))

    # AGV wheels (4 small cylinders)
    for _wi, (_wx, _wy) in enumerate([(-0.12, -0.10), (-0.12, 0.10),
                                        (0.12, -0.10), (0.12, 0.10)]):
        try:
            world.scene.add(VisualCylinder(
                prim_path=f"/World/Factory/AGV/Wheel_{_wi}",
                name=f"agv_wheel_{_wi}",
                position=[_agv_pos[0] + _wx, _agv_y + _wy, 0.03],
                radius=0.03, height=0.025,
                color=np.array([0.15, 0.15, 0.15]),
            ))
        except Exception:
            pass

    # ── Buffer zone (Cell A side of corridor) ───────────────────────
    # Handoff shelf positioned near Cell A's south boundary (_buf_y ≈ -0.85)
    # so Cell A can deposit finished parts without occupying the main lane.
    # The AGV travels to this y to pick up; it does NOT sit in the passage.
    _buf_x = 1.8
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/Buffer/Shelf",
        name="buffer_shelf",
        position=[_buf_x, _buf_y, 0.40],
        scale=[0.50, 0.30, 0.02],
        color=np.array([0.60, 0.60, 0.62]),
    ))
    # Shelf legs
    for _bi, _bx in enumerate([-0.20, 0.20]):
        for _bsi, _by2 in enumerate([-0.12, 0.12]):
            world.scene.add(VisualCuboid(
                prim_path=f"/World/Factory/Buffer/Leg_{_bi}_{_bsi}",
                name=f"buffer_leg_{_bi}_{_bsi}",
                position=[_buf_x + _bx, _buf_y + _by2, 0.20],
                scale=[0.04, 0.04, 0.40],
                color=np.array([0.50, 0.50, 0.52]),
            ))
    # Buffer slot markers
    for _si in range(4):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/Buffer/Slot_{_si}",
            name=f"buffer_slot_{_si}",
            position=[_buf_x - 0.15 + _si * 0.10, _buf_y, 0.415],
            scale=[0.08, 0.06, 0.005],
            color=np.array([0.90, 0.90, 0.92]),
        ))

    # ── Supervisor station (monitoring desk) ────────────────────────
    _sup_x = -2.5
    # Desk
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/Supervisor/Desk",
        name="supervisor_desk",
        position=[_sup_x, _agv_y, 0.38],
        scale=[0.80, 0.50, 0.03],
        color=np.array([0.50, 0.35, 0.20]),  # wood tone
    ))
    # Desk legs
    for _di, (_ddx, _ddy) in enumerate([(-0.35, -0.20), (-0.35, 0.20),
                                          (0.35, -0.20), (0.35, 0.20)]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/Supervisor/DeskLeg_{_di}",
            name=f"supervisor_deskleg_{_di}",
            position=[_sup_x + _ddx, _agv_y + _ddy, 0.19],
            scale=[0.04, 0.04, 0.38],
            color=np.array([0.45, 0.30, 0.18]),
        ))
    # Monitor
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/Supervisor/Monitor",
        name="supervisor_monitor",
        position=[_sup_x, _agv_y, 0.60],
        scale=[0.40, 0.03, 0.25],
        color=np.array([0.10, 0.10, 0.12]),  # dark screen
    ))
    # Monitor stand
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/Supervisor/MonitorStand",
        name="supervisor_stand",
        position=[_sup_x, _agv_y, 0.46],
        scale=[0.06, 0.06, 0.12],
        color=np.array([0.30, 0.30, 0.32]),
    ))
    # Chair (simple cube)
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/Supervisor/Chair",
        name="supervisor_chair",
        position=[_sup_x, _agv_y - 0.40, 0.25],
        scale=[0.35, 0.35, 0.50],
        color=np.array([0.15, 0.15, 0.18]),
    ))

    # ── Factory floor boundary ──────────────────────────────────────
    # Large floor plate to visually define the factory footprint
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/FloorPlate",
        name="factory_floor",
        position=[0.0, _agv_y, -0.005],
        scale=[6.5, 7.5, 0.005],   # wider Y to cover full Cell A→Cell B span (incl. pack station south)
        color=np.array([0.82, 0.82, 0.80]),  # light concrete
    ))

    # Adjust camera for factory-wide view if viewport is open
    if args.live_ui:
        try:
            cam_prim = world.stage.GetPrimAtPath("/World/CBPACell/OverviewCam")
            if cam_prim.IsValid():
                xform = UsdGeom.Xformable(cam_prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(0.0, _agv_y - 3.5, 5.5))  # pull back for full factory view
                    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        op.Set(Gf.Vec3d(50.0, 0.0, 0.0))
        except Exception:
            pass

    # ── Cell B workshop equipment (test & pack) ──────────────────────
    # Tool trolley south of Cell B conveyor (operator H2 side)
    # cy=_BY-1.20: north face at _BY-1.00, giving H2 (y=_BY-0.45) ~0.55 m clearance
    _make_tool_trolley("/World/Factory/CellB/ToolTrolley",
                       STATIONS_X[2] - 0.50, _BY - 1.20)

    # Andon towers behind Cell B robot fences (north side, inside fence boundary)
    _make_andon_tower("/World/Factory/CellB/AndonR3", STATIONS_X[1], _BY + 0.60)
    _make_andon_tower("/World/Factory/CellB/AndonR4", STATIONS_X[3], _BY + 0.60)

    # HMI terminal on the north side of Cell B (away from BinOut/PackStation)
    _make_hmi_terminal("/World/Factory/CellB/HMI",
                       STATIONS_X[4] + 0.40, _BY + 0.35)

    # Electrical cabinet at entry end of Cell B
    _make_elec_cabinet("/World/Factory/CellB/ElecCabinet",
                       STATIONS_X[0] - 0.55, _BY + 0.60)

    # Test & measurement stand at R3 station (functional testing)
    _make_test_stand("/World/Factory/CellB/TestStand",
                     STATIONS_X[1], _BY, _CONVEYOR_TOP_Z)

    # Packing workstation shifted west to clear BinOut (x=[1.31,1.59])
    # cx=1.05: right edge at x=1.40, left edge at x=0.70; BinOut left edge at x=1.31 → gap 0.09 m
    _make_pack_station("/World/Factory/CellB/PackStation",
                       1.05, _BY - 1.10)

    # Vision inspection camera above Cell B station 2
    _make_vision_camera("/World/Factory/CellB/InspCamera",
                        STATIONS_X[2], _BY + 0.18)

    # ── Factory corridor equipment ────────────────────────────────────
    # Material supply cart near Cell B side of corridor (_cart_y ≈ -2.60)
    # so it's out of the main AGV lane and close to where Cell B draws supplies.
    _make_tool_trolley("/World/Factory/MaterialCart",
                       _buf_x - 0.60, _cart_y)

    # Safety bollards at Cell A south boundary and Cell B north boundary.
    # _bol_ca_y and _bol_cb_y are defined earlier in this block.
    for _bol_i, (_bx, _by3) in enumerate([
        (-1.55, _bol_ca_y), (1.55, _bol_ca_y),    # Cell A south edge
        (-1.55, _bol_cb_y), (1.55, _bol_cb_y),    # Cell B north edge
    ]):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/Bollard_{_bol_i}",
            name=f"factory_bollard_{_bol_i}",
            position=[_bx, _by3, 0.25],
            scale=[0.10, 0.10, 0.50],
            color=np.array([0.90, 0.55, 0.05]),  # orange bollard
        ))
        # Reflective stripe
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Factory/BollardStripe_{_bol_i}",
            name=f"factory_bollard_stripe_{_bol_i}",
            position=[_bx, _by3, 0.30],
            scale=[0.102, 0.102, 0.06],
            color=np.array([0.96, 0.96, 0.96]),
        ))

    # Fire extinguisher near supervisor desk (red cylinder on wall bracket)
    try:
        world.scene.add(VisualCylinder(
            prim_path="/World/Factory/FireExt/Cylinder",
            name="fire_ext_cylinder",
            position=[_sup_x + 0.52, _agv_y, 0.55],
            radius=0.04, height=0.55,
            color=np.array([0.85, 0.10, 0.08]),
        ))
    except Exception:
        world.scene.add(VisualCuboid(
            prim_path="/World/Factory/FireExt/Cylinder",
            name="fire_ext_cylinder",
            position=[_sup_x + 0.52, _agv_y, 0.55],
            scale=[0.08, 0.08, 0.55],
            color=np.array([0.85, 0.10, 0.08]),
        ))
    world.scene.add(VisualCuboid(
        prim_path="/World/Factory/FireExt/Bracket",
        name="fire_ext_bracket",
        position=[_sup_x + 0.52, _agv_y + 0.06, 0.55],
        scale=[0.10, 0.05, 0.08],
        color=np.array([0.40, 0.40, 0.42]),
    ))

    logger.info("Multi-cell factory layout created (Cell A + Cell B + AGV + Buffer + Supervisor)")


# ── Animation state ─────────────────────────────────────────────────
import math

# Product flow
_product_station = [i % len(STATIONS_X) for i in range(NUM_PRODUCTS)]
_units_completed = 0
_steps_per_cycle = 50
_last_cycle_step = 0

# R1 degradation state (set by /scene/disturbance)
_r1_amplitude_scale = 1.0
_r1_speed_scale = 1.0

# Robot arm controllers (Franka joint drives)
_robot_controllers = {}

def _init_robot_controllers():
    """Find Franka articulations and set up joint drives."""
    try:
        try:
            from isaacsim.core.api.articulations import Articulation
        except ImportError:
            from omni.isaac.core.articulations import Articulation
    except ImportError as e:
        logger.warning(f"Articulation import failed — joint drive unavailable: {e}")
        return

    candidates = [("R1", "/World/CBPACell/Robot_R1"),
                  ("R2", "/World/CBPACell/Robot_R2")]
    if _MULTI_CELL:
        candidates += [("R3", "/World/Factory/CellB/Robot_R3"),
                       ("R4", "/World/Factory/CellB/Robot_R4")]
    for rname, prim_path in candidates:
        prim = world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logger.warning(f"{rname}: prim at {prim_path!r} is not valid — USD asset did not load")
            continue

        # Check the prim has physics/articulation schema (empty reference prims won't)
        prim_type = prim.GetTypeName()
        children = list(prim.GetChildren())
        logger.info(f"{rname}: prim found — type={prim_type!r}, children={len(children)}")

        art = None
        # Try reusing an already-registered scene object first
        try:
            art = world.scene.get_object(f"robot_{rname.lower()}_art")
            if art is not None:
                logger.info(f"{rname}: reused existing scene object")
        except Exception:
            art = None

        if art is None:
            try:
                art = Articulation(prim_path=prim_path, name=f"robot_{rname.lower()}_art")
                world.scene.add(art)
                logger.info(f"{rname}: Articulation created and added to scene")
            except Exception as e:
                logger.warning(f"{rname}: Articulation init failed: {e}")
                continue

        # Verify DOF count after world.reset() has been called
        try:
            ndof = art.num_dof
            logger.info(f"{rname}: num_dof={ndof}")
            if ndof > 0:
                _robot_controllers[rname] = art
                logger.info(f"{rname}: robot controller ready ({ndof} DOF)")
            else:
                logger.warning(f"{rname}: num_dof=0 — articulation has no driven joints; "
                               "Franka USD may not have loaded properly")
        except Exception as e:
            logger.warning(f"{rname}: could not read num_dof: {e}")

# ── Robot motion keyframes ────────────────────────────────────────────────────
# Franka Panda 7-DOF joint angles (radians).
# Index: 0=base_pan, 1=shoulder_pitch, 2=upper_arm_rot,
#        3=elbow_pitch, 4=forearm_rot, 5=wrist_pitch, 6=wrist_rot
# Finger joints (7, 8) are left at zero.

_R1_HOME = np.array([ 0.0,  -0.785,  0.0, -2.356,  0.0,  1.571,  0.785])
_R2_HOME = np.array([ 0.0,  -0.785,  0.0, -2.356,  0.0,  1.571,  0.785])

# R1: pick-and-place
#   reach over conveyor → descend to grasp height → lift part →
#   swing base to assembly point → lower to place → retract → home
_R1_CYCLE_STEPS = 300
_R1_PHASES = [
    # (end_fraction, joint_angles)
    (0.15, np.array([ 0.0,   0.40,  0.0, -1.50,  0.0,  2.00,  0.785])),  # extend over conveyor
    (0.28, np.array([ 0.0,   0.60,  0.0, -1.00,  0.0,  1.60,  0.785])),  # descend to pick
    (0.40, np.array([ 0.0,   0.10,  0.0, -1.80,  0.0,  1.80,  0.785])),  # lift part up
    (0.58, np.array([-0.80,  0.10,  0.0, -1.80,  0.0,  1.80,  0.785])),  # swing to place side
    (0.72, np.array([-0.80,  0.40,  0.0, -1.20,  0.0,  1.50,  0.785])),  # lower to place
    (0.85, np.array([-0.40,  0.00,  0.0, -2.00,  0.0,  1.90,  0.785])),  # retract after place
    (1.00, _R1_HOME.copy()),                                                # return home
]

# R2: screwdriving two fasteners per cycle
#   position above screw 1 → engage → drive (j6 spins as tool) → lift →
#   reposition to screw 2 → engage → drive → lift → home
_R2_CYCLE_STEPS = 260
_R2_PHASES = [
    (0.12, np.array([ 0.35,  0.30,  0.0, -1.40,  0.0,  1.20,  0.785])),  # above screw 1
    (0.22, np.array([ 0.35,  0.55,  0.0, -0.75,  0.0,  0.85,  0.785])),  # engage screw 1
    (0.42, np.array([ 0.35,  0.56,  0.0, -0.78,  0.0,  0.85,  0.785])),  # drive screw 1
    (0.50, np.array([ 0.35,  0.20,  0.0, -1.50,  0.0,  1.30,  0.785])),  # lift after screw 1
    (0.62, np.array([-0.35,  0.30,  0.0, -1.40,  0.0,  1.20,  0.785])),  # above screw 2
    (0.72, np.array([-0.35,  0.55,  0.0, -0.75,  0.0,  0.85,  0.785])),  # engage screw 2
    (0.92, np.array([-0.35,  0.56,  0.0, -0.78,  0.0,  0.85,  0.785])),  # drive screw 2
    (1.00, _R2_HOME.copy()),                                                # return home
]

# R1 carry window: product is attached to gripper between these phases
_R1_GRASP_PHASE  = 0.25   # grasp pause begins → product leaves conveyor
_R1_PLACE_PHASE  = 0.73   # release pause ends → product deposited at next station

# R2 drive windows: product vibrates during screwdriving contact
_R2_DRIVE_CONTACT = [(0.18, 0.42), (0.64, 0.88)]

# ── Cell B robot keyframes ────────────────────────────────────────────
# R3: electrical functional test — probe two contact points per cycle
_R3_HOME = np.array([ 0.0, -0.785,  0.0, -2.356,  0.0,  1.571,  0.785])
_R3_CYCLE_STEPS = 280
_R3_PHASES = [
    (0.12, np.array([ 0.30,  0.30,  0.0, -1.40,  0.0,  1.20,  0.785])),  # above test-point 1
    (0.22, np.array([ 0.30,  0.55,  0.0, -0.80,  0.0,  0.90,  0.785])),  # contact test-point 1
    (0.40, np.array([ 0.30,  0.55,  0.0, -0.80,  0.0,  0.90,  0.785])),  # hold (measure)
    (0.50, np.array([ 0.30,  0.20,  0.0, -1.50,  0.0,  1.30,  0.785])),  # lift after test 1
    (0.62, np.array([-0.30,  0.30,  0.0, -1.40,  0.0,  1.20,  0.785])),  # above test-point 2
    (0.72, np.array([-0.30,  0.55,  0.0, -0.80,  0.0,  0.90,  0.785])),  # contact test-point 2
    (0.88, np.array([-0.30,  0.55,  0.0, -0.80,  0.0,  0.90,  0.785])),  # hold (measure)
    (1.00, _R3_HOME.copy()),                                                # return home
]

# R4: vacuum suction pick-and-place (pack station)
_R4_HOME = np.array([ 0.0, -0.785,  0.0, -2.356,  0.0,  1.571,  0.785])
_R4_CYCLE_STEPS = 310
_R4_PHASES = [
    (0.15, np.array([ 0.0,   0.40,  0.0, -1.50,  0.0,  2.00,  0.785])),  # extend over conveyor
    (0.28, np.array([ 0.0,   0.60,  0.0, -1.00,  0.0,  1.60,  0.785])),  # descend to pick
    (0.40, np.array([ 0.0,   0.10,  0.0, -1.80,  0.0,  1.80,  0.785])),  # lift with part
    (0.58, np.array([ 0.75,  0.10,  0.0, -1.80,  0.0,  1.80,  0.785])),  # swing to pack side
    (0.72, np.array([ 0.75,  0.40,  0.0, -1.20,  0.0,  1.50,  0.785])),  # lower to pack station
    (0.85, np.array([ 0.40,  0.00,  0.0, -2.00,  0.0,  1.90,  0.785])),  # retract
    (1.00, _R4_HOME.copy()),                                                # return home
]

# ── World-space end-effector trajectories ─────────────────────────────
# Fallback EE positions used when the Franka hand prim can't be read.
# _CONTACT_Z: gripper at product height (product centre z + small clearance)
_CONTACT_Z = _PRODUCT_REST_Z + 0.03   # 0.375 — gripper just above product
_CARRY_Z   = 0.55   # lifted transport height

_R1_EE_HOME = np.array([STATIONS_X[1], 0.20, 0.58])
_R1_EE_PHASES = [
    (0.10, np.array([STATIONS_X[1],  0.05,  0.48])),       # reach over conveyor, lowering
    (0.20, np.array([STATIONS_X[1],  0.00,  _CONTACT_Z])), # descend to product — touching
    (0.25, np.array([STATIONS_X[1],  0.00,  _CONTACT_Z])), # pause at contact (grasp)
    (0.38, np.array([STATIONS_X[1],  0.00,  _CARRY_Z])),   # lift with product
    (0.58, np.array([STATIONS_X[2],  0.00,  _CARRY_Z])),   # swing to next station
    (0.68, np.array([STATIONS_X[2],  0.00,  _CONTACT_Z])), # lower to place height
    (0.73, np.array([STATIONS_X[2],  0.00,  _CONTACT_Z])), # pause at contact (release)
    (0.82, np.array([STATIONS_X[2],  0.15,  0.50])),       # retract upward
    (1.00, _R1_EE_HOME.copy()),                              # home
]

# R2 screwdriving: EE descends to product top and drives
_R2_EE_HOME = np.array([STATIONS_X[3], 0.20, 0.58])
_R2_EE_PHASES = [
    (0.10, np.array([STATIONS_X[3] + 0.06,  0.02,  0.48])),       # above screw 1
    (0.18, np.array([STATIONS_X[3] + 0.06,  0.02,  _CONTACT_Z])), # engage screw 1
    (0.42, np.array([STATIONS_X[3] + 0.06,  0.02,  _CONTACT_Z - 0.01])),  # drive (push in)
    (0.48, np.array([STATIONS_X[3] + 0.06,  0.02,  0.50])),       # lift after screw 1
    (0.56, np.array([STATIONS_X[3] - 0.06,  0.02,  0.48])),       # above screw 2
    (0.64, np.array([STATIONS_X[3] - 0.06,  0.02,  _CONTACT_Z])), # engage screw 2
    (0.88, np.array([STATIONS_X[3] - 0.06,  0.02,  _CONTACT_Z - 0.01])),  # drive
    (0.94, np.array([STATIONS_X[3] - 0.06,  0.02,  0.50])),       # lift
    (1.00, _R2_EE_HOME.copy()),
]

# Shared state written by _animate_robots, read by _advance_products
_r1_ee_pos: np.ndarray = _R1_EE_HOME.copy()
_r2_ee_pos: np.ndarray = _R2_EE_HOME.copy()
_r1_phase_current: float = 0.0
_r2_phase_current: float = 0.0
_r1_held_product: int = -1   # index into products[], -1 = empty gripper

# Smooth sliding speed (fraction of gap closed per frame)
_SLIDE_RATE = 0.06


def _coslerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Cosine-smoothed interpolation; produces ease-in/ease-out between poses."""
    alpha = (1.0 - math.cos(max(0.0, min(t, 1.0)) * math.pi)) * 0.5
    return a * (1.0 - alpha) + b * alpha


def _eval_cycle(start: np.ndarray, phases: list, phase: float) -> np.ndarray:
    """Return joint angles for the given cycle phase [0, 1).

    start  — pose at phase 0
    phases — list of (end_frac, pose); last end_frac must equal 1.0
    """
    prev_end, prev_pose = 0.0, start
    for end_frac, pose in phases:
        if phase <= end_frac:
            seg_t = (phase - prev_end) / max(end_frac - prev_end, 1e-9)
            return _coslerp(prev_pose, pose, seg_t)
        prev_end, prev_pose = end_frac, pose
    return phases[-1][1]


def _animate_robots():
    """Drive robot motion through task-appropriate keyframe cycles.

    Animates arm chain links (Bezier curve shoulder→elbow→EE), gripper palm,
    and finger open/close via set_world_pose every frame.
    Also drives Franka articulation joints when controllers are available.
    """
    steps = sim_state["step_count"]
    r1_speed = 1.0
    r2_speed = 1.0
    if sim_state["schedule"]:
        r1_speed = sim_state["schedule"].get("r1_speed_fraction", 1.0)
        r2_speed = sim_state["schedule"].get("r2_speed_fraction", 1.0)

    # ── R1 pose ──────────────────────────────────────────────────────────
    r1_spd   = r1_speed * _r1_speed_scale
    r1_cycle = max(10, int(_R1_CYCLE_STEPS / max(r1_spd, 0.05)))
    r1_phase = (steps % r1_cycle) / r1_cycle
    r1_pose  = _eval_cycle(_R1_HOME, _R1_PHASES, r1_phase)
    if _r1_amplitude_scale < 1.0:
        r1_pose = _R1_HOME + (r1_pose - _R1_HOME) * _r1_amplitude_scale

    # ── R2 pose ──────────────────────────────────────────────────────────
    r2_cycle = max(10, int(_R2_CYCLE_STEPS / max(r2_speed, 0.05)))
    r2_phase = (steps % r2_cycle) / r2_cycle
    r2_pose  = _eval_cycle(_R2_HOME, _R2_PHASES, r2_phase)

    # ── Articulation joints (set BEFORE reading EE — FK needs fresh joints) ──
    for rname, art in _robot_controllers.items():
        try:
            num_dof = art.num_dof
            if num_dof <= 0:
                continue
            pose = r1_pose if rname == "R1" else r2_pose
            padded = np.zeros(num_dof)
            padded[:min(7, num_dof)] = pose[:min(7, num_dof)]
            art.set_joint_positions(padded)
        except Exception:
            pass

    # ── Update EE positions (prefer actual Franka hand prim) ──────────────
    global _r1_ee_pos, _r2_ee_pos, _r1_phase_current, _r2_phase_current
    _r1_phase_current = r1_phase
    _r2_phase_current = r2_phase

    # Read the real Franka hand position (FK is now current).
    # Fall back to pre-computed world-space EE trajectory.
    ee1 = _read_hand_world_pos("R1")
    if ee1 is None:
        ee1 = _eval_cycle(_R1_EE_HOME, _R1_EE_PHASES, r1_phase)
        if _r1_amplitude_scale < 1.0:
            ee1 = _R1_EE_HOME + (ee1 - _R1_EE_HOME) * _r1_amplitude_scale
    _r1_ee_pos = ee1

    ee2 = _read_hand_world_pos("R2")
    if ee2 is None:
        ee2 = _eval_cycle(_R2_EE_HOME, _R2_EE_PHASES, r2_phase)
    _r2_ee_pos = ee2


def _animate_cellB_robots():
    """Drive Cell B robot joints: R3 (probe test) and R4 (suction pack).

    Uses the same articulation-controller path as _animate_robots().
    Speed can be scaled via Cell B schedule entries r3_speed_fraction /
    r4_speed_fraction (falls back to r2/r1 fractions if absent).
    """
    steps = sim_state["step_count"]
    sched = sim_state.get("schedule") or {}

    r3_spd = sched.get("r3_speed_fraction", sched.get("r2_speed_fraction", 1.0))
    r4_spd = sched.get("r4_speed_fraction", sched.get("r1_speed_fraction", 1.0))

    r3_cycle = max(10, int(_R3_CYCLE_STEPS / max(r3_spd, 0.05)))
    r3_pose  = _eval_cycle(_R3_HOME, _R3_PHASES, (steps % r3_cycle) / r3_cycle)

    r4_cycle = max(10, int(_R4_CYCLE_STEPS / max(r4_spd, 0.05)))
    r4_pose  = _eval_cycle(_R4_HOME, _R4_PHASES, (steps % r4_cycle) / r4_cycle)

    for rname, pose in [("R3", r3_pose), ("R4", r4_pose)]:
        art = _robot_controllers.get(rname)
        if art is None:
            continue
        try:
            ndof = art.num_dof
            if ndof <= 0:
                continue
            padded = np.zeros(ndof)
            padded[:min(7, ndof)] = pose[:min(7, ndof)]
            art.set_joint_positions(padded)
        except Exception:
            pass


def _advance_products():
    """Move products with smooth sliding and proper robot-object contact.

    Products slide smoothly along the conveyor (lerp x-position each frame).
    R1 picks a product up from the conveyor, carries it with the gripper, and
    places it down at the next station.  R2 pushes down slightly on products
    during screwdriving to show contact.  Products rest flush on the conveyor.
    """
    global _last_cycle_step, _units_completed, _r1_held_product

    steps = sim_state["step_count"]
    r1_phase = _r1_phase_current
    r2_phase = _r2_phase_current

    # ── R1 gripper attach / carry / release ─────────────────────────────
    in_carry = _R1_GRASP_PHASE <= r1_phase < _R1_PLACE_PHASE

    if in_carry:
        # Pick up the first product sitting at R1's station (station index 1)
        if _r1_held_product == -1:
            for i in range(NUM_PRODUCTS):
                if _product_station[i] == 1:
                    _r1_held_product = i
                    break

        # Carry: product hangs from the gripper.
        # Offset the product below the EE (hand prim) by a small amount
        # so it appears gripped.  Franka finger length ≈ 0.05 at 0.8 scale.
        if _r1_held_product != -1:
            carry_pos = np.array([
                _r1_ee_pos[0],
                _r1_ee_pos[1],
                _r1_ee_pos[2] - 0.06,
            ])
            _product_pos[_r1_held_product] = carry_pos
            try:
                products[_r1_held_product].set_world_pose(position=carry_pos)
            except Exception:
                pass
    else:
        # Release: advance station and let the product settle on conveyor
        if _r1_held_product != -1:
            idx = _r1_held_product
            next_station = _product_station[idx] + 1
            if next_station >= len(STATIONS_X):
                _product_station[idx] = 0
                _units_completed += 1
                _product_pos[idx] = np.array([STATIONS_X[0], 0.0, _PRODUCT_REST_Z])
            else:
                _product_station[idx] = next_station
                _product_pos[idx] = np.array([STATIONS_X[next_station], 0.0, _PRODUCT_REST_Z])
            _r1_held_product = -1

    # ── Timed station advance for non-held products ─────────────────────
    if steps - _last_cycle_step >= _steps_per_cycle:
        _last_cycle_step = steps
        for i in range(NUM_PRODUCTS):
            if i == _r1_held_product:
                continue
            _product_station[i] += 1
            if _product_station[i] >= len(STATIONS_X):
                _product_station[i] = 0
                _units_completed += 1
                # Teleport back to entry so it slides forward, not backward
                _product_pos[i] = np.array([STATIONS_X[0], 0.0, _PRODUCT_REST_Z])

    # ── Smooth slide every product toward its target station ────────────
    for i in range(NUM_PRODUCTS):
        if i == _r1_held_product:
            continue  # gripper controls this one

        target_x = STATIONS_X[_product_station[i]]
        target   = np.array([target_x, 0.0, _PRODUCT_REST_Z])

        # R2 contact: when product is at station 3 and R2 is in a drive phase,
        # push the product down slightly to show tool pressure
        if _product_station[i] == 3:
            for ds, de in _R2_DRIVE_CONTACT:
                if ds <= r2_phase < de:
                    drive_t = (r2_phase - ds) / (de - ds)
                    # slight downward compression + subtle x vibration
                    target[2] = _PRODUCT_REST_Z - 0.004 * math.sin(drive_t * 12 * math.pi)
                    target[0] += 0.003 * math.sin(drive_t * 20 * math.pi)
                    break

        # Lerp current position toward target (smooth sliding on conveyor)
        _product_pos[i] = _product_pos[i] + (_SLIDE_RATE) * (target - _product_pos[i])

        try:
            products[i].set_world_pose(position=_product_pos[i])
        except Exception:
            pass


# Cached UsdSkel state for the operator (populated on first call)
_operator_skel_cache: dict = {}
# Cached UsdSkel state for H2 operator in Cell B (multi-cell mode)
_h2_skel_cache: dict = {}

def _init_operator_skeleton():
    """One-time: find / create SkelAnimation, bind it, cache rest pose.

    Returns True if skeleton animation is ready, False otherwise.
    """
    from pxr import UsdSkel, Usd, Vt

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/CBPACell/Operator")
    if not root.IsValid():
        logger.warning("Operator: root prim not found")
        return False

    # ── 1. Find the Skeleton prim ───────────────────────────────────────
    skel_prim = None
    skel_root_prim = None
    all_types = []
    for prim in Usd.PrimRange(root):
        tname = prim.GetTypeName()
        all_types.append(f"{prim.GetName()}({tname})")
        if prim.IsA(UsdSkel.Skeleton):
            skel_prim = prim
        if prim.IsA(UsdSkel.Root):
            skel_root_prim = prim

    logger.info(f"Operator prim tree (first 30): {all_types[:30]}")

    if skel_prim is None:
        logger.warning("Operator: no UsdSkel.Skeleton prim found")
        return False

    skel = UsdSkel.Skeleton(skel_prim)
    joints = skel.GetJointsAttr().Get()
    if not joints:
        logger.warning(f"Operator: Skeleton at {skel_prim.GetPath()} has no joints")
        return False

    n = len(joints)
    logger.info(f"Operator: Skeleton at {skel_prim.GetPath()}, {n} joints")

    # Build name → index map (log arm/hand joints)
    jmap = {}
    for i, jpath in enumerate(joints):
        short = str(jpath).rsplit("/", 1)[-1] if "/" in str(jpath) else str(jpath)
        jmap[short] = i
    arm_joints = [j for j in jmap if any(k in j.lower() for k in ("arm", "hand", "shoulder", "elbow", "wrist"))]
    logger.info(f"Operator arm-related joints: {arm_joints}")

    # ── 2. Decompose rest transforms → rotations + translations ─────────
    rest_xforms = skel.GetRestTransformsAttr().Get()
    rest_rots = []
    rest_trans = []
    if rest_xforms and len(rest_xforms) == n:
        for mtx in rest_xforms:
            xf = Gf.Transform(mtx)
            qd = xf.GetRotation().GetQuat()        # Gf.Quatd
            w = float(qd.GetReal())
            im = qd.GetImaginary()
            rest_rots.append(Gf.Quatf(w, float(im[0]), float(im[1]), float(im[2])))
            td = xf.GetTranslation()
            rest_trans.append(Gf.Vec3f(float(td[0]), float(td[1]), float(td[2])))
        logger.info(f"Operator: decomposed {n} rest transforms")
    else:
        logger.warning(f"Operator: restTransforms missing or length mismatch "
                       f"({len(rest_xforms) if rest_xforms else 0} vs {n})")
        rest_rots = [Gf.Quatf(1, 0, 0, 0)] * n
        rest_trans = [Gf.Vec3f(0, 0, 0)] * n

    # ── 3. Find existing SkelAnimation OR create one ────────────────────
    anim = None

    # Check binding first
    binding = UsdSkel.BindingAPI(skel_prim)
    if binding:
        src_rel = binding.GetAnimationSourceRel()
        if src_rel:
            targets = src_rel.GetTargets()
            for t in targets:
                p = stage.GetPrimAtPath(t)
                if p.IsValid() and p.IsA(UsdSkel.Animation):
                    anim = UsdSkel.Animation(p)
                    logger.info(f"Operator: existing bound SkelAnimation at {t}")
                    break

    # Search descendants
    if anim is None:
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdSkel.Animation):
                anim = UsdSkel.Animation(prim)
                logger.info(f"Operator: found unbound SkelAnimation at {prim.GetPath()}")
                break

    # Create if none exists
    if anim is None:
        anim_path = skel_prim.GetPath().AppendChild("CBPAAnim")
        anim = UsdSkel.Animation.Define(stage, anim_path)
        logger.info(f"Operator: created SkelAnimation at {anim_path}")

    # ── 4. Bind animation → skeleton (idempotent) ──────────────────────
    anim_path = anim.GetPrim().GetPath()
    binding = UsdSkel.BindingAPI.Apply(skel_prim)
    binding.CreateAnimationSourceRel().SetTargets([anim_path])

    # Also bind on the SkelRoot if we found one
    if skel_root_prim and skel_root_prim != skel_prim:
        rb = UsdSkel.BindingAPI.Apply(skel_root_prim)
        rb.CreateAnimationSourceRel().SetTargets([anim_path])

    # ── 5. Set animation joints to match skeleton ───────────────────────
    anim.GetJointsAttr().Set(joints)

    # If the anim already had rotations, prefer those as the rest baseline
    existing_rots = anim.GetRotationsAttr().Get()
    if existing_rots and len(existing_rots) == n:
        rest_rots = list(existing_rots)
        logger.info("Operator: using existing anim rotations as rest baseline")

    existing_trans = anim.GetTranslationsAttr().Get()
    if existing_trans and len(existing_trans) == n:
        rest_trans = list(existing_trans)

    # Write the initial rest pose so the anim has valid data
    anim.GetRotationsAttr().Set(Vt.QuatfArray(rest_rots))
    anim.GetTranslationsAttr().Set(Vt.Vec3fArray(rest_trans))
    # Scales: uniform 1
    rest_scales = Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * n)
    anim.GetScalesAttr().Set(rest_scales)

    _operator_skel_cache["anim"] = anim
    _operator_skel_cache["joint_map"] = jmap
    _operator_skel_cache["rest_rots"] = rest_rots
    _operator_skel_cache["rest_trans"] = rest_trans
    _operator_skel_cache["n_joints"] = n
    _operator_skel_cache["root_prim"] = root   # needed to move character each frame
    logger.info("Operator: skeleton animation setup complete")
    return True


def _init_h2_skeleton():
    """One-time: find / create SkelAnimation for Cell B operator H2.

    Identical logic to _init_operator_skeleton() but targets the H2 prim.
    """
    from pxr import UsdSkel, Usd, Vt

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/Factory/CellB/Operator_H2")
    if not root.IsValid():
        logger.warning("H2 Operator: root prim not found")
        return False

    skel_prim = None
    skel_root_prim = None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdSkel.Skeleton):
            skel_prim = prim
        if prim.IsA(UsdSkel.Root):
            skel_root_prim = prim

    if skel_prim is None:
        logger.warning("H2 Operator: no UsdSkel.Skeleton prim found")
        return False

    skel = UsdSkel.Skeleton(skel_prim)
    joints = skel.GetJointsAttr().Get()
    if not joints:
        logger.warning(f"H2 Operator: Skeleton at {skel_prim.GetPath()} has no joints")
        return False

    n = len(joints)
    logger.info(f"H2 Operator: Skeleton at {skel_prim.GetPath()}, {n} joints")

    jmap = {}
    for i, jpath in enumerate(joints):
        short = str(jpath).rsplit("/", 1)[-1] if "/" in str(jpath) else str(jpath)
        jmap[short] = i

    rest_xforms = skel.GetRestTransformsAttr().Get()
    rest_rots = []
    rest_trans = []
    if rest_xforms and len(rest_xforms) == n:
        for mtx in rest_xforms:
            xf = Gf.Transform(mtx)
            qd = xf.GetRotation().GetQuat()
            w = float(qd.GetReal())
            im = qd.GetImaginary()
            rest_rots.append(Gf.Quatf(w, float(im[0]), float(im[1]), float(im[2])))
            td = xf.GetTranslation()
            rest_trans.append(Gf.Vec3f(float(td[0]), float(td[1]), float(td[2])))
        logger.info(f"H2 Operator: decomposed {n} rest transforms")
    else:
        rest_rots = [Gf.Quatf(1, 0, 0, 0)] * n
        rest_trans = [Gf.Vec3f(0, 0, 0)] * n

    anim = None
    binding = UsdSkel.BindingAPI(skel_prim)
    if binding:
        src_rel = binding.GetAnimationSourceRel()
        if src_rel:
            for t in src_rel.GetTargets():
                p = stage.GetPrimAtPath(t)
                if p.IsValid() and p.IsA(UsdSkel.Animation):
                    anim = UsdSkel.Animation(p)
                    break

    if anim is None:
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdSkel.Animation):
                anim = UsdSkel.Animation(prim)
                break

    if anim is None:
        anim_path = skel_prim.GetPath().AppendChild("CBPAAnimH2")
        anim = UsdSkel.Animation.Define(stage, anim_path)
        logger.info(f"H2 Operator: created SkelAnimation at {anim_path}")

    anim_path = anim.GetPrim().GetPath()
    binding = UsdSkel.BindingAPI.Apply(skel_prim)
    binding.CreateAnimationSourceRel().SetTargets([anim_path])
    if skel_root_prim and skel_root_prim != skel_prim:
        rb = UsdSkel.BindingAPI.Apply(skel_root_prim)
        rb.CreateAnimationSourceRel().SetTargets([anim_path])

    anim.GetJointsAttr().Set(joints)

    existing_rots = anim.GetRotationsAttr().Get()
    if existing_rots and len(existing_rots) == n:
        rest_rots = list(existing_rots)
    existing_trans = anim.GetTranslationsAttr().Get()
    if existing_trans and len(existing_trans) == n:
        rest_trans = list(existing_trans)

    anim.GetRotationsAttr().Set(Vt.QuatfArray(rest_rots))
    anim.GetTranslationsAttr().Set(Vt.Vec3fArray(rest_trans))
    anim.GetScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * n))

    _h2_skel_cache["anim"] = anim
    _h2_skel_cache["joint_map"] = jmap
    _h2_skel_cache["rest_rots"] = rest_rots
    _h2_skel_cache["rest_trans"] = rest_trans
    _h2_skel_cache["n_joints"] = n
    _h2_skel_cache["root_prim"] = root
    logger.info("H2 Operator: skeleton animation setup complete")
    return True



def _animate_operator():
    """Animate operator in a simple walk cycle.

    On the first call the skeleton is initialised and all joint names are
    logged so axis conventions can be verified.  Each subsequent frame drives
    a basic walk cycle (thigh/calf stride, arm counter-swing, elbow bend)
    using sinusoidal timing.  Axes are conservative (small angles) so that
    any sign error produces a mild effect rather than severe deformation.
    """
    steps = sim_state["step_count"]

    # ── Skeleton init (once) ────────────────────────────────────────────
    if "init_done" not in _operator_skel_cache:
        _operator_skel_cache["init_done"] = True
        try:
            _init_operator_skeleton()
        except Exception as e:
            logger.warning(f"Operator skel init: {e}")
            _operator_skel_cache["anim"] = None

    anim = _operator_skel_cache.get("anim")
    if anim is None:
        return

    try:
        from pxr import Vt
        rest_rots = _operator_skel_cache["rest_rots"]
        jmap      = _operator_skel_cache["joint_map"]

        if "joints_logged" not in _operator_skel_cache:
            _operator_skel_cache["joints_logged"] = True
            logger.info("OPERATOR JOINTS: " + ", ".join(sorted(jmap.keys())))

        rots = list(rest_rots)

        def _set(jname, q):
            idx = jmap.get(jname)
            if idx is not None:
                rots[idx] = q * rest_rots[idx]

        def qx(a): return Gf.Quatf(math.cos(a/2),  math.sin(a/2), 0, 0)
        def qy(a): return Gf.Quatf(math.cos(a/2), 0,  math.sin(a/2), 0)
        def qz(a): return Gf.Quatf(math.cos(a/2), 0, 0,  math.sin(a/2))

        # ── Advance walk position ────────────────────────────────────────
        global _op_walk_dir
        _op_walk_pos[0] += _op_walk_dir * _OP_WALK_STEP
        if _op_walk_pos[0] >= _OP_WALK_R:
            _op_walk_pos[0] = _OP_WALK_R
            _op_walk_dir = -1
        elif _op_walk_pos[0] <= _OP_WALK_L:
            _op_walk_pos[0] = _OP_WALK_L
            _op_walk_dir = +1

        # Move the root prim — update its TranslateOp and RotateXYZOp each frame.
        # Facing: right (+X) → rotZ=90°, left (−X) → rotZ=−90°.
        root = _operator_skel_cache.get("root_prim")
        if root and root.IsValid():
            facing_z = 90.0 if _op_walk_dir > 0 else -90.0
            for op in UsdGeom.Xformable(root).GetOrderedXformOps():
                ot = op.GetOpType()
                if ot == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(_op_walk_pos[0], _op_walk_pos[1], _op_walk_pos[2]))
                elif ot == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3d(0.0, 0.0, facing_z))

        # ── Walk-cycle joint animation ───────────────────────────────────
        w = 2.0 * math.pi * (steps % _OP_WALK_CYCLE) / _OP_WALK_CYCLE
        sw = math.sin(w)   # −1..+1, drives alternating leg phases

        STRIDE  = math.radians(20)   # max thigh swing
        K_BEND  = math.radians(25)   # max knee flex at mid-swing
        ARM_SW  = math.radians(16)   # arm counter-swing amplitude
        ADDUCT  = math.radians(88)   # arms-down adduction (same as before)

        # Thighs: symmetric swing around neutral (upright).
        # Small FWDBIAS gives a natural slight forward lean without the "sitting" look.
        FWDBIAS  = math.radians(5)
        l_thigh_a = -FWDBIAS + STRIDE * sw
        r_thigh_a = -FWDBIAS - STRIDE * sw
        _set("L_Thigh", qx(l_thigh_a))
        _set("R_Thigh", qx(r_thigh_a))

        # Calves: bend only during swing phase, straight during stance.
        # L swing: w ∈ (π/2 → 3π/2), mid-swing at w=π → use −cos(w), peaks at w=π.
        # R swing: w ∈ (3π/2 → π/2), mid-swing at w=0  → use  cos(w), peaks at w=0.
        l_calf_a = K_BEND * max(0.0, -math.cos(w))
        r_calf_a = K_BEND * max(0.0,  math.cos(w))
        _set("L_Calf", qx(-l_calf_a))
        _set("R_Calf", qx(-r_calf_a))

        # Feet: counter-rotate to cancel the accumulated thigh+calf tilt so
        # the sole stays parallel to the ground instead of floating upward.
        # Foot contact illusion:
        #   sw > 0  → L leg swinging forward → toe-up (dorsiflexion, foot clears ground)
        #   sw < 0  → L leg in stance/push-off → toe-down (plantarflexion, push off ground)
        # The 0.45 factor on chain compensation avoids over-correcting.
        TOE = math.radians(10)
        _set("L_Foot", qx((l_thigh_a + l_calf_a) * 0.45 +  TOE * sw))
        _set("R_Foot", qx((r_thigh_a + r_calf_a) * 0.45 + -TOE * sw))

        # Arms: adducted down + counter-swing.
        # The Mixamo rig mirrors left-side joint axes, so both arms need the
        # SAME qx sign to produce opposite world-space swing.
        # Arms counter-swing: opposite to same-side leg.
        # R arm opposite R leg (+sw): uses +sw ✓
        # L arm opposite L leg (+sw): needs −sw
        _set("R_Upperarm", qy(-ARM_SW * sw) * qz( ADDUCT))
        _set("L_Upperarm", qy(-ARM_SW * sw) * qz(-ADDUCT))

        # Subtle spine sway in sync with stride
        _set("Waist",   qx( 0.008 * sw))
        _set("Spine01", qx( 0.005 * sw))
        _set("Spine02", qx( 0.003 * sw))

        anim.GetRotationsAttr().Set(Vt.QuatfArray(rots))

    except Exception as e:
        logger.debug(f"Operator anim frame: {e}")


def _animate_h2_operator():
    """Animate Cell B operator (H2) with full skeleton walk cycle.

    Mirrors _animate_operator() exactly — same joint math, same walk speed —
    but uses _h2_skel_cache and the H2 prim path.  The stride phase is offset
    by half a cycle so H1 and H2 don't stride in unison visually.
    """
    steps = sim_state["step_count"]

    # ── Skeleton init (once) ────────────────────────────────────────────
    if "init_done" not in _h2_skel_cache:
        _h2_skel_cache["init_done"] = True
        try:
            _init_h2_skeleton()
        except Exception as e:
            logger.warning(f"H2 skel init: {e}")
            _h2_skel_cache["anim"] = None

    anim = _h2_skel_cache.get("anim")

    # ── Advance walk position ────────────────────────────────────────
    global _h2_walk_dir
    _h2_walk_pos[0] += _h2_walk_dir * _H2_WALK_STEP
    if _h2_walk_pos[0] >= _H2_WALK_R:
        _h2_walk_pos[0] = _H2_WALK_R
        _h2_walk_dir = -1
    elif _h2_walk_pos[0] <= _H2_WALK_L:
        _h2_walk_pos[0] = _H2_WALK_L
        _h2_walk_dir = +1

    facing_z = 90.0 if _h2_walk_dir > 0 else -90.0

    # Move root prim (USD character or fallback capsule)
    root = _h2_skel_cache.get("root_prim")
    if root is None or not root.IsValid():
        # Fall back to searching for valid prim (handles capsule fallback too)
        for ppath in ("/World/Factory/CellB/Operator_H2",
                      "/World/Factory/CellB/Operator_H2_fallback"):
            p = world.stage.GetPrimAtPath(ppath)
            if p and p.IsValid():
                root = p
                break
    if root and root.IsValid():
        try:
            for op in UsdGeom.Xformable(root).GetOrderedXformOps():
                ot = op.GetOpType()
                if ot == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(float(_h2_walk_pos[0]),
                                    float(_h2_walk_pos[1]),
                                    float(_h2_walk_pos[2])))
                elif ot == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3d(0.0, 0.0, facing_z))
        except Exception as e:
            logger.debug(f"H2 root move: {e}")

    if anim is None:
        return  # no skeleton — position-only fallback is enough

    try:
        from pxr import Vt
        rest_rots = _h2_skel_cache["rest_rots"]
        jmap      = _h2_skel_cache["joint_map"]

        rots = list(rest_rots)

        def _set(jname, q):
            idx = jmap.get(jname)
            if idx is not None:
                rots[idx] = q * rest_rots[idx]

        def qx(a): return Gf.Quatf(math.cos(a/2),  math.sin(a/2), 0, 0)
        def qy(a): return Gf.Quatf(math.cos(a/2), 0,  math.sin(a/2), 0)
        def qz(a): return Gf.Quatf(math.cos(a/2), 0, 0,  math.sin(a/2))

        # Offset phase by half cycle so H2 strides out of sync with H1
        w = 2.0 * math.pi * ((steps + _H2_WALK_CYCLE // 2) % _H2_WALK_CYCLE) / _H2_WALK_CYCLE
        sw = math.sin(w)

        STRIDE  = math.radians(20)
        K_BEND  = math.radians(25)
        ARM_SW  = math.radians(16)
        ADDUCT  = math.radians(88)
        FWDBIAS = math.radians(5)

        _set("L_Thigh", qx(-FWDBIAS + STRIDE * sw))
        _set("R_Thigh", qx(-FWDBIAS - STRIDE * sw))

        _set("L_Calf", qx(-K_BEND * max(0.0, -math.cos(w))))
        _set("R_Calf", qx(-K_BEND * max(0.0,  math.cos(w))))

        l_thigh_a = -FWDBIAS + STRIDE * sw
        r_thigh_a = -FWDBIAS - STRIDE * sw
        l_calf_a  = K_BEND * max(0.0, -math.cos(w))
        r_calf_a  = K_BEND * max(0.0,  math.cos(w))
        TOE = math.radians(10)
        _set("L_Foot", qx((l_thigh_a + l_calf_a) * 0.45 +  TOE * sw))
        _set("R_Foot", qx((r_thigh_a + r_calf_a) * 0.45 + -TOE * sw))

        _set("R_Upperarm", qy(-ARM_SW * sw) * qz( ADDUCT))
        _set("L_Upperarm", qy(-ARM_SW * sw) * qz(-ADDUCT))

        _set("Waist",   qx( 0.008 * sw))
        _set("Spine01", qx( 0.005 * sw))
        _set("Spine02", qx( 0.003 * sw))

        anim.GetRotationsAttr().Set(Vt.QuatfArray(rots))

    except Exception as e:
        logger.debug(f"H2 anim frame: {e}")


# ── Operator walk state ──────────────────────────────────────────────
# Patrol path: walk back and forth in front of the conveyor (stations 1–3).
_op_walk_pos  = np.array([STATIONS_X[2], -0.45, 0.0], dtype=float)
_op_walk_dir  = 1          # +1 = walking right (+X), -1 = left (−X)
_OP_WALK_R    = STATIONS_X[3] - 0.25   # right turn-around point
_OP_WALK_L    = STATIONS_X[1] + 0.25   # left  turn-around point
_OP_WALK_STEP = 0.006      # world-units per sim step ≈ 0.36 m/s at 60 Hz
_OP_WALK_CYCLE = 48        # sim steps per complete stride

# ── H2 operator walk state (Cell B, multi-cell mode) ─────────────────
# H2 patrols the same x-range but at Cell B's y position.
_h2_walk_pos  = np.array([STATIONS_X[2], _CELL_B_Y_OFFSET - 0.45, 0.0], dtype=float)
_h2_walk_dir  = -1         # start walking left (opposite phase to H1)
_H2_WALK_R    = STATIONS_X[3] - 0.25
_H2_WALK_L    = STATIONS_X[1] + 0.25
_H2_WALK_STEP = 0.006
_H2_WALK_CYCLE = 52        # slightly different stride so they don't sync visually

# ── Simulation state ─────────────────────────────────────────────────
import queue as _queue

# Step requests are enqueued by the HTTP handler and processed by
# the main thread (Isaac Sim requires main-thread physics calls).
_step_queue: _queue.Queue = _queue.Queue()

sim_state = {
    "running": False,
    "step_count": 0,
    "schedule": None,
    "constraints": {},
    "sensor_data": {
        "robot_r1": {"speed_mps": 0.0, "force_n": 0.0, "joint_positions": [0.0] * 6},
        "robot_r2": {"speed_mps": 0.0, "force_n": 0.0, "joint_positions": [0.0] * 6},
        "acoustic": {"noise_db": 65.0},
        "human_operator": {"fatigue_estimate": 0.0, "posture_score": 1.0},
    },
}


def update_sensor_data():
    """Update simulated sensor readings and animate the scene."""
    import random

    base_noise = 65.0
    speed_factor = 1.0
    if sim_state["schedule"]:
        speed_factor = sim_state["schedule"].get("r1_speed_fraction", 1.0)

    sim_state["sensor_data"]["robot_r1"]["speed_mps"] = 1.5 * speed_factor * _r1_speed_scale
    sim_state["sensor_data"]["robot_r1"]["force_n"] = 10.0 + random.gauss(0, 0.5)

    r2_speed = sim_state["schedule"].get("r2_speed_fraction", 1.0) if sim_state["schedule"] else 1.0
    sim_state["sensor_data"]["robot_r2"]["speed_mps"] = 1.2 * r2_speed
    sim_state["sensor_data"]["robot_r2"]["force_n"] = 8.0 + random.gauss(0, 0.3)

    # Noise increases with robot speed
    sim_state["sensor_data"]["acoustic"]["noise_db"] = (
        base_noise + 10 * speed_factor + random.gauss(0, 1.0)
    )

    # Fatigue accumulates over steps
    steps = sim_state["step_count"]
    sim_state["sensor_data"]["human_operator"]["fatigue_estimate"] = min(
        1.0, 0.05 + steps * 0.0001
    )
    sim_state["sensor_data"]["human_operator"]["posture_score"] = max(
        0.5, 1.0 - steps * 0.00005
    )

    # ── Visual animation ──
    _advance_products()
    _animate_robots()
    _animate_operator()

    # Multi-cell: animate Cell B robots, products, AGV, and H2 operator
    if _MULTI_CELL:
        _animate_cellB_robots()
        _advance_cellB_products()
        _animate_agv()
        _animate_h2_operator()


# ── Cell B product animation (multi-cell mode) ─────────────────────
_cellB_last_cycle_step = 0

def _advance_cellB_products():
    """Advance Cell B products through stations (simplified version)."""
    global _cellB_units_completed, _cellB_last_cycle_step
    if not _cellB_products:
        return

    steps = sim_state["step_count"]
    # Advance products at Cell B's pace
    if steps - _cellB_last_cycle_step >= _cellB_steps_per_cycle:
        _cellB_last_cycle_step = steps
        for i in range(len(_cellB_products)):
            _cellB_product_station[i] += 1
            if _cellB_product_station[i] >= len(STATIONS_X):
                _cellB_product_station[i] = 0
                _cellB_units_completed += 1
                _cellB_product_pos[i] = np.array([STATIONS_X[0], _CELL_B_Y_OFFSET, _PRODUCT_REST_Z])

    # Smooth sliding toward target station
    for i in range(len(_cellB_products)):
        target_x = STATIONS_X[_cellB_product_station[i]]
        _cellB_product_pos[i][0] += (_SLIDE_RATE) * (target_x - _cellB_product_pos[i][0])
        try:
            _cellB_products[i].set_world_pose(
                position=_cellB_product_pos[i],
            )
        except Exception:
            pass


def _animate_agv():
    """Animate the AGV sliding back and forth in the corridor."""
    global _agv_dir
    _agv_pos[0] += _agv_dir * _AGV_SPEED
    if _agv_pos[0] >= _AGV_RIGHT:
        _agv_pos[0] = _AGV_RIGHT
        _agv_dir = -1
    elif _agv_pos[0] <= _AGV_LEFT:
        _agv_pos[0] = _AGV_LEFT
        _agv_dir = 1

    _agv_y = _CELL_B_Y_OFFSET / 2  # corridor y
    agv_prim = world.stage.GetPrimAtPath("/World/Factory/AGV/Vehicle")
    if agv_prim and agv_prim.IsValid():
        for op in UsdGeom.Xformable(agv_prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(_agv_pos[0]), _agv_y, 0.08))
                break


# ── REST API server ──────────────────────────────────────────────────

class CBPAHandler(BaseHTTPRequestHandler):
    """Simple REST API for the CBPA IsaacBridge remote mode."""

    def log_message(self, format, *a):
        logger.debug(format % a)

    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        if self.path == "/status":
            resp = {
                "status": "running" if sim_state["running"] else "idle",
                "step_count": sim_state["step_count"],
                "scene": "/World/CBPACell",
                "multi_cell": _MULTI_CELL,
            }
            self._json_response(resp)
        elif self.path == "/scene/sensors" or self.path == "/scene/cellA/sensors":
            update_sensor_data()
            data = dict(sim_state["sensor_data"])
            data["production"] = {
                "units_completed": _units_completed,
                "product_stations": list(_product_station),
            }
            self._json_response(data)
        elif self.path == "/scene/cellB/sensors" and _MULTI_CELL:
            update_sensor_data()
            self._json_response({
                "robot_r3": {"speed_mps": 1.2 * 0.6, "force_n": 9.0},
                "robot_r4": {"speed_mps": 1.0 * 0.55, "force_n": 7.0},
                "acoustic": {"noise_db": sim_state["sensor_data"]["acoustic"]["noise_db"] - 2.0},
                "human_operator_h2": {
                    "fatigue_estimate": min(1.0, 0.04 + sim_state["step_count"] * 0.00009),
                    "posture_score": max(0.5, 1.0 - sim_state["step_count"] * 0.00004),
                },
                "production": {
                    "units_completed": _cellB_units_completed,
                    "product_stations": list(_cellB_product_station),
                },
            })
        elif self.path == "/scene/agv/status" and _MULTI_CELL:
            self._json_response({
                "position": [float(_agv_pos[0]), float(_CELL_B_Y_OFFSET / 2), 0.08],
                "direction": _agv_dir,
                "carrying": _agv_carrying,
            })
        elif self.path == "/scene/factory/sensors" and _MULTI_CELL:
            # Combined factory sensor summary
            update_sensor_data()
            cellA_noise = sim_state["sensor_data"]["acoustic"]["noise_db"]
            cellB_noise = cellA_noise - 2.0
            import math as _math
            factory_noise = 10.0 * _math.log10(10**(cellA_noise/10) + 10**(cellB_noise/10))
            self._json_response({
                "factory_noise_db": round(factory_noise, 1),
                "cellA_units": _units_completed,
                "cellB_units": _cellB_units_completed,
                "total_units": _units_completed + _cellB_units_completed,
                "agv_position": float(_agv_pos[0]),
            })
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        global _steps_per_cycle, _cellB_steps_per_cycle
        global _r1_amplitude_scale, _r1_speed_scale
        body = self._read_body()

        if self.path == "/scene/deploy":
            sim_state["schedule"] = body
            sim_state["running"] = True
            # Recalculate cycle speed from deployed schedule
            avg_speed = (body.get("r1_speed_fraction", 1.0) + body.get("r2_speed_fraction", 1.0)) / 2.0
            _steps_per_cycle = max(10, int(50 / max(avg_speed, 0.1)))
            logger.info(f"Schedule deployed: {body.get('schedule_name', 'unknown')} "
                        f"(steps_per_cycle={_steps_per_cycle})")
            self._json_response({
                "scene_path": "/World/CBPACell",
                "schedule_name": body.get("schedule_name", "unknown"),
                "deployed": True,
                "stub": False,
            })

        elif self.path == "/scene/cellA/deploy":
            # Alias for /scene/deploy (Cell A)
            sim_state["schedule"] = body
            sim_state["running"] = True
            avg_speed = (body.get("r1_speed_fraction", 1.0) + body.get("r2_speed_fraction", 1.0)) / 2.0
            _steps_per_cycle = max(10, int(50 / max(avg_speed, 0.1)))
            self._json_response({"deployed": True, "cell": "A"})

        elif self.path == "/scene/cellB/deploy" and _MULTI_CELL:
            avg_speed = (body.get("r1_speed_fraction", 0.6) + body.get("r2_speed_fraction", 0.55)) / 2.0
            _cellB_steps_per_cycle = max(10, int(55 / max(avg_speed, 0.1)))
            logger.info(f"Cell B schedule deployed (steps_per_cycle={_cellB_steps_per_cycle})")
            self._json_response({"deployed": True, "cell": "B",
                                 "steps_per_cycle": _cellB_steps_per_cycle})

        elif self.path == "/scene/verify":
            constraints = body.get("constraints", [])
            results = []
            update_sensor_data()
            for c in constraints:
                name = c.get("name", "")
                limit = c.get("limit", 0)
                # Use actual sensor data for verification
                if "fatigue" in name.lower():
                    sim_val = sim_state["sensor_data"]["human_operator"]["fatigue_estimate"]
                elif "noise" in name.lower():
                    sim_val = sim_state["sensor_data"]["acoustic"]["noise_db"]
                else:
                    sim_val = limit * 0.85
                passed = sim_val <= limit if limit > 0 else True
                results.append({
                    "constraint": name,
                    "limit": limit,
                    "sim_value": round(sim_val, 3),
                    "passed": passed,
                    "margin_pct": round((1 - sim_val / limit) * 100, 1) if limit else 0,
                })
            self._json_response({
                "verification_results": results,
                "all_passed": all(r["passed"] for r in results),
                "sim_steps": sim_state["step_count"],
                "stub": False,
            })

        elif self.path == "/scene/disturbance":
            dtype = body.get("type", "")
            if dtype == "demand_surge":
                magnitude_pct = body.get("magnitude_pct", 20.0)
                _steps_per_cycle = max(5, int(_steps_per_cycle * (1 - magnitude_pct / 100.0)))
                logger.info(f"Disturbance demand_surge: steps_per_cycle={_steps_per_cycle}")
                self._json_response({"applied": True, "type": dtype, "steps_per_cycle": _steps_per_cycle})
            elif dtype == "r1_degradation":
                fraction = body.get("fraction", 1.0)
                _r1_amplitude_scale = fraction
                _r1_speed_scale = fraction
                logger.info(f"Disturbance r1_degradation: scale={fraction}")
                self._json_response({"applied": True, "type": dtype,
                                     "r1_amplitude_scale": _r1_amplitude_scale,
                                     "r1_speed_scale": _r1_speed_scale})
            else:
                self._json_response({"applied": False, "error": f"unknown disturbance: {dtype}"}, 400)

        elif self.path == "/scene/guard_action":
            action = body.get("action", "")
            if action == "clamp_speed":
                reduction = body.get("reduction_fraction", 0.1)
                if sim_state["schedule"]:
                    for key in ("r1_speed_fraction", "r2_speed_fraction"):
                        old = sim_state["schedule"].get(key, 1.0)
                        sim_state["schedule"][key] = old * (1 - reduction)
                avg_speed = (sim_state["schedule"].get("r1_speed_fraction", 1.0)
                             + sim_state["schedule"].get("r2_speed_fraction", 1.0)) / 2.0
                _steps_per_cycle = max(10, int(50 / max(avg_speed, 0.1)))
                logger.info(f"Guard clamp_speed: reduction={reduction}, steps_per_cycle={_steps_per_cycle}")
                self._json_response({"applied": True, "action": action,
                                     "steps_per_cycle": _steps_per_cycle})
            elif action == "clamp_noise":
                reduction = body.get("reduction_fraction", 0.1)
                if sim_state["schedule"]:
                    for key in ("r1_speed_fraction", "r2_speed_fraction"):
                        old = sim_state["schedule"].get(key, 1.0)
                        sim_state["schedule"][key] = old * (1 - reduction)
                logger.info(f"Guard clamp_noise: reduction={reduction}")
                self._json_response({"applied": True, "action": action})
            else:
                self._json_response({"applied": False, "error": f"unknown action: {action}"}, 400)

        elif self.path == "/scene/step":
            num_steps = body.get("num_steps", 100)
            num_steps = min(num_steps, 10000)
            # Enqueue the request — main thread will process it
            result_event = threading.Event()
            result_holder: list[dict] = []
            _step_queue.put((num_steps, result_event, result_holder))
            # Wait for main thread to complete the steps (max 60s)
            if result_event.wait(timeout=60.0):
                self._json_response(result_holder[0] if result_holder else {
                    "steps_completed": 0, "error": "no result",
                })
            else:
                self._json_response({
                    "steps_completed": 0, "error": "step timeout",
                }, 504)
        else:
            self._json_response({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_api_server(port: int):
    server = HTTPServer(("0.0.0.0", port), CBPAHandler)
    logger.info(f"REST API server listening on port {port}")
    server.serve_forever()


# ── Main loop ────────────────────────────────────────────────────────
world.reset()

# Initialize robot articulation controllers (must be after world.reset)
_init_robot_controllers()

# Start REST API in background thread
api_thread = threading.Thread(target=run_api_server, args=(args.port,), daemon=True)
api_thread.start()

logger.info(f"CBPA Isaac Sim scene ready. REST API at http://localhost:{args.port}")
logger.info("Connect CBPA dashboard with: Isaac Sim URL = http://localhost:8211")
logger.info("Press Ctrl+C to stop.")

try:
    import time as _time
    while sim_app.is_running():
        # Process pending step requests from the HTTP handler thread.
        # Physics must be stepped from the main thread.
        try:
            num_steps, result_event, result_holder = _step_queue.get_nowait()
            for _ in range(num_steps):
                world.step(render=False)
                sim_state["step_count"] += 1
            update_sensor_data()

            # Render the scene after stepping so the viewport shows movement
            if not config["headless"]:
                sim_app.update()

            result_holder.append({
                "steps_completed": num_steps,
                "sim_time_s": sim_state["step_count"] * world.get_physics_dt(),
                "units_completed": _units_completed,
                "stub": False,
            })
            result_event.set()
        except _queue.Empty:
            # No pending HTTP request — still advance one physics frame so
            # continuous animations (robots, operator, products) stay live.
            world.step(render=False)
            sim_state["step_count"] += 1
            update_sensor_data()

        # Keep the app alive; render idle frames for viewport responsiveness
        if not config["headless"]:
            sim_app.update()
        _time.sleep(0.01)  # 100 Hz poll
except KeyboardInterrupt:
    pass
finally:
    logger.info("Shutting down Isaac Sim...")
    sim_app.close()
