def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


import numpy as np
from collections import deque

#Supporting Functions
NBRS6 = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _flood_fill_outside_to_neg1(vv):
    X, Y, Z = vv.shape
    q = deque()

    for x in [0, X - 1]:
        for y in range(Y):
            for z in range(Z):
                q.append((x, y, z))
    for x in range(X):
        for y in [0, Y - 1]:
            for z in range(Z):
                q.append((x, y, z))
    for x in range(X):
        for y in range(Y):
            for z in [0, Z - 1]:
                q.append((x, y, z))

    checked = np.zeros(vv.shape, dtype=bool)

    while q:
        x, y, z = q.popleft()
        if checked[x, y, z]:
            continue
        checked[x, y, z] = True

        if vv[x, y, z] == 0:
            vv[x, y, z] = -1

        if vv[x, y, z] == -1:
            for dx, dy, dz in NBRS6:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z and not checked[nx, ny, nz]:
                    if vv[nx, ny, nz] in (-1, 0):
                        q.append((nx, ny, nz))

    return vv


def _collect_connected_zero_region(enclosed_mask, start):
    X, Y, Z = enclosed_mask.shape
    q = deque([start])
    checked_local = {start}
    region = []

    while q:
        x, y, z = q.popleft()
        region.append((x, y, z))

        for dx, dy, dz in NBRS6:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                if enclosed_mask[nx, ny, nz] and (nx, ny, nz) not in checked_local:
                    checked_local.add((nx, ny, nz))
                    q.append((nx, ny, nz))

    return region


def _validate_and_build_void_shell(region, datacube, voidvolume_bg):
    """
    region: connected enclosed 0-region
    datacube: original 0/1 cube
    voidvolume_bg: cropped flood-filled cube
                   -1 = true outside
                    0 = enclosed zero region
                    1 = disturbed/object

    A region is valid only if:
      1. it has an immediate surrounding shell of 1s
      2. that shell does not touch true outside (-1)

    This enforces:
      - no VOID-VOLUME without VOID
      - no VOID on the outside surface
    """
    X, Y, Z = datacube.shape
    shell1 = set()

    # Build immediate shell around the 0-region
    for x, y, z in region:
        for dx, dy, dz in NBRS6:
            nx, ny, nz = x + dx, y + dy, z + dz

            if not (0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z):
                return False, set()

            if datacube[nx, ny, nz] == 1:
                shell1.add((nx, ny, nz))

    if len(shell1) == 0:
        return False, set()

    # Candidate VOID shell must itself be internal, not exposed to outside
    for x, y, z in shell1:
        for dx, dy, dz in NBRS6:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                if voidvolume_bg[nx, ny, nz] == -1:
                    return False, set()

    return True, shell1


def _find_valid_void_regions(datacube):
    """
    Returns:
      valid_void_mask: bool mask for VOID-VOLUME voxels
      void_shell_mask: bool mask for immediate valid VOID shell voxels
      voidvolume_bg: flood-filled background cube cropped to original size
    """
    dimx, dimy, dimz = datacube.shape

    # padded background cube for flood fill
    lrgdatacube = np.zeros((dimx + 2, dimy + 2, dimz + 2), dtype=int)
    lrgdatacube2 = lrgdatacube - 1
    lrgdatacube[1:-1, 1:-1, 1:-1] = datacube
    lrgdatacube2[1:-1, 1:-1, 1:-1] = datacube

    # flood fill true outside
    voidvolume = _flood_fill_outside_to_neg1(lrgdatacube2.copy())
    voidvolume = voidvolume[1:-1, 1:-1, 1:-1]

    enclosed_mask = (voidvolume == 0)

    valid_void_mask = np.zeros_like(enclosed_mask, dtype=bool)
    void_shell_mask = np.zeros_like(enclosed_mask, dtype=bool)
    checked = np.zeros_like(enclosed_mask, dtype=bool)

    starts = np.argwhere(enclosed_mask)
    for sx, sy, sz in starts:
        if checked[sx, sy, sz]:
            continue

        region = _collect_connected_zero_region(enclosed_mask, (sx, sy, sz))
        for rx, ry, rz in region:
            checked[rx, ry, rz] = True

        is_valid, shell1 = _validate_and_build_void_shell(region, datacube, voidvolume)

        if is_valid:
            for rx, ry, rz in region:
                valid_void_mask[rx, ry, rz] = True
            for vx, vy, vz in shell1:
                void_shell_mask[vx, vy, vz] = True

    return valid_void_mask, void_shell_mask, voidvolume


def morph3d(
    DATACUBE=None,
    VERBOSE=False,
    PLOT=False,
    FINALPLOT=True,
    PLOTIDS=None,
    AUTOSAVE=True,
    SAVE_DIR="morph_outputs",
    SAVE_PREFIX="morph3d",
    SAVE_DPI=300,
    YEARS=None,
    SITE_ID="site",
    SAVE_MORPH_ARRAY=True,
):
    
    import os
    import time
    import psutil
    import igraph as ig
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import pandas as pd
    import networkx as nx
    from pathlib import Path
    from datetime import datetime
    from itertools import product
    from .utils import morph3dlinks, morph3dplot, morph3dprep, validate_data_cube, arraytrim

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if AUTOSAVE:
        run_folder = Path(SAVE_DIR) / f"{SAVE_PREFIX}_{SITE_ID}_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)
    else:
        run_folder = None
    def _savepath(name: str):
        if not AUTOSAVE or run_folder is None:
            return None
        return str(run_folder / f"{SAVE_PREFIX}_{SITE_ID}_{name}_{timestamp}.svg")

    _proc = psutil.Process(os.getpid())
    _start_ts = time.time()
    _start_ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_start_ts))
    _start_rss = _proc.memory_info().rss
    try:
        _peak = getattr(_proc.memory_full_info(), "peak_wset", None) or getattr(
            _proc.memory_full_info(), "peak_rss", None
        )
    except Exception:
        _peak = None

    valid_data = validate_data_cube(datacube=DATACUBE)
    if not valid_data:
        print("ERROR: Invalid Data Cube Provided")

    if VERBOSE:
        print("\nInsetting 3D data cube into a larger array to handle edge effects")

    dimdatacube = DATACUBE.shape

    lrgdatacube = np.zeros(
        (dimdatacube[0] + 2, dimdatacube[1] + 2, dimdatacube[2] + 2), dtype=int
    )
    lrgdatacube2 = lrgdatacube - 1
    lrgdatacube[1:dimdatacube[0] + 1, 1:dimdatacube[1] + 1, 1:dimdatacube[2] + 1] = DATACUBE
    lrgdatacube2[1:dimdatacube[0] + 1, 1:dimdatacube[1] + 1, 1:dimdatacube[2] + 1] = DATACUBE

    if VERBOSE:
        print("\n\n    Performing initializations of: voxelID, objectID, coreCode, and morphCode arrays")

    voxelID = np.arange(1, np.prod(DATACUBE.shape) + 1).reshape(DATACUBE.shape)
    objectID = np.zeros_like(voxelID)
    coreCode = np.zeros_like(objectID)
    expandedCoreCode = np.copy(coreCode)
    morphCode = np.zeros_like(objectID)

    if VERBOSE:
        print("\n   Computing neighbours")

    neighbour = morph3dlinks(VOLOBJ=lrgdatacube, VOXELIDS=voxelID)
    neighbourtab = neighbour.iloc[:, 1:8].copy()

    goodIDs = voxelID[DATACUBE == 1]
    for row in range(neighbourtab.shape[0]):
        neighbourtab.iloc[row, ~neighbourtab.iloc[row].isin(goodIDs)] = 0

    newneighbourtab = neighbourtab.iloc[:, 0].to_frame()
    newneighbourtab = pd.concat([newneighbourtab] * 6, ignore_index=True)
    vectorised_data = neighbourtab.iloc[:, 1:7].values.flatten("F")
    newneighbourtab = pd.concat(
        [newneighbourtab, pd.Series(vectorised_data, name="Flatten_data")], axis=1
    )

    sorted_rows = np.apply_along_axis(np.sort, axis=1, arr=newneighbourtab)
    unique_rows_mask = ~pd.DataFrame(sorted_rows).duplicated(keep="first")
    newneighbourtab = newneighbourtab[unique_rows_mask]

    if VERBOSE:
        print("\n  Generating network graph object")

    import networkx as nx
    maingraph = nx.from_pandas_edgelist(
        newneighbourtab,
        source="VOXELIDS",
        target="Flatten_data",
        create_using=nx.Graph(),
    )

    if 0 in maingraph.nodes():
        cutgraph = maingraph.copy()
        cutgraph.remove_node(0)
    else:
        cutgraph = maingraph.copy()

    connected_components = nx.connected_components(cutgraph)
    decompgraph = [cutgraph.subgraph(component).copy() for component in connected_components]

    if VERBOSE:
        print("\n  There are", len(decompgraph), "discrete objects in this graph")
        print("\n  Initiating the 3D morphological segmentation")

    for uq, cluster in enumerate(decompgraph):
        b = list(cluster)
        numvoxels = len(b)
        for vox in range(numvoxels):
            objectID[voxelID == b[vox]] = uq + 1

    if PLOT:
        objmask = np.where(objectID > 0, 10, 0).astype(np.uint8)
        morphs = morph3dprep(objmask)
        morph3dplot(
            morphs,
            plot_codes=[10],
            show_skin=False,
            show_outside=False,
            CELLID=None if not PLOTIDS else PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="Objects IDs",
            save_path=_savepath("ObjectMask"),
            save_dpi=SAVE_DPI,
        )

    # ------------ MASS (CODE = 2) ------------
    if VERBOSE:
        print("\n\n  Identifying MASS voxels")

    for uq, g in enumerate(decompgraph, start=1):
        deg = dict(g.degree())
        sel = [node for node, d in deg.items() if d == 6]

        if VERBOSE:
            print(f"\n    Processing unique object {uq} of {len(decompgraph)}")
            if sel:
                print(f"      There are {len(sel)} MASS voxels in this unique object")
            else:
                print("      There are no MASS voxels in this unique object")

        for node in sel:
            morphCode[voxelID == node] = 2

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 2] = 2
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="MASS Voxels",
            save_path=_savepath("MASS"),
            save_dpi=SAVE_DPI,
        )

    # ------------ SKIN (CODE = 3) ------------
    if VERBOSE:
        print("\n\n  Identifying SKIN voxels")

    for components in nx.connected_components(cutgraph):
        subgraph = cutgraph.subgraph(components)
        component_degree = dict(subgraph.degree())
        massIDS = [key for key, value in component_degree.items() if value == 6]
        numMASS = len(massIDS)

        if numMASS > 0:
            for massID in massIDS:
                MASSneigh = list(subgraph.neighbors(massID))
                MASSneigh = [neighbor for neighbor in MASSneigh if neighbor not in massIDS]

                if MASSneigh:
                    for newrep in range(len(MASSneigh)):
                        morphCode[voxelID == MASSneigh[newrep]] = 3

    # ------------ CRUMB (CODE = 4) ------------
    allMASS = voxelID[morphCode == 2]
    allSKIN = voxelID[morphCode == 3]

    if VERBOSE:
        print("\n\n  Identifying CRUMB voxels")

    for uq in range(len(decompgraph)):
        print(f"Processing unique object {uq+1} of {len(decompgraph)}")
        if len(decompgraph[uq]) == 1:
            print("Object has only 1 voxel, thus it is a CRUMB")
            morphCode[voxelID == list(decompgraph[uq])[0]] = 4
        else:
            curVoxels = list(decompgraph[uq])
            print(f"object has more than 1 voxels. It has: {len(curVoxels)}")
            if any(vox in allMASS for vox in curVoxels):
                print("Not a CRUMB since it connects to a MASS")
            else:
                if any(vox in allSKIN for vox in curVoxels):
                    print("Not a CRUMB since it connects to a SKIN")
                else:
                    print(f"There are {len(curVoxels)} CRUMB voxels in this object")
                    for rep in range(len(curVoxels)):
                        morphCode[voxelID == curVoxels[rep]] = 4

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 4] = 4
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="CRUMB Voxels",
            save_path=_savepath("CRUMB"),
            save_dpi=SAVE_DPI,
        )

    # ------------ CONNECTOR (CODE = 5) ------------
    if VERBOSE:
        print("\n\n  Identifying CONNECTOR voxels")

    morphCode[(DATACUBE > 0) & (morphCode == 0)] = 5
    morphCode[morphCode == 0] = 1

    if VERBOSE:
        print(f"\n    There are {len(voxelID[morphCode == 5])} generic CONNECTOR voxels")
        print("\n\n  Identifying OUTSIDE voxels")
        print(f"\n    There are {len(voxelID[morphCode == 1])} OUTSIDE voxels")

    # ------------ ANTENNA (CODE = 6) ------------
    if VERBOSE:
        print("\n\n  Identifying ANTENNA voxels")

    MASSVoxels = voxelID[morphCode == 2]
    SKINVoxels = voxelID[morphCode == 3]
    CONNECTORVoxels = voxelID[morphCode == 5]

    for obj in range(len(decompgraph)):
        if VERBOSE:
            print(f"\n    Processing object {obj + 1} of {len(decompgraph)} in gph.")

        MassVoxelsInObject = [v for v in MASSVoxels if v in decompgraph[obj]]
        SkinVoxelsInObject = [v for v in SKINVoxels if v in decompgraph[obj]]
        ConnectorVoxelsInObject = [v for v in CONNECTORVoxels if v in decompgraph[obj]]

        if VERBOSE:
            print("\n    MassVoxelsInObject:", MassVoxelsInObject)
            print("      SkinVoxelsInObject:", SkinVoxelsInObject)
            print("      ConnectorVoxelsInObject:", ConnectorVoxelsInObject)

        if len(MassVoxelsInObject) > 0:
            if VERBOSE:
                print(f"\n      Object {obj + 1}: There are MASS and SKIN voxels to delete")

            gph2 = decompgraph[obj].copy()
            gph2.remove_nodes_from(MassVoxelsInObject)
            gph3 = gph2.copy()
            gph3.remove_nodes_from(SkinVoxelsInObject)

            if VERBOSE:
                print("\n      Decomposing...")

            gph4 = [gph3.subgraph(c).copy() for c in nx.connected_components(gph3)]
            numSubparts = len(gph4)

            if VERBOSE:
                print(f"\n      The decomposed graph now has {numSubparts} sub parts")

            for connector, subgraph in enumerate(gph4):
                if VERBOSE:
                    print(f"\n\n        On sub part {connector + 1} of {numSubparts}")

                ConnectorVoxelsInSubGraph = [v for v in CONNECTORVoxels if v in subgraph.nodes]

                if VERBOSE:
                    print(f"           Vertices: {list(subgraph.nodes)}")

                if len(gph4[connector].nodes) == 1:
                    if VERBOSE:
                        print("\n          Single voxel antenna")

                    voxID = int(next(iter(gph4[connector].nodes)))
                    locco = np.argwhere(voxelID == voxID)
                    morphCode[tuple(locco[0])] = 6

                    loc = np.argwhere(voxelID == voxID)
                    counter = 0
                    directions = NBRS6

                    for dx, dy, dz in directions:
                        neighbor_loc = (loc[0][0] + dx, loc[0][1] + dy, loc[0][2] + dz)
                        if (
                            0 <= neighbor_loc[0] < dimdatacube[0]
                            and 0 <= neighbor_loc[1] < dimdatacube[1]
                            and 0 <= neighbor_loc[2] < dimdatacube[2]
                        ):
                            neighbor_val = morphCode[neighbor_loc]
                            if neighbor_val == 2:
                                counter += 1

                    if counter == 1:
                        morphCode[voxelID == ConnectorVoxelsInSubGraph] = 6
                    else:
                        morphCode[voxelID == ConnectorVoxelsInSubGraph] = 5

                    checkcounter = 0
                    for dx, dy, dz in directions:
                        neighbor_loc = (loc[0][0] + dx, loc[0][1] + dy, loc[0][2] + dz)
                        if (
                            0 <= neighbor_loc[0] < dimdatacube[0]
                            and 0 <= neighbor_loc[1] < dimdatacube[1]
                            and 0 <= neighbor_loc[2] < dimdatacube[2]
                        ):
                            neighbor_val = morphCode[neighbor_loc]
                            if neighbor_val == 3:
                                checkcounter += 1

                    if checkcounter == 1:
                        morphCode[tuple(locco[0])] = 6

            if VERBOSE:
                print("\n      Done with object", obj + 1)

        else:
            if VERBOSE:
                print("\n      There are no MASS and SKIN voxels; therefore there are no possible connectors to process")
                print("\n      Done with object", obj + 1)

    if VERBOSE:
        print("\n\n  Finished identifying ANTENNA voxels\n")

    # ------------ BOND (CODE = 7) ------------
    if VERBOSE:
        print("\n\n  Identifying BOND voxels")
        print("\n    Identifying unique cores of MASS voxels")

    for _, obj_graph in enumerate(decompgraph):
        MASSVoxels = voxelID[morphCode == 2]
        MassVoxelsInObj = [v for v in MASSVoxels if v in obj_graph.nodes]

        if len(MassVoxelsInObj) > 0:
            if VERBOSE:
                print("\n    Need to provide unique IDs to each core of MASS")

            obj_core_graph = obj_graph.copy()
            notMASS = [int(v) for v in obj_graph.nodes if v not in MassVoxelsInObj]
            obj_core_graph.remove_nodes_from(notMASS)

            components = list(nx.connected_components(obj_core_graph))
            numcores = len(components)

            if VERBOSE:
                print(f"\n    There are {numcores} core MASSES in this object")

            for _, core_nodes in enumerate(components):
                coreID = coreCode.max() + 1
                MassInCore = list(core_nodes)

                for voxel in MassInCore:
                    coreCode[voxelID == voxel] = coreID

                expandedCoreCode[np.isin(voxelID, MassInCore)] = coreID

                inclSKIN = set()
                for voxel in MassInCore:
                    neighbors = list(obj_graph.neighbors(voxel))
                    inclSKIN.update(neighbors)

                MASSSKIN = set(inclSKIN)
                inclfirstconnector = set()
                for voxel in MASSSKIN:
                    neighbors = list(obj_graph.neighbors(voxel))
                    inclfirstconnector.update(neighbors)

                combined = np.unique(list(MASSSKIN.union(inclfirstconnector)))

                for voxel in combined:
                    expandedCoreCode[voxelID == voxel] = coreID
        else:
            if VERBOSE:
                print("\n    No MASS in this object; therefore no cores")

    print("\n\n  Cores, if present have been identified and coded\n")

    if PLOT:
        plotmorph = expandedCoreCode
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="Expanded CORE Voxels",
        )
        print("Expanded CORE Voxels")

    if VERBOSE:
        print("\n\n  Starting secondary test for ANTENNA detection")

    for obj in range(len(decompgraph)):
        if VERBOSE:
            print(f"\n\n  ANTENNA FIX object: {obj + 1}")

        has_mass = np.any((morphCode == 2) & (objectID == obj + 1))
        if not has_mass:
            if VERBOSE:
                print(f"Object {obj+1} has no MASS, skipping")
            continue

        if VERBOSE:
            print(f"Object {obj+1} has MASS, processing")

        possible = voxelID[(morphCode == 5) & (expandedCoreCode == 0) & (objectID == obj + 1)]
        allinobj = list(decompgraph[obj].nodes)
        to_delete = [node for node in allinobj if node not in possible]

        tmpgph = decompgraph[obj].copy()
        tmpgph.remove_nodes_from(to_delete)
        subgraphs = list(nx.connected_components(tmpgph))

        if VERBOSE:
            print(f"Object has {len(subgraphs)} possible ANTENNA to test for connectivity")

        for subgraph_nodes in subgraphs:
            subgraph = tmpgph.subgraph(subgraph_nodes).copy()
            if len(subgraph.nodes) <= 1:
                continue

            endpoints = [node for node, degree in subgraph.degree if degree == 1]
            if VERBOSE:
                print(f"Subgraph endpoints: {endpoints}")

            plugpoints = {ep: list(maingraph.neighbors(ep)) for ep in endpoints}
            if not plugpoints:
                continue

            tab = {ep: plugpoints[ep] for ep in endpoints}
            tab2 = {
                ep: [expandedCoreCode[voxelID == n] if n in voxelID else 0 for n in neighbors]
                for ep, neighbors in tab.items()
            }

            cores_linked = {int(core) for neighbors in tab2.values() for core in neighbors if core > 0}

            if len(cores_linked) == 1:
                rowsums = [sum(1 for core in neighbors if core > 0) for neighbors in tab2.values()]
                if sum(rowsums) == 1:
                    for node in subgraph.nodes:
                        morphCode[voxelID == node] = 6

            elif len(cores_linked) > 1:
                for node in subgraph.nodes:
                    morphCode[voxelID == node] = 7

                if VERBOSE:
                    print("An update of CIRCUIT to BOND is required")

                for node in subgraph.nodes:
                    loc = np.where(voxelID == node)
                    neighbors = {
                        "xlow": morphCode[loc[0] - 1, loc[1], loc[2]] if loc[0] > 0 else 0,
                        "xhigh": morphCode[loc[0] + 1, loc[1], loc[2]] if loc[0] < dimdatacube[0] - 1 else 0,
                        "ylow": morphCode[loc[0], loc[1] - 1, loc[2]] if loc[1] > 0 else 0,
                        "yhigh": morphCode[loc[0], loc[1] + 1, loc[2]] if loc[1] < dimdatacube[1] - 1 else 0,
                        "zlow": morphCode[loc[0], loc[1], loc[2] - 1] if loc[2] > 0 else 0,
                        "zhigh": morphCode[loc[0], loc[1], loc[2] + 1] if loc[2] < dimdatacube[2] - 1 else 0,
                    }
                    for key, val in neighbors.items():
                        if val == 5:
                            indices = {
                                "xlow": (loc[0] - 1, loc[1], loc[2]),
                                "xhigh": (loc[0] + 1, loc[1], loc[2]),
                                "ylow": (loc[0], loc[1] - 1, loc[2]),
                                "yhigh": (loc[0], loc[1] + 1, loc[2]),
                                "zlow": (loc[0], loc[1], loc[2] - 1),
                                "zhigh": (loc[0], loc[1], loc[2] + 1),
                            }
                            morphCode[indices[key]] = 7

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 6] = 6
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="ANTENNA Voxels",
            save_path=_savepath("ANTENNA"),
            save_dpi=SAVE_DPI,
        )

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 7] = 7
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="BOND Voxels",
            save_path=_savepath("BOND"),
            save_dpi=SAVE_DPI,
        )

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 5] = 5
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="CIRCUIT Voxels",
            save_path=_savepath("CIRCUIT"),
            save_dpi=SAVE_DPI,
        )

    # ------------ VOID-VOLUME (CODE = 8) + VOID (CODE = 9) ------------
    if VERBOSE:
        print("\n\nIdentifying VOID-VOLUME and VOID voxels")

    valid_void_mask, void_shell_mask, voidvolume = _find_valid_void_regions(DATACUBE)

    # There cannot be VOID-VOLUME without VOID
    # so assign both together only for validated regions
    morphCode[valid_void_mask] = 8

    object_codes = {2, 3, 4, 5, 6, 7}
    morphCode[void_shell_mask & np.isin(morphCode, list(object_codes))] = 9

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 8] = 8
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="VOID-VOLUME Voxels",
            save_path=_savepath("VOIDVOLUME"),
            save_dpi=SAVE_DPI,
        )

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 9] = 9
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="VOID Voxels",
            save_path=_savepath("VOID"),
            save_dpi=SAVE_DPI,
        )

    if PLOT:
        plotmorph = np.zeros_like(morphCode)
        plotmorph[morphCode == 3] = 3
        morphs = morph3dprep(plotmorph)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="SKIN Voxels",
            save_path=_savepath("SKIN"),
            save_dpi=SAVE_DPI,
        )

    # ------------ FINISHING UP ------------
    descriptions = [
        "OUTSIDE", "MASS", "SKIN", "CRUMB",
        "CIRCUIT", "ANTENNA", "BOND", "VOID-VOLUME", "VOID"
    ]
    summaries = []

    for code in range(1, 10):
        n_voxels = np.sum(morphCode == code)
        percentage = (n_voxels / np.max(voxelID)) * 100
        summaries.append([code, descriptions[code - 1], n_voxels, percentage])

    summaries_df = pd.DataFrame(
        summaries,
        columns=["Code", "Description", "NVoxels", "Percentage"],
    )

    if VERBOSE:
        print("\n\nSummary of Morphology Codes:\n")
        print(summaries_df)
        print("\n")

    if FINALPLOT:
        morphs = morph3dprep(morphCode, orig=False, FINAL=False)
        morph3dplot(
            morphs,
            CELLID=PLOTIDS,
            LEGEND=False,
            ORIGTRANSP=False,
            title="3D Morphology",
            save_path=_savepath("3D_Morphology"),
            save_dpi=SAVE_DPI,
            full_shape=morphCode.shape,
        )
        print("3D Morphology")

    _end_ts = time.time()
    _end_ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_end_ts))
    _elapsed = _end_ts - _start_ts
    _end_rss = _proc.memory_info().rss
    try:
        _peak_now = getattr(_proc.memory_full_info(), "peak_wset", None) or getattr(
            _proc.memory_full_info(), "peak_rss", None
        )
        if _peak_now:
            _peak = max((_peak or 0), _peak_now)
    except Exception:
        pass

    RunMetrics = {
        "start_time_str": _start_ts_str,
        "end_time_str": _end_ts_str,
        "elapsed_seconds": float(f"{_elapsed:.3f}"),
        "start_rss_bytes": int(_start_rss),
        "end_rss_bytes": int(_end_rss),
        "peak_rss_bytes": int(_peak) if _peak is not None else None,
        "start_rss_hr": _fmt_bytes(_start_rss),
        "end_rss_hr": _fmt_bytes(_end_rss),
        "peak_rss_hr": _fmt_bytes(_peak) if _peak is not None else None,
    }

    if VERBOSE:
        print("\n--- Run Metrics ---")
        print(f"Start:   {RunMetrics['start_time_str']}")
        print(f"End:     {RunMetrics['end_time_str']}")
        print(f"Elapsed: {RunMetrics['elapsed_seconds']} s")
        print(f"RSS:     {RunMetrics['start_rss_hr']} ➜ {RunMetrics['end_rss_hr']}")
        if RunMetrics["peak_rss_hr"] is not None:
            print(f"Peak RSS (best-effort): {RunMetrics['peak_rss_hr']}")
        print("----------------------------------------------------------------")

    morphCode_xyz = morphCode.copy()

    outobj = {
        "OriginalData": DATACUBE,
        "Graph": decompgraph,
        "VoxelIDs": voxelID,
        "ObjectID": objectID,
        "Morphology": morphCode_xyz,
        "Cores": coreCode,
        "ExpCores": expandedCoreCode,
        "Summary": summaries_df,
        "Egg": maingraph,
        "Bgrnd": lrgdatacube2,
        "VOIDvolume": voidvolume,
        "RunMetrics": RunMetrics,
    }
#Autosaving the 3D Morphology Array
    if AUTOSAVE and SAVE_MORPH_ARRAY and run_folder is not None:
        cube = np.asarray(morphCode_xyz).astype(int)
        if YEARS is None:
            YEARS = list(range(cube.shape[0]))
        T_expected = len(YEARS)
        shape = cube.shape
        time_axes = [ax for ax, n in enumerate(shape) if n == T_expected]
        if not time_axes:
            raise ValueError(
                f"None of the morphology cube dimensions match len(YEARS)={T_expected}. "
                f"Morphology shape is {shape}. Check YEARS or cube orientation."
            )
        t_axis = time_axes[0]
        cube_txy = np.moveaxis(cube, t_axis, 0)
        T, X, Y = cube_txy.shape
        blocks = []
        for ti in range(T):
            grid = pd.DataFrame(cube_txy[ti])
            blocks.append(grid)
            blocks.append(pd.DataFrame([[np.nan] * Y]))
        stacked = pd.concat(blocks, ignore_index=True)
        years_start = YEARS[0]
        years_end = YEARS[-1]
        out_csv = run_folder / f"{SITE_ID}_morph_codes_PUREGRID_{years_start}_{years_end}_{timestamp}.csv"
        stacked.to_csv(out_csv, index=False, header=False)
        map_rows = []
        row = 1
        for yr in YEARS:
            start = row
            end = row + (X - 1)
            map_rows.append([yr, start, end])
            row = end + 2
        map_df = pd.DataFrame(map_rows, columns=["Year", "StartRow", "EndRow"])
        map_csv = run_folder / f"{SITE_ID}_PUREGRID_row_map_{years_start}_{years_end}_{timestamp}.csv"
        map_df.to_csv(map_csv, index=False)
        outobj["MorphologyCSV"] = str(out_csv)
        outobj["MorphologyRowMapCSV"] = str(map_csv)
        outobj["RunFolder"] = str(run_folder)
        if VERBOSE:
            print("Morphology array saved:", out_csv)
            print("Row map saved:", map_csv)

    return outobj