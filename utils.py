import igraph as ig
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def arraytrim(VOLOBJ):
    # Store the dimensions of the input 3D array
    dimx, dimy, dimz = VOLOBJ.shape

    # Prepare vectors that will indicate which planes to trim
    trimx = np.full(dimx, -1)
    trimy = np.full(dimy, -1)
    trimz = np.full(dimz, -1)

    # Identify which planes need to be cut and update the appropriate vectors for X, Y, Z dimensions
    for plane in range(dimx):
        if np.sum(VOLOBJ[plane, :, :]) == 0:
            trimx[plane] = plane
    for plane in range(dimy):
        if np.sum(VOLOBJ[:, plane, :]) == 0:
            trimy[plane] = plane
    for plane in range(dimz):
        if np.sum(VOLOBJ[:, :, plane]) == 0:
            trimz[plane] = plane

    # Identify low-end indices for performing the trimming
    xindexlow = next((i for i, x in enumerate(trimx) if x == -1), dimx)
    yindexlow = next((i for i, y in enumerate(trimy) if y == -1), dimy)
    zindexlow = next((i for i, z in enumerate(trimz) if z == -1), dimz)

    # Identify high-end indices for performing the trimming
    xindexhigh = next((i for i, x in enumerate(trimx[::-1]) if x == -1), dimx)
    yindexhigh = next((i for i, y in enumerate(trimy[::-1]) if y == -1), dimy)
    zindexhigh = next((i for i, z in enumerate(trimz[::-1]) if z == -1), dimz)

    xindexhigh = dimx - xindexhigh - 1
    yindexhigh = dimy - yindexhigh - 1
    zindexhigh = dimz - zindexhigh - 1

    # Correct for null dimensions when trimming is not required
    if xindexlow == dimx:
        xindexlow = 0
    if yindexlow == dimy:
        yindexlow = 0
    if zindexlow == dimz:
        zindexlow = 0

    if xindexhigh == -1:
        xindexhigh = dimx - 1
    if yindexhigh == -1:
        yindexhigh = dimy - 1
    if zindexhigh == -1:
        zindexhigh = dimz - 1

    # Perform and return the subset
    return VOLOBJ[xindexlow:xindexhigh + 1, yindexlow:yindexhigh + 1, zindexlow:zindexhigh + 1]


def validate_data_cube(datacube):
    # First check if data provided is a numpy array
    if isinstance(datacube, np.ndarray):
        # If data is a numpy array
        # # Check if the array is numeric
        if np.issubdtype(datacube.dtype, np.number):
            #If array is numeric, check if the array is 3 dimensional
            if datacube.ndim == 3:
                # Finally check if array is binary
                if np.all(np.logical_or(datacube == 0, datacube == 1)):
                    print("\n\n Data passes all tests \n\n")
                    return True
                else:
                    print("\n\n ERROR 004 - Input data does not contain proper 0,1 binary data.\n\n")
                    return False
            else:
                print("\n\nERROR 003 - Input data is not a 3D array.\n\n")
                return False
        else:
            print("\n\nERROR 001 - Input data is not numeric.\n\n")
            return False
    else:
        print("\n\nERROR 002 - Input data is not a numpy array.\n\n")
        return False



def morph3dlinks(VOLOBJ=None, VOXELIDS=None, VERBOSE=False):
    
    """
    Performs 3D Morphological Segmentation on a volumetric object.

    Parameters:
    - VOLOBJ: 3D numpy array (0 for background, 1 for object of interest)
    - VOXELIDS: 3D numpy array of the same shape as VOLOBJ
    - VERBOSE: Boolean, if True, prints additional information

    Returns:
    - DataFrame: Pandas DataFrame containing the results of the segmentation
    """

    if VERBOSE:
        print(f"\nStarting 3D Morphological Segmentation on object: {VOLOBJ}.\n")

    # Store dimensions of the expanded array
    lrgarraydim = VOLOBJ.shape

    # Initialize shifting arrays to zeros
    up = np.zeros_like(VOLOBJ)
    down = np.zeros_like(VOLOBJ)
    left = np.zeros_like(VOLOBJ)
    right = np.zeros_like(VOLOBJ)
    forward = np.zeros_like(VOLOBJ)
    backward = np.zeros_like(VOLOBJ)

    lrgvoxelIDS = np.zeros_like(VOLOBJ)
    lrgvoxelIDS[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1] = VOXELIDS

    # Shift down
    down[:, :, 1:lrgarraydim[2]-1] = lrgvoxelIDS[:, :, 2:lrgarraydim[2]]
    down = down[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Shift up
    up[:, :, 1:lrgarraydim[2]-1] = lrgvoxelIDS[:, :, 0:lrgarraydim[2]-2]
    up = up[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Shift left
    left[:, 1:lrgarraydim[1]-1, :] = lrgvoxelIDS[:, 2:lrgarraydim[1], :]
    left = left[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Shift right
    right[:, 1:lrgarraydim[1]-1, :] = lrgvoxelIDS[:, 0:lrgarraydim[1]-2, :]
    right = right[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Shift forward
    forward[1:lrgarraydim[0]-1, :, :] = lrgvoxelIDS[2:lrgarraydim[0], :, :]
    forward = forward[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Shift backward
    backward[1:lrgarraydim[0]-1, :, :] = lrgvoxelIDS[0:lrgarraydim[0]-2, :, :]
    backward = backward[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Extract the original size from the large array
    VOLOBJ = VOLOBJ[1:lrgarraydim[0]-1, 1:lrgarraydim[1]-1, 1:lrgarraydim[2]-1]

    # Flatten the arrays for easier aggregation into a DataFrame
    VOLOBJ_flat = VOLOBJ.flatten()
    VOXELIDS_flat = VOXELIDS.flatten()
    up_flat = up.flatten()
    down_flat = down.flatten()
    left_flat = left.flatten()
    right_flat = right.flatten()
    forward_flat = forward.flatten()
    backward_flat = backward.flatten()

    # Aggregate results into a DataFrame
    df = pd.DataFrame({
        'VOLOBJ': VOLOBJ_flat,
        'VOXELIDS': VOXELIDS_flat,
        'up': up_flat,
        'down': down_flat,
        'left': left_flat,
        'right': right_flat,
        'forward': forward_flat,
        'backward': backward_flat
    })

    # Filter the DataFrame for rows where VOLOBJ == 1
    df = df[df['VOLOBJ'] == 1]

    # Return the result
    return df



def _dark_3d_style(ax, fig):
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_visible(False)
        except Exception:
            pass

    try:
        ax.w_xaxis.line.set_color((0, 0, 0, 0))
        ax.w_yaxis.line.set_color((0, 0, 0, 0))
        ax.w_zaxis.line.set_color((0, 0, 0, 0))
    except Exception:
        pass

    ax.grid(False)


def _light_3d_style(ax, fig):
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_visible(False)
        except Exception:
            pass

    try:
        ax.w_xaxis.line.set_color((0, 0, 0, 0))
        ax.w_yaxis.line.set_color((0, 0, 0, 0))
        ax.w_zaxis.line.set_color((0, 0, 0, 0))
    except Exception:
        pass

    ax.grid(False)
    ax.set_axis_off()


def _draw_cube_lattice(ax, shape, step=1, color=(1, 1, 1, 0.08), lw=0.45):
    """
    Faint full-cube wireframe lattice, like your example.
    shape = (nx, ny, nz)
    """
    nx, ny, nz = shape
    xs = np.arange(0, nx + 1, step)
    ys = np.arange(0, ny + 1, step)
    zs = np.arange(0, nz + 1, step)

    # lines parallel to x
    for y in ys:
        for z in zs:
            ax.plot([0, nx], [y, y], [z, z], color=color, linewidth=lw)

    # lines parallel to y
    for x in xs:
        for z in zs:
            ax.plot([x, x], [0, ny], [z, z], color=color, linewidth=lw)

    # lines parallel to z
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [0, nz], color=color, linewidth=lw)

def _set_axes_equal(ax, xlim, ylim, zlim):
    """
    Force equal scale on x/y/z so voxels look like perfect cubes.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    max_range = max(x_range, y_range, z_range)

    x_mid = (xlim[0] + xlim[1]) / 2
    y_mid = (ylim[0] + ylim[1]) / 2
    z_mid = (zlim[0] + zlim[1]) / 2

    half = max_range / 2
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    # Newer Matplotlib supports this (best when available)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def morph3dplot(
    data,
    title="3D Morphology",
    show_title=False,
    save_path=None,
    save_dpi=300,

    # what to show
    plot_codes=None,          # e.g. [5] for CIRCUIT only, or [2,5,7]
    show_skin=True,           # show SKIN (code 3) as ash
    show_outside=False,       # outside is code 1

    # styling
    dark=False,                # black background
    ticks=False,              # remove tick numbering
    wireframe_cube=False,      # draw full cube lattice
    lattice_step=2,

    # skin style
    skin_as_ash=True,
    skin_alpha=0.03,

    # voxel edges
    solid_alpha=0.98,
    solid_edge_alpha=0.02,
    linewidth=0.15,
    axis_order=(2, 1, 0),
    flip_axes=(True, False, False),
    elev=18,
    azim=60,
    show_axes_arrows=False,
    arrow_scale=1.0,
    arrow_color="black",
    arrow_linewidth=2.0,
    show_axis_labels=True,
    axis_name_map=("X", "Y", "T"),
    **kwargs,
):
    """
    Expects a DataFrame with columns: x, y, z, value

    Codes:
      0 background
      1 OUTSIDE
      2 MASS
      3 SKIN
      4 CRUMB
      5 CIRCUIT
      6 ANTENNA
      7 BOND
      8 VOID-VOLUME
      9 VOID
    """

    # ---- build cube from points ----
    x = data["x"].to_numpy().astype(int)
    y = data["y"].to_numpy().astype(int)
    z = data["z"].to_numpy().astype(int)
    v = data["value"].to_numpy().astype(int)

    nx = x.max() + 1
    ny = y.max() + 1
    nz = z.max() + 1

    vals = np.zeros((nx, ny, nz), dtype=int)
    vals[x, y, z] = v

# --- axis mapping to match R/rgl orientation ---
    vals = np.transpose(vals, axis_order)

    fx, fy, fz = flip_axes
    if fx:
       vals = vals[::-1, :, :]
    if fy:
       vals = vals[:, ::-1, :]
    if fz:
       vals = vals[:, :, ::-1]

# update dims after transforms (important for lattice + limits)
    nx, ny, nz = vals.shape

    # ---- consistent RGBA colors by CODE ----
    cmap = {
        1: (0.75, 0.75, 0.75, 0.10),      # OUTSIDE  -> grey
        2: (0.00, 0.50, 0.00, solid_alpha), # MASS -> green
        3: (0.00, 0.00, 0.00, skin_alpha),  # SKIN -> black
        4: (0.65, 0.16, 0.16, solid_alpha), # CRUMB -> brown
        5: (1.00, 0.65, 0.00, solid_alpha), # CIRCUIT -> orange
        6: (1.00, 0.75, 0.80, solid_alpha), # ANTENNA -> pink
        7: (0.39, 0.58, 0.93, solid_alpha), # BOND -> cornflowerblue
        8: (0.00, 0.00, 0.50, solid_alpha), # VOID-VOLUME -> navy
        9: (0.40, 0.55, 0.34, solid_alpha), # VOID -> seagreen
        10:(0.25, 0.45, 0.85, solid_alpha)  # ObjectID (custom)
    }

    # ---- decide what to plot ----
    if plot_codes is None:
        # default: all morphological classes except 0 and 1
        plot_codes = [2, 4, 5, 6, 7, 8, 9]
        if show_skin:
            plot_codes = [3] + plot_codes

    if not show_outside and 1 in plot_codes:
        plot_codes = [c for c in plot_codes if c != 1]

    if not show_skin and 3 in plot_codes:
        plot_codes = [c for c in plot_codes if c != 3]

    # ---- start plot ----
    fig = plt.figure(figsize=(7.8, 7.8), dpi=150)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")

    if dark:
        _dark_3d_style(ax, fig)
    else:
      _light_3d_style(ax, fig)


    if wireframe_cube:
       lattice_color = (1, 1, 1, 0.08) if dark else (0, 0, 0, 0.08)
       _draw_cube_lattice(ax, (nx, ny, nz), step=lattice_step, color=lattice_color, lw=0.45)


    # ---- plot SKIN as transparent ash (code 3) ----
    # IMPORTANT: SKIN IS 3 
    if show_skin and (3 in plot_codes):
        mask_skin = (vals == 3)
        if mask_skin.any():
            ash = (0.75, 0.75, 0.75, 0.28)
            ax.voxels(
                mask_skin,
                facecolors=ash,
                edgecolor=(0.4, 0.4, 0.4, 0.18),
                linewidth=0.22,
                shade=False
            )

    # ---- plot selected classes as SOLID colored voxels ----
    for code in plot_codes:
        if code == 3:
            continue
        mask = (vals == code)
        if not mask.any():
            continue
        col = cmap.get(code, (1, 1, 1, solid_alpha))
        ax.voxels(
            mask,
            facecolors=col,
            edgecolor=(0.15, 0.15, 0.15, 0.25),
            linewidth=0.35,
            shade=False
        )

    # ---- formatting ----
    if show_title and title:
       ax.set_title(title, color=("white" if dark else "black"), pad=14)

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    _set_axes_equal(ax, (0, nx), (0, ny), (0, nz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
        # ---- optional axis arrows ----
    if show_axes_arrows:
        L = max(nx, ny, nz) * 0.25 * arrow_scale
        # Determine which original axis each plotted axis corresponds to
        # axis_order tells us: plotted axis 0 came from original axis axis_order[0], etc.
        # flip_axes tells us whether plotted axis direction is reversed.
        fx, fy, fz = flip_axes
        orig_names = list(axis_name_map)

        plotted_to_orig = [orig_names[i] for i in axis_order]
        plotted_signs = [-1 if fx else 1, -1 if fy else 1, -1 if fz else 1]

        # Helper to format label like X, -X, T, -T
        def _fmt(name, sgn):
            return f"-{name}" if sgn < 0 else f"+{name}"

        # X arrow
        ax.quiver(0, 0, 0,
                  L, 0, 0,
                  color=arrow_color,
                  linewidth=arrow_linewidth, 
                  arrow_length_ratio=0.12 )

        # Y arrow
        ax.quiver(0, 0, 0,
                  0, L, 0,
                  color=arrow_color,
                  linewidth=arrow_linewidth,
                  arrow_length_ratio=0.12)

        # Z arrow
        ax.quiver(0, 0, 0,
                  0, 0, L,
                  color=arrow_color,
                  linewidth=arrow_linewidth,
                  arrow_length_ratio=0.12)

        if show_axis_labels:
            ax.text(L, 0, 0, _fmt(plotted_to_orig[0], plotted_signs[0]), color=arrow_color)
            ax.text(0, L, 0, _fmt(plotted_to_orig[1], plotted_signs[1]), color=arrow_color)
            ax.text(0, 0, L, _fmt(plotted_to_orig[2], plotted_signs[2]), color=arrow_color)   # or "Z"
    
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0, transparent=True)

    return fig, ax




def morph3dprep(incube, orig=False, FINAL=False):
    """
    Convert a 3D numpy array (nx, ny, nz) into a DataFrame with columns:
    x, y, z, value

    IMPORTANT:
    - Coordinate generation must match incube.ravel() order (C-order).
    - Do NOT shift codes (no df['value'] -= 1).
    """
    nx, ny, nz = incube.shape

    # Build coordinates in the SAME axis order as incube indexing: incube[x, y, z]
    x, y, z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )

    df = pd.DataFrame({
        "x": x.ravel(order="C"),
        "y": y.ravel(order="C"),
        "z": z.ravel(order="C"),
        "value": incube.ravel(order="C")
    })

    # If you ever need the "orig" behavior, keep it explicit
    if orig:
        df["value"] = df["value"] + 1

    return df



