#!/usr/bin/env python3
"""
Spatial IDW reconstruction for the Re=100 fixed-cylinder dataset.

Default experiment:
  * 256 sensors selected from the fluid crop using balanced 0.5D x 0.5D bins.
  * Bins are filled round-robin so their sensor counts differ by at most one
    whenever the available mesh nodes permit it. Nodes within each bin are
    selected randomly without replacement.
  * k=6 nearest sensors, inverse-square distance weights.
  * Sensors hidden behind the cylinder are excluded when at least one
    line-of-sight sensor is available.
  * 41 observed snapshots: t=400, 400.5, ..., 420.
  * Spatial evaluation on 2,000 held-out crop nodes.

IMPORTANT:
The report states "balanced 0.5D bins" but does not document the exact
bin origin, within-bin selection rule, or random seed. This script therefore
implements a transparent approximation: 0.5 x 0.5 bins anchored at the crop
minimum, with round-robin equalization and random mesh-node selection. It is
not guaranteed to reproduce the report's exact 8.02% / 16.07% values.

Usage:
  python spatial_idw.py --data /scratch/$USER/cylinder_dataset/fixed_cylinder_atRe100
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, Delaunay


# Dataset facts from the project report
N_FRAMES = 201
N_NODES = 82872
T0 = 400.0
DT = 0.1

# Requested spatial crop
XMIN, XMAX = -1.5, 8.0
YMIN, YMAX = -2.0, 2.0
CYLINDER_R = 0.5
SENSOR_RING_R = 0.55


def read_field_file(path):
    """
    Stream the publisher's text export into preallocated arrays.
    The report specifies 201 blocks of 82,872 rows, with columns x,y,u,v,p.
    Non-data/header lines are ignored; each accepted row must contain
    exactly five numeric values. Streaming avoids holding millions of Python
    lists in RAM for this 1.2 GB text file.
    """
    xy = np.empty((N_NODES, 2), dtype=np.float64)
    uvp = np.empty((N_FRAMES, N_NODES, 3), dtype=np.float32)
    float_re = re.compile(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
    )
    row_count = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            vals = float_re.findall(line)
            if len(vals) != 5:
                continue
            try:
                row = [float(v.replace("D", "E").replace("d", "e")) for v in vals]
            except ValueError:
                continue
            if not np.all(np.isfinite(row)):
                continue

            frame = row_count // N_NODES
            node = row_count % N_NODES
            if frame >= N_FRAMES:
                raise ValueError("Found more numeric field rows than expected.")
            if frame == 0:
                xy[node] = row[:2]
            else:
                # Coordinates must be stationary and retain identical ordering.
                if not np.allclose(row[:2], xy[node], rtol=0, atol=1e-10):
                    raise ValueError(
                        f"Mesh coordinate/order changed at frame {frame}, node {node}."
                    )
            uvp[frame, node] = row[2:5]
            row_count += 1

    expected = N_FRAMES * N_NODES
    if row_count != expected:
        raise ValueError(
            f"Parsed {row_count:,} five-column rows; expected {expected:,}. "
            "The raw file format may differ. Inspect text_flow.py / file headers."
        )

    times = T0 + DT * np.arange(N_FRAMES)
    return xy, uvp, times


def crop_indices(xy):
    x, y = xy[:, 0], xy[:, 1]
    r = np.hypot(x, y)
    inside_rect = (
        (x >= XMIN) & (x <= XMAX) &
        (y >= YMIN) & (y <= YMAX)
    )
    fluid = r > CYLINDER_R + 1e-8
    return np.flatnonzero(inside_rect & fluid)


def make_sensor_targets(n_side=16):
    """Regular 16x16 target lattice; move points in the solid onto r=0.55."""
    xs = np.linspace(XMIN, XMAX, n_side)
    ys = np.linspace(YMIN, YMAX, n_side)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    targets = np.column_stack([xx.ravel(), yy.ravel()])

    rr = np.hypot(targets[:, 0], targets[:, 1])
    solid = rr <= SENSOR_RING_R
    # Radially project solid/near-solid targets to a thin fluid-side ring.
    safe = np.maximum(rr[solid], 1e-12)
    targets[solid, 0] *= SENSOR_RING_R / safe
    targets[solid, 1] *= SENSOR_RING_R / safe
    return targets


def snap_targets_to_unique_nodes(targets, xy, candidate_pool=64):
    """Snap each target to a distinct nearest mesh node."""
    tree = cKDTree(xy)
    k = min(candidate_pool, len(xy))
    distances, indices = tree.query(targets, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]

    # Greedy assignment in order of target difficulty (nearest candidate distance).
    order = np.argsort(distances[:, 0])[::-1]
    chosen = np.full(len(targets), -1, dtype=int)
    used = set()
    for ti in order:
        for node in indices[ti]:
            node = int(node)
            if node not in used:
                chosen[ti] = node
                used.add(node)
                break
    if np.any(chosen < 0):
        raise RuntimeError("Could not assign 256 unique mesh nodes; increase candidate_pool.")
    return chosen



def select_balanced_bin_sensors(xy, n_sensors=256, bin_size=0.5, seed=2026):
    """
    Select actual fluid mesh nodes with balanced occupancy across 0.5D bins.

    Bins are anchored at (XMIN, YMIN). We randomize node order within each
    occupied bin, then take one node per bin per round until n_sensors are
    selected. This keeps counts as even as possible without inventing
    off-mesh sensor coordinates.
    """
    rng = np.random.default_rng(seed)
    bx = np.floor((xy[:, 0] - XMIN) / bin_size).astype(int)
    by = np.floor((xy[:, 1] - YMIN) / bin_size).astype(int)
    bins = {}
    for i, key in enumerate(zip(bx, by)):
        bins.setdefault(key, []).append(i)

    # Randomize bin order and the available nodes within each bin.
    keys = list(bins.keys())
    rng.shuffle(keys)
    nodes_by_bin = {key: rng.permutation(bins[key]).tolist() for key in keys}
    selected = []
    round_id = 0
    while len(selected) < n_sensors:
        added_this_round = 0
        for key in keys:
            nodes = nodes_by_bin[key]
            if round_id < len(nodes):
                selected.append(int(nodes[round_id]))
                added_this_round += 1
                if len(selected) == n_sensors:
                    break
        if added_this_round == 0:
            raise ValueError(
                f"Only {len(selected)} fluid nodes available across the bins; "
                f"cannot select {n_sensors} sensors."
            )
        round_id += 1
    return np.asarray(selected, dtype=int)


def segment_intersects_circle(a, b, radius=CYLINDER_R):
    """True if the open segment a--b passes through the solid circle."""
    d = b - a
    denom = np.sum(d * d, axis=-1)
    safe_denom = np.maximum(denom, 1e-20)
    tau = -np.sum(a * d, axis=-1) / safe_denom
    tau = np.clip(tau, 0.0, 1.0)
    closest = a + tau[..., None] * d
    # Strictly inside radius; endpoints on/near the wall are not considered blocked.
    return np.sum(closest * closest, axis=-1) < (radius - 1e-6) ** 2


def idw_reconstruct(query_xy, sensor_xy, sensor_values, k=6, power=2.0):
    """IDW at query points. sensor_values shape: (n_sensors, n_components)."""
    tree = cKDTree(sensor_xy)
    dist, idx = tree.query(query_xy, k=min(k, len(sensor_xy)))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]

    vals = sensor_values[idx]  # (n_query, k, n_components)

    # Exact sensor/query coincidence: return that sensor's value exactly.
    exact = dist[:, 0] < 1e-12
    weights = 1.0 / np.maximum(dist, 1e-12) ** power

    # Remove sensors whose connecting segment crosses the cylinder.
    # If all selected neighbours are blocked, retain the ordinary k-neighbour set.
    blocked = segment_intersects_circle(
        query_xy[:, None, :], sensor_xy[idx, :], CYLINDER_R
    )
    visible = ~blocked
    has_visible = np.any(visible, axis=1)
    use = np.where(has_visible[:, None], visible, np.ones_like(visible))
    weights *= use

    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    pred = np.sum(weights[..., None] * vals, axis=1)

    if np.any(exact):
        pred[exact] = vals[exact, 0, :]
    return pred


def lumped_area_weights(xy):
    """
    Approximate nodal integration weights from Delaunay triangles.
    Triangles whose centroids lie inside the cylinder are removed to respect
    the hole. This is an inferred triangulation, not the original FEM mesh.
    """
    tri = Delaunay(xy)
    p = xy[tri.simplices]
    twice_area = np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    )
    area = 0.5 * twice_area
    cent = p.mean(axis=1)
    keep = np.hypot(cent[:, 0], cent[:, 1]) > CYLINDER_R
    weights = np.zeros(len(xy), dtype=np.float64)
    for j in range(3):
        np.add.at(weights, tri.simplices[keep, j], area[keep] / 3.0)
    return weights


def relative_l2(pred, truth, weights=None):
    """Relative L2 per component, expressed as percent."""
    diff2 = (pred - truth) ** 2
    truth2 = truth ** 2
    if weights is None:
        num = np.sum(diff2, axis=0)
        den = np.sum(truth2, axis=0)
    else:
        w = weights[:, None]
        num = np.sum(w * diff2, axis=0)
        den = np.sum(w * truth2, axis=0)
    return 100.0 * np.sqrt(num / np.maximum(den, 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to fixed_cylinder_atRe100")
    parser.add_argument("--outdir", default="idw_results")
    parser.add_argument("--sensors", type=int, default=256)
    parser.add_argument("--bin-size", type=float, default=0.5,
                        help="Spatial bin width in D; report specifies 0.5D")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--holdout", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame-step", type=int, default=5,
                        help="5 gives t=400,400.5,...,420 (41 frames)")
    args = parser.parse_args()

    if args.sensors < 1:
        raise ValueError("--sensors must be positive.")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading CFD field file...")
    xy_all, uvp, times = read_field_file(args.data)
    crop = crop_indices(xy_all)
    xy = xy_all[crop]
    print(f"Full mesh: {len(xy_all):,} nodes; fluid crop: {len(crop):,} nodes")

    # Select sensors directly from actual fluid nodes using balanced 0.5D bins.
    sensor_local = select_balanced_bin_sensors(
        xy, n_sensors=args.sensors, bin_size=args.bin_size, seed=args.seed
    )
    sensor_global = crop[sensor_local]
    sensor_xy = xy[sensor_local]

    # Save sensor locations and their bin IDs for reproducibility.
    sensor_bx = np.floor((sensor_xy[:, 0] - XMIN) / args.bin_size).astype(int)
    sensor_by = np.floor((sensor_xy[:, 1] - YMIN) / args.bin_size).astype(int)
    with open(outdir / "sensor_locations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "global_node_index", "x", "y", "bin_x", "bin_y"])
        for j, (gi, pt, ibx, iby) in enumerate(zip(sensor_global, sensor_xy, sensor_bx, sensor_by)):
            writer.writerow([j, int(gi), float(pt[0]), float(pt[1]), int(ibx), int(iby)])

    # Permanently held-out spatial sites, shared across all observed frames.
    candidates = np.setdiff1d(np.arange(len(crop)), sensor_local)
    rng = np.random.default_rng(args.seed)
    n_hold = min(args.holdout, len(candidates))
    test_local = np.sort(rng.choice(candidates, size=n_hold, replace=False))
    test_xy = xy[test_local]

    # Approximate area weights, normalized over the fixed test sites.
    print("Building inferred Delaunay area weights...")
    all_area = lumped_area_weights(xy)
    test_weights = all_area[test_local]
    if np.sum(test_weights) <= 0:
        raise RuntimeError("Area weights are zero on the test sites.")
    test_weights /= np.sum(test_weights)

    frame_ids = np.arange(0, N_FRAMES, args.frame_step)
    rows = []
    pred_store = []
    full_pred_store = []
    pooled_err_area = np.zeros(2, dtype=np.float64)
    pooled_truth_area = np.zeros(2, dtype=np.float64)
    pooled_err_node = np.zeros(2, dtype=np.float64)
    pooled_truth_node = np.zeros(2, dtype=np.float64)

    print(f"Using {len(sensor_local)} sensors in balanced {args.bin_size:g}D bins; "
          f"evaluating {len(test_local)} held-out sites")
    print(f"Evaluating {len(frame_ids)} snapshots from t={times[frame_ids[0]]:.1f} "
          f"to t={times[frame_ids[-1]]:.1f}")

    for frame in frame_ids:
        truth_all = uvp[frame, crop, :2]
        sensor_values = truth_all[sensor_local]
        truth = truth_all[test_local]
        pred = idw_reconstruct(
            test_xy, sensor_xy, sensor_values, k=args.k, power=args.power
        )

        err_area = relative_l2(pred, truth, test_weights)
        err_node = relative_l2(pred, truth, None)
        pooled_err_area += np.sum(test_weights[:, None] * (pred - truth) ** 2, axis=0)
        pooled_truth_area += np.sum(test_weights[:, None] * truth ** 2, axis=0)
        pooled_err_node += np.sum((pred - truth) ** 2, axis=0)
        pooled_truth_node += np.sum(truth ** 2, axis=0)
        rows.append([
            int(frame), float(times[frame]),
            float(err_area[0]), float(err_area[1]),
            float(err_node[0]), float(err_node[1])
        ])
        pred_store.append(pred.astype(np.float32))
        full_pred = idw_reconstruct(
            xy, sensor_xy, sensor_values, k=args.k, power=args.power
        )
        full_pred_store.append(full_pred.astype(np.float32))
        print(
            f"t={times[frame]:6.2f} | area rL2 u={err_area[0]:7.3f}% "
            f"v={err_area[1]:7.3f}% | node rL2 u={err_node[0]:7.3f}% "
            f"v={err_node[1]:7.3f}%"
        )

    rows = np.asarray(rows, dtype=np.float64)
    with open(outdir / "idw_metrics_per_frame.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "time", "area_relative_L2_u_percent",
            "area_relative_L2_v_percent", "node_relative_L2_u_percent",
            "node_relative_L2_v_percent"
        ])
        writer.writerows(rows.tolist())

    mean_frame_area = np.mean(rows[:, 2:4], axis=0)
    mean_frame_node = np.mean(rows[:, 4:6], axis=0)
    pooled_area = 100.0 * np.sqrt(pooled_err_area / np.maximum(pooled_truth_area, 1e-30))
    pooled_node = 100.0 * np.sqrt(pooled_err_node / np.maximum(pooled_truth_node, 1e-30))
    summary = np.vstack([pooled_area, pooled_node, mean_frame_area, mean_frame_node])
    np.savetxt(
        outdir / "idw_summary.txt", summary,
        header=("Rows: pooled area-weighted u,v; pooled node-weighted u,v; "
                "mean per-frame area-weighted u,v; mean per-frame node-weighted u,v. "
                "Values in percent.")
    )

    np.savez_compressed(
        outdir / "idw_reconstruction.npz",
        times=times[frame_ids],
        test_xy=test_xy,
        sensor_xy=sensor_xy,
        sensor_global_indices=sensor_global,
        test_global_indices=crop[test_local],
        predictions=np.stack(pred_store, axis=0),
        crop_xy=xy,
        full_crop_predictions=np.stack(full_pred_store, axis=0),
    )

    print("\n=== Pooled errors over all evaluated snapshots and held-out sites ===")
    print(f"Area-weighted relative L2: u={pooled_area[0]:.4f}%, v={pooled_area[1]:.4f}%")
    print(f"Node-weighted relative L2: u={pooled_node[0]:.4f}%, v={pooled_node[1]:.4f}%")
    print("Mean per-frame area-weighted errors: "
          f"u={mean_frame_area[0]:.4f}%, v={mean_frame_area[1]:.4f}%")
    print(f"Saved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
