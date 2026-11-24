"""Reproduce the cylinder-wake EDA from the supplied text exports.

Run from any directory with Python, NumPy, and Matplotlib:
    python cylinder_eda/analyze.py
"""
from pathlib import Path
import hashlib
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "Data".parent / "Data"
COLORS = ["#174e83", "#b44a35", "#188275"]


def read_fields(data_path=None):
    digest = hashlib.md5()
    with Path(data_path or (ROOT / "fixed_cylinder_atRe100")).open("rb") as source:
        def read_line():
            raw = source.readline()
            digest.update(raw)
            return raw

        re, ur = map(float, read_line().split())
        assert not read_line().strip()
        nt, nn = map(int, read_line().split())
        assert not read_line().strip()
        times = np.empty(nt)
        fields = np.empty((nt, nn, 3))
        xy = None
        for i in range(nt):
            times[i] = float(read_line())
            raw = source.read(70 * nn)
            digest.update(raw)
            records = np.frombuffer(raw, dtype="S70")
            chars = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 70)
            assert len(records) == nn and np.all(chars[:, -1] == 10), i
            assert np.all(chars[:, [13, 27, 41, 55]] == 32), i
            # Validate five tokens on every line independently of the pooled parser.
            whitespace = (chars == 32) | (chars == 10)
            assert np.all(np.count_nonzero(whitespace[:, :-1] & ~whitespace[:, 1:], axis=1) + (~whitespace[:, 0]) == 5), i
            values = np.fromstring(raw.decode("ascii"), sep=" ").reshape(nn, 5)
            assert np.isfinite(values).all(), i
            if xy is None:
                xy = values[:, :2].copy()
            assert np.array_equal(xy, values[:, :2]), i
            fields[i] = values[:, 2:]
        assert not source.read(1), "Trailing records"
    assert np.isfinite(times).all() and np.all(np.diff(times) > 0)
    assert len(np.unique(xy, axis=0)) == nn
    return re, ur, times, xy, fields, digest.hexdigest()


def quantiles(values, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return np.percentile(values, qs, axis=0).tolist()


def peaks(t, a):
    """Refine local maxima using a quadratic in local time."""
    ii = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
    result = []
    for i in ii:
        aa, bb, _ = np.polyfit(t[i-1:i+2] - t[i], a[i-1:i+2], 2)
        if aa < 0:
            result.append(t[i] - bb / (2 * aa))
    return np.array(result)


def spectrum(t, a):
    grid = np.linspace(t[0], t[-1], len(t))
    uniform = np.interp(grid, t, a)
    nfft = 32768
    frequency = np.fft.rfftfreq(nfft, grid[1] - grid[0])
    windowed = (uniform - uniform.mean()) * np.hanning(len(t))
    power = np.abs(np.fft.rfft(windowed, n=nfft))**2
    return frequency, power / power.max()


def save(fig, name):
    fig.savefig(OUT / name, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def error_metrics(truth, pred, weights):
    e = pred - truth
    mse = np.mean(np.sum(e**2 * weights[None, :, None], axis=1), axis=0)
    ref = np.mean(np.sum(truth**2 * weights[None, :, None], axis=1), axis=0)
    return {"rmse": np.sqrt(mse), "relative_l2_percent": 100 * np.sqrt(mse / ref)}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=ROOT / 'fixed_cylinder_atRe100')
    parser.add_argument('--reaction-path', type=Path, help='Optional reaction-force file for force evaluation')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    re, ur, t, xy, fields, md5 = read_fields(args.data_path)
    if args.reaction_path is None:
        summary = {'field_md5': md5, 'reynolds': re, 'frames': len(t), 'nodes': len(xy), 'time': [float(t[0]), float(t[-1])], 'extent': [float(xy[:,0].min()), float(xy[:,0].max()), float(xy[:,1].min()), float(xy[:,1].max())], 'field_min': fields.min(axis=(0,1)).tolist(), 'field_max': fields.max(axis=(0,1)).tolist(), 'force_evaluation': 'skipped; supply --reaction-path'}
        (OUT / 'field_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return
    print("Field audit finished", flush=True)
    x, y = xy.T
    radius = np.hypot(x, y)
    wall = np.isclose(radius, .5, rtol=0, atol=1e-4)
    tri = mtri.Triangulation(x, y)
    faces = tri.triangles
    points = xy[faces]
    dx1, dy1 = (points[:, 1] - points[:, 0]).T
    dx2, dy2 = (points[:, 2] - points[:, 0]).T
    cross = dx1 * dy2 - dx2 * dy1
    # All-wall triangles are the inferred hole fill in the Delaunay mesh.
    hole = np.all(wall[faces], axis=1)
    tri.set_mask(hole)
    area = np.abs(cross) / 2
    weights = np.bincount(faces[~hole].ravel(), weights=np.repeat(area[~hole] / 3, 3), minlength=len(x))
    weights /= weights.sum()
    max_edge = np.maximum.reduce([np.hypot(dx1, dy1), np.hypot(dx2, dy2),
                                  np.hypot(dx2-dx1, dy2-dy1)])

    def gradients(z):
        a = z[faces[:, 1]] - z[faces[:, 0]]
        b = z[faces[:, 2]] - z[faces[:, 0]]
        return (a * dy2 - b * dy1) / cross, (dx1 * b - dx2 * a) / cross

    def curl_div(frame):
        ux, uy = gradients(frame[:, 0])
        vx, vy = gradients(frame[:, 1])
        return vx - uy, ux + vy

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    inlet = x == xmin
    outlet = x == xmax
    top = y == ymax
    bottom = y == ymin
    outer = inlet | outlet | top | bottom
    near = (x >= -1.5) & (x <= 1.5) & (np.abs(y) <= 2) & ~wall
    wake = (x > 1.5) & (x <= 8) & (np.abs(y) <= 2)
    roi = near | wake | wall
    rest = ~(roi | outer)
    regions = {"cylinder": wall, "inlet": inlet, "outlet": outlet,
               "top": top, "bottom": bottom, "near": near, "wake": wake, "rest": rest}

    raw_force = args.reaction_path.read_bytes()
    lines = [line.split() for line in raw_force.splitlines()]
    assert all(len(line) == 5 for line in lines)
    reactions = np.array([line[:4] for line in lines], dtype=float)
    tr, fx, fy, mz = reactions.T
    assert np.isfinite(reactions).all() and np.all(np.diff(tr) > 0)
    overlap = (tr >= t[0]) & (tr <= t[-1])

    late = (tr >= t[0]) & (tr <= tr[-1])
    fx_peak, fy_peak = peaks(tr[late], fx[late]), peaks(tr[late], fy[late])
    f = 1 / np.median(np.diff(fy_peak))

    aligned_force = np.column_stack([np.interp(t, tr, s) for s in (fx, fy, mz)])
    force_nearest = np.searchsorted(tr, t)
    force_offset = np.minimum(abs(tr[force_nearest] - t), abs(tr[force_nearest-1] - t))
    probes = [(1, .5), (2, 0), (4, 0), (6, 1), (8, 0), (-2, 0)]
    probe_ids = [np.argmin((x-a)**2 + (y-b)**2) for a, b in probes]
    probe_stats = []
    for loc, i in zip(probes, probe_ids):
        sig = fields[:, i, :]
        freqs = [spectrum(t, col) for col in sig.T]
        probe_stats.append({"target": loc, "actual": xy[i], "std": sig.std(axis=0),
                            "peak_frequency": [fq[pw.argmax()] for fq, pw in freqs]})

    # Pressure-traction integral on the supplied wall nodes (polygon quadrature).
    wall_ids = np.flatnonzero(wall)
    angle = np.arctan2(y[wall], x[wall])
    wall_ids = wall_ids[np.argsort(angle)]
    edge = np.roll(xy[wall_ids], -1, axis=0) - xy[wall_ids]
    normal_ds = np.column_stack([edge[:, 1], -edge[:, 0]])
    pwall = fields[:, wall_ids, 2]
    pressure_force = -(pwall + np.roll(pwall, -1, axis=1)) @ normal_ds / 2
    wall_edges = np.stack([wall[faces[:, 0]] & wall[faces[:, 1]],
                           wall[faces[:, 1]] & wall[faces[:, 2]],
                           wall[faces[:, 2]] & wall[faces[:, 0]]], axis=1)
    adjoining, which = np.where(wall_edges & ~hole[:, None])
    edge_from = xy[faces[adjoining, which]]
    edge_to = xy[faces[adjoining, (which+1) % 3]]
    tang = edge_to - edge_from
    wall_normal_ds = np.column_stack([-tang[:, 1], tang[:, 0]])
    # For counter-clockwise exterior triangles the body is on the right of
    # the fluid-boundary edge, so the solid-to-fluid normal points left.
    assert np.all(np.sum(wall_normal_ds * (edge_to + edge_from), axis=1) > 0)
    assert len(adjoining) == len(wall_ids)
    visc = []
    div_rms = []
    stride = 5
    visible = ~hole & (points[:, :, 0].mean(axis=1) >= -1.5) & (
        points[:, :, 0].mean(axis=1) <= 8) & (np.abs(points[:, :, 1].mean(axis=1)) <= 2)
    for k in range(len(t)):
        if k % stride == 0:
            omega, div = curl_div(fields[k])
            div_rms.append([t[k], *[np.sqrt(np.average(div[m]**2, weights=area[m])) for m in (~hole, visible)],
                            *np.percentile(abs(omega[visible]), [50, 90, 95, 99]), float(np.mean(abs(omega[visible]) >= 1))])
        a, b = gradients(fields[k, :, 0])
        c, d = gradients(fields[k, :, 1])
        nx, ny = wall_normal_ds.T
        visc.append([np.sum(2*a[adjoining]*nx + (b+c)[adjoining]*ny)/re,
                     np.sum((b+c)[adjoining]*nx + 2*d[adjoining]*ny)/re])
    visc = np.array(visc)
    estimated_force = pressure_force + visc

    dist = np.diff(t)
    p_gauge = fields[:, outlet, 2].mean(axis=1)
    reverse = fields[:, :, 0] < 0
    central_nodes = (abs(y) < .025) & (x > .5) & (x < 5)
    central_reverse = (fields[:, :, 0].mean(axis=0) < 0) & central_nodes
    region_stats = {}
    for label, mask in regions.items():
        values = fields[:, mask, :]
        region_stats[label] = {"nodes": int(mask.sum()), "area_share": float(weights[mask].sum()),
            "p01_p50_p99": quantiles(values.reshape(-1, 3), (1, 50, 99)),
            "mean": values.mean(axis=(0, 1)), "std": values.std(axis=(0, 1)),
            "min": values.min(axis=(0, 1)), "max": values.max(axis=(0, 1)),
            "rms_temporal_fluctuation": np.sqrt(np.mean(values.var(axis=0), axis=0))}
    region_stats["roi"] = {"nodes": int(roi.sum()), "area_share": float(weights[roi].sum())}

    # Dense-in-space temporal interpolation and harmonic extrapolation are
    # diagnostic references, with label access different from sparse PINNs.
    baseline = {}
    nominal_step = np.median(np.diff(t))
    eval_nodes = np.flatnonzero(roi & ~wall)
    w = weights[eval_nodes] / weights[eval_nodes].sum()
    for step in (2, 3, 5, 10, 15, 20, 30):
        seen = np.r_[np.arange(0, len(t)-1, step), len(t)-1]
        held = np.setdiff1d(np.arange(len(t)), seen)
        right = np.searchsorted(seen, held)
        left, right = seen[right-1], seen[right]
        alpha = ((t[held]-t[left])/(t[right]-t[left]))[:, None, None]
        truth = fields[held][:, eval_nodes]
        pred = (1-alpha)*fields[left][:, eval_nodes] + alpha*fields[right][:, eval_nodes]
        baseline[str(step)] = {"observed_frames": len(seen), "held_frames": len(held), "gap_over_period": step*nominal_step*f,
                               **error_metrics(truth, pred, w)}

    # Use the same hidden sites for each sensor layout and the same sparse
    # training snapshots for spatial interpolation. No pressure enters IDW.
    sparse = {}
    central_frames = np.arange(0, len(t), 5)
    omitted = np.setdiff1d(np.arange(len(t)), central_frames)
    right = np.searchsorted(central_frames, omitted)
    left = right - 1
    frac = ((t[omitted]-t[central_frames[left]]) /
            (t[central_frames[right]]-t[central_frames[left]]))[:, None, None]
    bins = np.floor((xy - [-1.5, -2.0]) / .5).astype(int)
    for seed in (11, 23, 37):
        rng = np.random.default_rng(seed)
        hold = rng.choice(eval_nodes, size=2000, replace=False)
        candidate = np.setdiff1d(eval_nodes, hold)
        truth = fields[central_frames][:, hold, :2]
        omitted_truth = fields[omitted][:, hold, :2]
        w_hold = weights[hold] / weights[hold].sum()
        seed_stats = {}
        for count in (64, 256, 1024):
            shuffled = rng.permutation(candidate)
            _, group = np.unique(bins[shuffled], axis=0, return_inverse=True)
            order = np.argsort(group, kind="stable")
            group_sorted = group[order]
            start = np.r_[True, np.diff(group_sorted) != 0]
            rank = np.arange(len(order)) - np.maximum.accumulate(np.where(start, np.arange(len(order)), 0))
            balanced = shuffled[order[np.lexsort((rng.random(len(order)), rank))[:count]]]
            for layout, ids in (("random_nodes", shuffled[:count]), ("balanced_0.5D_bins", balanced)):
                distances = np.sum((xy[hold, None] - xy[ids])**2, axis=2)
                nearest = np.argpartition(distances, 5, axis=1)[:, :6]
                # Skip known neighbors whose segment crosses solid when other
                # neighbors are available. Fall back to the nearest one only
                # if every candidate is occluded.
                q = xy[hold, None, :]
                d = xy[ids[nearest]] - q
                dist2 = np.take_along_axis(distances, nearest, axis=1)
                closest = q + np.clip(-np.sum(q*d, axis=2)/dist2, 0, 1)[:, :, None]*d
                clear = np.sum(closest**2, axis=2) > .5**2
                all_blocked = ~clear.any(axis=1)
                clear[all_blocked, np.argmin(dist2[all_blocked], axis=1)] = True
                inv = clear / dist2
                inv /= inv.sum(axis=1, keepdims=True)
                pred = np.einsum("tqkc,qk->tqc", fields[central_frames][:, ids[nearest], :2], inv)
                seed_stats[f"{count}_{layout}"] = {
                    **error_metrics(truth, pred, w_hold),
                    "joint_relative_l2_percent": error_metrics(
                        omitted_truth, (1-frac)*pred[left] + frac*pred[right], w_hold
                    )["relative_l2_percent"],
                    "distance_median": np.median(np.sqrt(distances.min(axis=1))),
                    "distance_p95": np.percentile(np.sqrt(distances.min(axis=1)), 95),
                    "near_body_sites": int(near[ids].sum())
                }
        sparse[str(seed)] = seed_stats

    # Compare repeated phases using the reaction-derived period. The target
    # timeline is shifted and interpolated on the stored field grid.
    period = 1/f
    eligible = np.flatnonzero(t + period <= t[-1])
    shifted = t[eligible] + period
    r = np.searchsorted(t, shifted)
    a = ((shifted-t[r-1])/(t[r]-t[r-1]))[:, None, None]
    phase_pred = (1-a)*fields[r-1][:, eval_nodes] + a*fields[r][:, eval_nodes]
    phase_repeat = error_metrics(fields[eligible][:, eval_nodes], phase_pred, w)
    del phase_pred

    def basis(tt, frequency, harmonics):
        phase = 2*np.pi*frequency*(tt-t[0])
        return np.column_stack([np.ones(len(tt))] + [fn(k*phase) for k in range(1, harmonics+1)
                                for fn in (np.cos, np.sin)])

    # Estimate frequency with a single probe using only the first six time units.
    first = t <= 406
    probe_v = fields[first, probe_ids[2], 1]
    frequencies = np.linspace(.13, .20, 1401)
    def fit_residual(freq):
        mat = basis(t[first], freq, 3)
        return np.linalg.norm(probe_v - mat @ np.linalg.lstsq(mat, probe_v, rcond=None)[0])
    f_train = min(frequencies, key=fit_residual)
    test = t >= 408
    first_values = fields[first][:, eval_nodes]
    test_values = fields[test][:, eval_nodes]
    harm = {}
    mean_pred = np.broadcast_to(first_values.mean(axis=0), test_values.shape)
    harm["train_mean"] = error_metrics(test_values, mean_pred, w)
    for k in (1, 3, 5):
        train_basis = basis(t[first], f_train, k)
        coef = np.linalg.lstsq(train_basis, first_values.reshape(first.sum(), -1), rcond=None)[0]
        pred = (basis(t[test], f_train, k) @ coef).reshape(test_values.shape)
        harm[str(k)] = error_metrics(test_values, pred, w)

    all_values = fields.reshape(-1, 3)
    snapshot_mean = fields.mean(axis=1)
    area_mean = np.einsum("tni,n->ti", fields, weights)
    area_second = np.einsum("tni,tni,n->i", fields, fields, weights) / len(t)
    metrics = {
        "field_md5": md5, "reaction_md5": hashlib.md5(raw_force).hexdigest(),
        "header": [re, ur], "frames": len(t), "nodes": len(x), "rows": len(t)*len(x),
        "time": [t[0], t[-1]], "dt": nominal_step, "dt_range": [dist.min(), dist.max()],
        "omega_frequency": f, "shedding_period": 1/f, "cycles": (t[-1]-t[0])*f,
        "extent": [xmin, xmax, ymin, ymax], "wall_radius": [radius[wall].min(), radius[wall].max()],
        "interior_nodes_r_lt_half_minus_tol": int((radius < .5-1e-4).sum()),
        "mesh_edges": quantiles(max_edge[~hole], (0, 1, 50, 95, 99, 100)),
        "triangles": [len(faces), int(hole.sum())], "mesh_area": area[~hole].sum(),
        "regions": region_stats, "node_quantiles": quantiles(all_values),
        "node_mean": all_values.mean(axis=0), "node_std": all_values.std(axis=0),
        "node_correlation": np.corrcoef(all_values.T), "area_mean": area_mean.mean(axis=0),
        "area_std": np.sqrt(area_second - area_mean.mean(axis=0)**2),
        "exact_zero_uv_all_frames_nodes": int(np.all(fields[:, :, :2] == 0, axis=(0, 2)).sum()),
        "wall_pressure_front_rear": [pwall[:, np.argmin(x[wall_ids])].mean(), pwall[:, np.argmax(x[wall_ids])].mean()],
        "reverse_share": [reverse.mean(axis=1).min(), reverse.mean(axis=1).max(),
                          (reverse @ weights).min(), (reverse @ weights).max()],
        "centerline_mean_reverse_sampled_extent": [x[central_reverse].min(), x[central_reverse].max()],
        "outlet_mean_pressure_span": [p_gauge.min(), p_gauge.max()],
        "reaction_rows": len(tr), "reaction_label": sorted(set(line[-1].decode() for line in lines)),
        "reaction_time": [tr[0], tr[-1]], "reaction_dt_quantiles": quantiles(np.diff(tr), (0, 50, 100)),
        "reaction_startup_first_row": reactions[0], "reaction_overlap_rows": int(overlap.sum()),
        "reaction_nearest_offset": [np.median(force_offset), np.max(force_offset)],
        "reaction_exact_time_matches": int(np.sum(force_offset == 0)),
        "reaction_window_stats": {"min": aligned_force.min(axis=0), "max": aligned_force.max(axis=0),
                                  "mean": aligned_force.mean(axis=0), "std": aligned_force.std(axis=0)},
        "lift_peaks": fy_peak, "drag_peaks": fx_peak,
        "lift_period_quantiles": quantiles(np.diff(fy_peak), (0, 50, 100)),
        "drag_period_median": np.median(np.diff(fx_peak)),
        "lift_overlap_spectrum_peak": spectrum(t, aligned_force[:, 1])[0][spectrum(t, aligned_force[:, 1])[1].argmax()],
        "drag_overlap_spectrum_peak": spectrum(t, aligned_force[:, 0])[0][spectrum(t, aligned_force[:, 0])[1].argmax()],
        "probes": probe_stats, "derivatives": np.array(div_rms),
        "pressure_force_mean": pressure_force.mean(axis=0), "viscous_force_mean": visc.mean(axis=0),
        "traction_force_rmse": np.sqrt(np.mean((estimated_force-aligned_force[:, :2])**2, axis=0)),
        "traction_force_corr": [np.corrcoef(estimated_force[:, k], aligned_force[:, k])[0, 1] for k in range(2)],
        "phase_repeat_diagnostic": phase_repeat, "sparse_spatial_baseline": sparse, "linear_baseline": baseline, "harmonic_fit_train_frequency": f_train, "harmonic_baseline": harm,
    }

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))

    (OUT / "metrics.json").write_text(json.dumps(metrics, default=convert, indent=2, allow_nan=False), encoding="utf-8")
    np.savetxt(OUT / "snapshot_diagnostics.csv",
               np.column_stack([t, snapshot_mean, area_mean, aligned_force, pressure_force, visc, estimated_force]),
               delimiter=",", fmt="%.9g", comments="",
               header="t,node_mean_u,node_mean_v,node_mean_p,area_mean_u,area_mean_v,area_mean_p,Fx_interp,Fy_interp,Mz_interp,Fpx,Fpy,Fvx,Fvy,Fx_est,Fy_est")
    print("Metrics saved", flush=True)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#22354a", "axes.titlesize": 11, "grid.alpha": .21})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].scatter(x, y, s=.21, alpha=.38, color=COLORS[0], rasterized=True)
    axes[0].set(xlabel="x / D", ylabel="y / D", title="Full exported mesh", aspect="equal")
    axes[1].scatter(x[roi], y[roi], s=.43, alpha=.40, color=COLORS[0], rasterized=True)
    axes[1].scatter(x[wall], y[wall], s=.8, color=COLORS[1], label="No-slip wall")
    axes[1].set(xlabel="x / D", ylabel="y / D", title="Near-body and wake crop", aspect="equal",
                xlim=(-1.5, 8), ylim=(-2, 2))
    axes[1].axvline(1.5, color="#6c7686", lw=1, ls="--")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, "mesh.png")

    k = 80
    omega, divergence = curl_div(fields[k])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for ax, col, label, limits, cmap in zip(axes.flat, fields[k].T, ["Streamwise velocity u", "Crossflow velocity v", "Pressure p"],
                                              [(-.3, 1.35), (-.6, .6), (-.75, .55)],
                                              ["viridis", "RdBu_r", "RdBu_r"]):
        im = ax.tripcolor(tri, col, shading="gouraud", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=.76)
    im = axes[1, 0].tripcolor(tri, facecolors=omega, cmap="RdBu_r", vmin=-5, vmax=5)
    axes[1, 0].set_title("Vorticity estimate")
    fig.colorbar(im, ax=axes[1, 0], shrink=.76)
    for ax, col, label, vmax in [(axes[1, 1], 1, "Temporal SD of v", .5),
                                  (axes[1, 2], 2, "Temporal SD of p", .2)]:
        im = ax.tripcolor(tri, fields[:, :, col].std(axis=0), shading="gouraud", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=.76)
    for ax in axes.flat:
        ax.set(xlim=(-1.5, 8), ylim=(-2, 2), aspect="equal", xlabel="x / D", ylabel="y / D")
    save(fig, "fields.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6), layout="constrained")
    for j, (ax, name, force) in enumerate(zip(axes[:, 0], ("Streamwise reaction F?", "Transverse reaction F?"), (fx, fy))):
        ax.plot(tr[overlap], force[overlap], color=COLORS[j], label="Recorded reaction")
        ax.plot(t, estimated_force[:, j], ls="--", color="#354654", lw=1, label="Estimated wall traction")
        ax.set(title=name, xlabel="t U8 / D", ylabel="Exported units")
        ax.grid()
        ax.legend(frameon=False, fontsize=8)
    for j, (name, force) in enumerate(zip(("F?", "F?"), (fx, fy))):
        fq, pw = spectrum(tr[late], force[late])
        axes[0, 1].plot(fq, pw, color=COLORS[j], label=name)
    axes[0, 1].set(xlim=(0, .8), title="Late reaction spectrum, t = 400",
                   xlabel="Frequency (U8 / D)", ylabel="Relative spectral power")
    axes[0, 1].legend(frameon=False)
    for j, i in enumerate(probe_ids[:4]):
        axes[1, 1].plot(t, fields[:, i, 1], color=(COLORS+["#8158a2"])[j], label=str(probes[j]))
    axes[1, 1].set(title="Crossflow velocity at four probe locations",
                   xlabel="t U8 / D", ylabel="v / U8")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes[:, 1]:
        ax.grid()
    save(fig, "dynamics.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), layout="constrained")
    for j, label in enumerate(("u", "v", "p")):
        axes[0].plot([row["gap_over_period"] for row in baseline.values()],
                     [row["relative_l2_percent"][j] for row in baseline.values()],
                     marker="o", color=COLORS[j], label=label)
    axes[0].set(xlabel="Observation gap / shedding period", ylabel="Relative L² error (%)",
                title="Dense-space linear time interpolation")
    axes[0].legend(frameon=False)
    axes[0].grid()
    dd = np.array(div_rms)
    axes[1].plot(dd[:, 0], dd[:, 1], color=COLORS[0], label="Full domain")
    axes[1].plot(dd[:, 0], dd[:, 2], color=COLORS[1], label="Near body + wake")
    axes[1].set(xlabel="t U8 / D", ylabel="Area-weighted RMS", title="Divergence from linear triangle gradients")
    axes[1].legend(frameon=False)
    axes[1].grid()
    save(fig, "diagnostics.png")


if __name__ == "__main__":
    main()





