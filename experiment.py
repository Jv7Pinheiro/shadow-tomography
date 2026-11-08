"""Reproducible minimax scaling experiment for anticommuting Pauli targets.

Adapts shadow.py's spectral Pauli measurements and frequency estimator. Binomial
counts have exactly the same law as the original independent sign samples.
Run: python3 experiment.py --output results
"""

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OBSERVABLE_COUNTS = (3, 5, 9, 17, 33)
PRECISIONS = (0.05, 0.075, 0.10)
TAUS = (0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0, 4.0)
STATES = ("center", "hypercube", "boundary")
ESTIMATORS = ("mean", "shrinkage", "projection")
LABELS = {"mean": "Sample mean", "shrinkage": "Shrinkage", "projection": "Projection"}
COLORS = {"mean": "#245b99", "shrinkage": "#bd5532", "projection": "#328060"}
RADIUS = 0.5
UPSTREAM_COMMIT = "4b009a1652e4794ebe4a602d81eb66e4bd43807c"


def anticommuting_strings(n):
    strings = []
    for site in range(n):
        for axis in "XY":
            strings.append("Z" * site + axis + "I" * (n - site - 1))
    return strings + ["Z" * n]


def verify_measurement_model():
    """Check symbolic anticommutation at all sizes and dense Born laws up to 4 qubits."""
    identity = np.eye(2, dtype=complex)
    paulis = {
        "I": identity,
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1.0, -1.0]).astype(complex),
    }
    maximum_probability_error = 0.0
    for m in OBSERVABLE_COUNTS:
        n = (m - 1) // 2
        strings = anticommuting_strings(n)
        assert len(strings) == m
        for j, left in enumerate(strings):
            for right in strings[j + 1:]:
                odd_sites = sum(a != b and a != "I" and b != "I" for a, b in zip(left, right))
                assert odd_sites % 2 == 1
        if n > 4:
            continue
        operators = []
        for word in strings:
            matrix = np.array([[1.0]], dtype=complex)
            for axis in word:
                matrix = np.kron(matrix, paulis[axis])
            operators.append(matrix)
        dimension = 2 ** n
        for q in (0.0, 0.2, 0.3, 0.4, 0.5):
            theta = q * np.where(np.arange(m) % 2, -1, 1) / math.sqrt(m)
            rho = np.eye(dimension, dtype=complex)
            for coefficient, operator in zip(theta, operators):
                rho += coefficient * operator
            rho /= dimension
            eigenvalues = np.linalg.eigvalsh(rho)
            np.testing.assert_allclose(eigenvalues[0], (1 - q) / dimension, atol=1e-12)
            np.testing.assert_allclose(eigenvalues[-1], (1 + q) / dimension, atol=1e-12)
            for coefficient, operator in zip(theta, operators):
                for sign in (-1, 1):
                    projector = (np.eye(dimension) + sign * operator) / 2
                    born = np.trace(rho @ projector).real
                    maximum_probability_error = max(maximum_probability_error, abs(born - (1 + sign * coefficient) / 2))
    assert maximum_probability_error < 1e-12
    return {"dense_max_qubits": 4, "maximum_born_probability_error": maximum_probability_error}


def wilson_interval(successes, trials):
    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1 + z * z / trials
    center = (fraction + z * z / (2 * trials)) / denominator
    width = z * math.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials * trials)) / denominator
    return center - width, center + width


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(repetitions, seed):
    rows = []
    for m in OBSERVABLE_COUNTS:
        for precision_index, epsilon in enumerate(PRECISIONS):
            for state_index, state in enumerate(STATES):
                q = {"center": 0.0, "hypercube": 4 * epsilon, "boundary": RADIUS}[state]
                theta = q * np.where(np.arange(m) % 2, -1, 1) / math.sqrt(m)
                for tau_index, tau in enumerate(TAUS):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, m, precision_index, state_index, tau_index]))
                    h = math.ceil(tau * m / epsilon ** 2)
                    copies = m * h
                    counts = rng.binomial(h, (1 + theta) / 2, size=(repetitions, m))
                    mean = 2 * counts / h - 1
                    shrinkage = copies * RADIUS ** 2 / (copies * RADIUS ** 2 + m ** 2)
                    norms = np.linalg.norm(mean, axis=1)
                    scales = np.minimum(1.0, RADIUS / np.maximum(norms, np.finfo(float).tiny))
                    estimates = {"mean": mean, "shrinkage": shrinkage * mean, "projection": mean * scales[:, None]}
                    baseline_errors = np.sum((mean - theta) ** 2, axis=1)
                    baseline_risk = (m ** 2 - m * q ** 2) / copies
                    for estimator in ESTIMATORS:
                        estimate = estimates[estimator]
                        errors = np.sum((estimate - theta) ** 2, axis=1)
                        failures = int(np.count_nonzero(errors > epsilon ** 2))
                        lower, upper = wilson_interval(failures, repetitions)
                        analytic_mse = math.nan
                        if estimator == "mean":
                            analytic_mse = baseline_risk
                        elif estimator == "shrinkage":
                            analytic_mse = shrinkage ** 2 * baseline_risk + (1 - shrinkage) ** 2 * q ** 2
                        else:
                            assert np.all(errors <= baseline_errors + 1e-12)
                        rows.append({
                            "m": m, "qubits": (m - 1) // 2, "epsilon": epsilon,
                            "state": state, "q": q, "tau_requested": tau,
                            "h": h, "N": copies, "tau_actual": copies * epsilon ** 2 / m ** 2,
                            "estimator": estimator, "repetitions": repetitions,
                            "failures": failures, "failure_probability": failures / repetitions,
                            "failure_ci_lower": lower, "failure_ci_upper": upper,
                            "mse": float(np.mean(errors)),
                            "mse_se": float(np.std(errors, ddof=1) / math.sqrt(repetitions)),
                            "analytic_mse": analytic_mse,
                            "paired_mse_difference": float(np.mean(errors - baseline_errors)),
                            "paired_mse_difference_se": float(np.std(errors - baseline_errors, ddof=1) / math.sqrt(repetitions)),
                            "projection_fraction": float(np.mean(norms > RADIUS)),
                            "shrinkage_factor": shrinkage,
                        })
        print(f"Completed m={m}", flush=True)
    return rows


def estimate_thresholds(rows):
    thresholds = []
    for m in OBSERVABLE_COUNTS:
        for epsilon in PRECISIONS:
            for estimator in ESTIMATORS:
                curve = []
                for tau in TAUS:
                    cells = [row for row in rows if row["m"] == m and row["epsilon"] == epsilon and row["estimator"] == estimator and row["tau_requested"] == tau]
                    worst = max(cells, key=lambda row: row["failure_probability"])
                    curve.append((worst["N"], worst["failure_probability"], worst["state"], max(row["failure_ci_upper"] for row in cells)))
                passing_index = next((index for index, cell in enumerate(curve) if cell[1] <= 1 / 3), None)
                if passing_index is None or passing_index == 0:
                    raise RuntimeError("Budget grid does not bracket the success threshold")
                previous = curve[passing_index - 1]
                current = curve[passing_index]
                fraction = (previous[1] - 1 / 3) / (previous[1] - current[1])
                crossing = math.exp(math.log(previous[0]) + fraction * math.log(current[0] / previous[0]))
                confidence_index = next((index for index, cell in enumerate(curve) if cell[3] <= 1 / 3), None)
                thresholds.append({
                    "m": m, "epsilon": epsilon, "estimator": estimator,
                    "N_interpolated": crossing,
                    "scaled_N_interpolated": crossing * epsilon ** 2 / m ** 2,
                    "N_last_failing_grid": previous[0], "N_first_passing_grid": current[0],
                    "worst_state_at_first_pass": current[2],
                    "N_first_pointwise_ci_pass": curve[confidence_index][0] if confidence_index is not None else math.nan,
                })
    return thresholds


def make_figure(rows, thresholds, output, report_layout=False):
    font_size = 8 if report_layout else 10
    plt.rcParams.update({"font.size": font_size, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    if report_layout:
        figure = plt.figure(figsize=(7.2, 6.5), layout="constrained")
        grid = figure.add_gridspec(2, 2)
        axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])]
    else:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    epsilon = 0.075
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(OBSERVABLE_COUNTS)))
    for m, color in zip(OBSERVABLE_COUNTS, palette):
        cells = [row for row in rows if row["m"] == m and row["epsilon"] == epsilon and row["state"] == "hypercube" and row["estimator"] == "mean"]
        tau = np.array([row["tau_actual"] for row in cells])
        normalized = np.array([row["N"] * row["mse"] / (m ** 2 - m * row["q"] ** 2) for row in cells])
        uncertainty = np.array([1.96 * row["N"] * row["mse_se"] / (m ** 2 - m * row["q"] ** 2) for row in cells])
        axes[0].errorbar(tau, normalized, yerr=uncertainty, marker="o", markersize=3, linewidth=1, color=color, label=f"m={m}")
    axes[0].axhline(1, color="0.3", linestyle="--", linewidth=1)
    axes[0].set(xscale="log", xlabel=r"Scaled copies $N\varepsilon^2/m^2$", ylabel=r"$N\,\mathrm{MSE}/(m^2-mq^2)$", title=r"(a) Exact MSE check, $\varepsilon=0.075$")
    axes[0].legend(ncol=2, fontsize=6 if report_layout else 8)
    line_styles = {3: ":", 9: "--", 33: "-"}
    for estimator in ESTIMATORS:
        for m in (3, 9, 33):
            cells = [row for row in rows if row["m"] == m and row["epsilon"] == epsilon and row["state"] == "hypercube" and row["estimator"] == estimator]
            x = np.array([row["tau_actual"] for row in cells])
            y = np.array([row["failure_probability"] for row in cells])
            axes[1].plot(x, y, color=COLORS[estimator], linestyle=line_styles[m], linewidth=1.4)
            axes[1].fill_between(x, [row["failure_ci_lower"] for row in cells], [row["failure_ci_upper"] for row in cells], color=COLORS[estimator], alpha=0.08)
    axes[1].axhline(1 / 3, color="0.3", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", xlabel=r"Scaled copies $N\varepsilon^2/m^2$", ylabel="Failure probability", ylim=(-0.02, 1.02), title=r"(b) Hypercube states, $q=4\varepsilon$")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[key], label=LABELS[key]) for key in ESTIMATORS]
    handles += [Line2D([0], [0], color="0.35", linestyle=line_styles[m], label=f"m={m}") for m in line_styles]
    axes[1].legend(handles=handles, fontsize=6 if report_layout else 8, ncol=2)
    markers = {0.05: "o", 0.075: "s", 0.1: "^"}
    for estimator in ESTIMATORS:
        for precision in PRECISIONS:
            cells = [row for row in thresholds if row["estimator"] == estimator and row["epsilon"] == precision]
            axes[2].plot([row["m"] for row in cells], [row["N_interpolated"] * precision ** 2 for row in cells], color=COLORS[estimator], marker=markers[precision], markersize=4, linewidth=1, alpha=0.8)
    sizes = np.array(OBSERVABLE_COUNTS)
    axes[2].plot(sizes, sizes ** 2, color="0.3", linestyle="--", linewidth=1, label=r"$m^2$")
    axes[2].set(xscale="log", yscale="log", xlabel=r"Observables $m$", ylabel=r"Empirical threshold $N_{2/3}\varepsilon^2$", title="(c) Largest failure rate over three radii")
    handles = [Line2D([0], [0], marker=markers[e], color="0.35", linestyle="none", label=f"epsilon={e:g}") for e in PRECISIONS]
    handles += [Line2D([0], [0], color="0.3", linestyle="--", label=r"$m^2$")]
    axes[2].legend(handles=handles, fontsize=7 if report_layout else 8)
    basename = "minimax_experiment_report" if report_layout else "minimax_experiment"
    figure.savefig(output / (basename + ".pdf"), bbox_inches="tight")
    figure.savefig(output / (basename + ".png"), dpi=210, bbox_inches="tight")
    plt.close(figure)


def summarize(rows, thresholds, validations, elapsed, repetitions, seed):
    analytic_rows = [row for row in rows if math.isfinite(row["analytic_mse"])]
    z_scores = [(row["mse"] - row["analytic_mse"]) / row["mse_se"] for row in analytic_rows]
    relative_errors = [abs(row["mse"] / row["analytic_mse"] - 1) for row in analytic_rows]
    fits = []
    for estimator in ESTIMATORS:
        selected = [row for row in thresholds if row["estimator"] == estimator]
        design = np.array([[1, math.log(row["m"]), math.log(1 / row["epsilon"])] for row in selected])
        coefficients = np.linalg.lstsq(design, np.log([row["N_interpolated"] for row in selected]), rcond=None)[0]
        fits.append({"estimator": estimator, "m_exponent": float(coefficients[1]), "inverse_epsilon_exponent": float(coefficients[2]), "scaled_threshold_min": min(row["scaled_N_interpolated"] for row in selected), "scaled_threshold_max": max(row["scaled_N_interpolated"] for row in selected)})
    return {
        "upstream_commit": UPSTREAM_COMMIT, "seed": seed, "radius": RADIUS,
        "m_values": OBSERVABLE_COUNTS, "epsilons": PRECISIONS, "taus": TAUS,
        "repetitions_per_cell": repetitions, "measurement_cells": len(rows) // len(ESTIMATORS),
        "independent_measurement_repetitions": repetitions * len(rows) // len(ESTIMATORS),
        "estimators_paired_on_identical_data": True,
        "python_version": platform.python_version(), "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "elapsed_seconds": elapsed, "validations": validations,
        "analytic_risk_check": {"comparisons": len(z_scores), "maximum_absolute_standardized_error": max(abs(z) for z in z_scores), "maximum_relative_error": max(relative_errors)},
        "threshold_method": "First grid crossing of maximum empirical failure across three radii; log-N interpolation. CIs are pointwise Wilson intervals, not simultaneous coverage.",
        "scaling_fits": fits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261108)
    arguments = parser.parse_args()
    if arguments.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validations = verify_measurement_model()
    rows = run_sweep(arguments.repetitions, arguments.seed)
    thresholds = estimate_thresholds(rows)
    write_csv(arguments.output / "results.csv", rows)
    write_csv(arguments.output / "thresholds.csv", thresholds)
    make_figure(rows, thresholds, arguments.output)
    make_figure(rows, thresholds, arguments.output, report_layout=True)
    summary = summarize(rows, thresholds, validations, time.perf_counter() - started, arguments.repetitions, arguments.seed)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
