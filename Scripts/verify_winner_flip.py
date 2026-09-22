#!/usr/bin/env python3
"""Offline gate for energy-aware tuning results (GPU-free, stdlib only).

Reads a legacy KTT JSON result file (as saved by ``Tuner::SaveResults`` with
``OutputFormat::JSON``) and re-scores every kernel result with a weighted
energy-performance objective::

    score = performanceWeight * duration / referenceDuration
          + energyWeight     * energy   / referenceEnergy

Two modes:

1. Offline gate (single file)::

       verify_winner_flip.py TuningOutput.json --weights 0.5,0.5 --refs first

   Fails when any result lacks energy data (proves the NVML/IGCL power path
   produced nothing) or when the weighted scores cannot be computed.
   Reports the pure-duration winner versus the weighted winner; a difference
   ("flip") is informational only, since it depends on hardware.

2. Simulation check (two files)::

       verify_winner_flip.py SimOutput.json --expect-winner-from TuningOutput.json \
           --weights 0.5,0.5 --refs first

   Additionally requires the weighted winner of the simulated run to match
   the weighted winner of the reference run, proving loader SimulateTuning
   reproduces the offline prediction.

Ratios cancel the file's time unit, so no unit conversion is needed here.
"""

import argparse
import json
import math
import sys


def parse_weights(text):
    try:
        performance, energy = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "weights must look like '0.5,0.5' (performance,energy)"
        )

    if not (math.isfinite(performance) and math.isfinite(energy)):
        raise argparse.ArgumentTypeError("weights must be finite")
    if performance < 0.0 or energy < 0.0 or performance + energy <= 0.0:
        raise argparse.ArgumentTypeError(
            "weights must be non-negative with a positive sum"
        )

    total = performance + energy
    return performance / total, energy / total


def load_results(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("Results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: no 'Results' array found")

    return results


def total_energy(entry):
    """Total energy in joules for one kernel result entry."""
    computations = entry.get("ComputationResults")
    if not computations:
        return math.nan

    total = 0.0
    for computation in computations:
        # NOTE: only direct Joule measurements count here. Deriving energy
        # from power-only entries would need the file's time unit, so such
        # entries are treated as missing energy data by this gate.
        energy = computation.get("EnergyConsumption")
        if energy is None:
            return math.nan
        total += float(energy)

    return total


def missing_energy_indices(results):
    return [
        index
        for index, entry in enumerate(results)
        if not math.isfinite(total_energy(entry))
    ]


def score_entry(entry, performance_weight, energy_weight,
                reference_duration, reference_energy):
    duration = float(entry["TotalDuration"])
    energy = total_energy(entry)
    return (
        performance_weight * duration / reference_duration
        + energy_weight * energy / reference_energy
    )


def configuration_label(entry):
    configuration = entry.get("Configuration")
    if isinstance(configuration, list):
        return ", ".join(
            f"{pair.get('Name')}={pair.get('Value')}" for pair in configuration
        )
    return str(configuration)


def analyze(path, performance_weight, energy_weight):
    results = load_results(path)

    bad = missing_energy_indices(results)
    if bad:
        raise ValueError(
            f"{path}: {len(bad)}/{len(results)} results lack energy data "
            f"(first at index {bad[0]}); power measurement produced nothing"
        )

    reference_duration = float(results[0]["TotalDuration"])
    reference_energy = total_energy(results[0])
    if reference_duration <= 0.0 or not math.isfinite(reference_energy) \
            or reference_energy <= 0.0:
        raise ValueError(f"{path}: first result has unusable references")

    duration_winner = min(
        range(len(results)), key=lambda i: float(results[i]["TotalDuration"])
    )
    scores = [
        score_entry(entry, performance_weight, energy_weight,
                    reference_duration, reference_energy)
        for entry in results
    ]
    weighted_winner = min(range(len(results)), key=lambda i: scores[i])

    return {
        "path": path,
        "count": len(results),
        "reference_duration": reference_duration,
        "reference_energy": reference_energy,
        "duration_winner": duration_winner,
        "weighted_winner": weighted_winner,
        "scores": scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="KTT legacy JSON output to check")
    parser.add_argument(
        "--weights",
        default="0.5,0.5",
        type=parse_weights,
        help="performance,energy weights (default: 0.5,0.5)",
    )
    parser.add_argument(
        "--refs",
        choices=["first"],
        default="first",
        help="reference config for normalization (default: first)",
    )
    parser.add_argument(
        "--expect-winner-from",
        default=None,
        metavar="REFERENCE_JSON",
        help="require the weighted winner to match this file's weighted winner",
    )
    args = parser.parse_args(argv)

    performance_weight, energy_weight = args.weights

    try:
        report = analyze(args.results, performance_weight, energy_weight)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"{report['path']}: {report['count']} results, "
        f"refs T={report['reference_duration']:.0f} E={report['reference_energy']:.4f}J, "
        f"weights {performance_weight:.2f}/{energy_weight:.2f}"
    )
    print(
        f"duration winner: [{report['duration_winner']}] "
        f"{configuration_label(load_results(args.results)[report['duration_winner']])}"
    )
    print(
        f"weighted winner: [{report['weighted_winner']}] "
        f"{configuration_label(load_results(args.results)[report['weighted_winner']])}"
    )

    if report["duration_winner"] != report["weighted_winner"]:
        print("winner flip observed: energy term changed the optimum")
    else:
        print("no winner flip: duration optimum already wins on energy")

    if args.expect_winner_from is not None:
        try:
            reference = analyze(
                args.expect_winner_from, performance_weight, energy_weight
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1

        if report["weighted_winner"] != reference["weighted_winner"]:
            print(
                f"FAIL: simulated winner [{report['weighted_winner']}] != "
                f"reference winner [{reference['weighted_winner']}]",
                file=sys.stderr,
            )
            return 1
        print(
            f"simulation matches reference winner [{reference['weighted_winner']}]"
        )

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
