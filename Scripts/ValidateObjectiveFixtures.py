#!/usr/bin/env python3
"""GPU-free validation of tuning-loader Objective fixtures (stdlib only).

Checks every ``*Script*.json`` loader script under ``Tutorials/``:

* the file parses as JSON and carries the sections the loader requires
  (``ConfigurationSpace``, ``Search``, ``General``, ``KernelSpecification``),
* any ``Objective`` block matches ``TuningLoader/TuningSchema.h`` (``Name``
  is a known objective; weighted objectives carry ``PerformanceWeight``,
  ``EnergyWeight``, ``ReferenceDuration`` and ``ReferenceEnergy``),
* weighted objectives satisfy the ``WeightedEnergyPerformanceObjective``
  constructor constraints (finite, non-negative weights with a positive sum;
  finite, positive references).

Run from anywhere inside the repository::

    python3 Scripts/ValidateObjectiveFixtures.py
"""

import glob
import json
import math
import os
import sys

KNOWN_OBJECTIVES = ("Duration", "WeightedEnergyPerformance")
REQUIRED_SECTIONS = (
    "ConfigurationSpace",
    "Search",
    "General",
    "KernelSpecification",
)

failures = []


def fail(message):
    failures.append(message)
    print(f"FAIL: {message}", file=sys.stderr)


def check_weighted_objective(path, objective):
    for key in (
        "PerformanceWeight",
        "EnergyWeight",
        "ReferenceDuration",
        "ReferenceEnergy",
    ):
        if key not in objective:
            fail(f"{path}: weighted objective misses '{key}'")
            return

    performance = objective["PerformanceWeight"]
    energy = objective["EnergyWeight"]
    reference_duration = objective["ReferenceDuration"]
    reference_energy = objective["ReferenceEnergy"]

    for key, value in (
        ("PerformanceWeight", performance),
        ("EnergyWeight", energy),
        ("ReferenceDuration", reference_duration),
        ("ReferenceEnergy", reference_energy),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(f"{path}: objective '{key}' must be a finite number")
            return

    if performance < 0.0 or energy < 0.0 or performance + energy <= 0.0:
        fail(
            f"{path}: objective weights must be non-negative "
            "with a positive sum"
        )
    if reference_duration <= 0.0 or reference_energy <= 0.0:
        fail(f"{path}: objective references must be greater than zero")


def check_objective(path, script):
    objective = script.get("Objective")
    if objective is None:
        return

    name = objective.get("Name")
    if name not in KNOWN_OBJECTIVES:
        fail(f"{path}: unknown objective '{name}'")
        return

    if name == "WeightedEnergyPerformance":
        check_weighted_objective(path, objective)


def check_script(path):
    try:
        with open(path, encoding="utf-8") as handle:
            script = json.load(handle)
    except (OSError, ValueError) as error:
        fail(f"{path}: invalid JSON ({error})")
        return

    for section in REQUIRED_SECTIONS:
        if section not in script:
            fail(f"{path}: misses required section '{section}'")

    check_objective(path, script)
    print(f"checked {path}")


def find_repo_root(start):
    current = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(current, "premake5.lua")) and os.path.isdir(
            os.path.join(current, "Tutorials")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("repository root not found")
        current = parent


def main():
    try:
        root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    scripts = sorted(
        glob.glob(os.path.join(root, "Tutorials", "*", "*Script*.json"))
    )
    if not scripts:
        print("FAIL: no loader scripts found under Tutorials/", file=sys.stderr)
        return 1

    for path in scripts:
        check_script(path)

    if failures:
        print(f"{len(failures)} fixture problem(s) found", file=sys.stderr)
        return 1

    print(f"PASS: {len(scripts)} loader script(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
