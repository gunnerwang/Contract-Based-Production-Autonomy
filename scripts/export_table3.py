#!/usr/bin/env python3
"""Generate Table 3 from a completed experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.results_exporter import ResultsExporter


def main() -> None:
    experiment = CBPAExperiment(use_llm=False)
    result = experiment.run_all_phases()

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/results"
    exporter = ResultsExporter(output_dir)
    exporter.export_all(result)

    # Print the LaTeX table
    latex_path = Path(output_dir) / "table3.tex"
    if latex_path.exists():
        print(latex_path.read_text())


if __name__ == "__main__":
    main()
