"""Single entry point for generation, detection, and paper experiments."""

from __future__ import annotations

import argparse
import runpy
import sys


COMMANDS = {
    "generate": "denomark.core.generate",
    "detect": "denomark.evaluation.detect",
    "baseline": {
        "clean": {
            "generate": "denomark.baselines.clean.generate",
        },
        "dlm_kgw": {
            "generate": "denomark.baselines.dlm_kgw.generate",
            "detect": "denomark.baselines.dlm_kgw.detect",
        },
        "dgmark": {
            "generate": "denomark.baselines.dgmark.generate",
            "detect": "denomark.baselines.dgmark.detect",
        },
        "patternmark": {
            "generate": "denomark.baselines.patternmark.generate",
            "detect": "denomark.baselines.patternmark.detect",
        },
        "umr": {
            "generate": "denomark.baselines.umr.generate",
            "detect": "denomark.baselines.umr.detect",
        },
        "block_best_of_k": {
            "generate": "denomark.baselines.block_best_of_k.generate",
            "detect": "denomark.evaluation.detect",
        },
        "pmark": {
            "generate": "denomark.baselines.pmark.generate",
        },
        "semstamp": {
            "generate": "denomark.baselines.semstamp.generate",
        },
        "semantic-detect": "denomark.baselines.semantic_detect",
    },
    "attack": {
        "sentence": "denomark.attacks.gpt_sentence_level",
        "document": "denomark.attacks.gpt_document_level",
        "parrot": "denomark.attacks.parrot",
        "dipper": "denomark.attacks.dipper",
        "backtranslation": "denomark.attacks.backtranslation",
        "token": "denomark.attacks.token_level",
        "mixed-length": "denomark.attacks.mixed_length",
        "variable-compression": "denomark.attacks.variable_compression",
    },
    "data": {
        "prompts": "denomark.data.prepare_waterbench",
        "filter": "denomark.data.filter",
        "negatives": "denomark.data.negatives",
    },
    "encoder": {
        "prepare": "denomark.encoder.prepare",
        "paraphrase": "denomark.encoder.paraphrase",
        "merge": "denomark.encoder.merge",
        "train": "denomark.encoder.train",
    },
    "evaluate": {
        "ppl": "denomark.evaluation.ppl",
        "ppl-delta": "denomark.evaluation.compare_ppl",
        "judge": "denomark.evaluation.judge",
        "collect": "denomark.evaluation.collect_metrics",
        "plot": "denomark.evaluation.plot_figures",
        "runtime": "denomark.evaluation.summarize_runtime",
    },
    "downstream": {
        "prepare": "denomark.experiments.downstream.prepare",
        "run": "denomark.experiments.downstream.run",
        "aggregate": "denomark.experiments.downstream.aggregate",
    },
}


def main() -> None:
    remaining = sys.argv[1:]
    node = COMMANDS
    program = "python -m denomark"
    while isinstance(node, dict):
        parser = argparse.ArgumentParser(prog=program)
        parser.add_argument("command", choices=tuple(node))
        if not remaining:
            parser.print_help()
            return
        choice = parser.parse_args(remaining[:1]).command
        program += " " + choice
        node = node[choice]
        remaining = remaining[1:]
    sys.argv = [program, *remaining]
    runpy.run_module(node, run_name="__main__")


if __name__ == "__main__":
    main()
