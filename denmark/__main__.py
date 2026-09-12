"""Single entry point for generation, detection, and paper experiments."""

from __future__ import annotations

import argparse
import runpy
import sys


COMMANDS = {
    "generate": "denmark.core.generate",
    "detect": "denmark.evaluation.detect",
    "baseline": {
        "clean": {
            "generate": "denmark.baselines.clean.generate",
        },
        "dlm_kgw": {
            "generate": "denmark.baselines.dlm_kgw.generate",
            "detect": "denmark.baselines.dlm_kgw.detect",
        },
        "dgmark": {
            "generate": "denmark.baselines.dgmark.generate",
            "detect": "denmark.baselines.dgmark.detect",
        },
        "patternmark": {
            "generate": "denmark.baselines.patternmark.generate",
            "detect": "denmark.baselines.patternmark.detect",
        },
        "umr": {
            "generate": "denmark.baselines.umr.generate",
            "detect": "denmark.baselines.umr.detect",
        },
        "block_best_of_k": {
            "generate": "denmark.baselines.block_best_of_k.generate",
            "detect": "denmark.evaluation.detect",
        },
        "pmark": {
            "generate": "denmark.baselines.pmark.generate",
        },
        "semstamp": {
            "generate": "denmark.baselines.semstamp.generate",
        },
        "semantic-detect": "denmark.baselines.semantic_detect",
    },
    "attack": {
        "sentence": "denmark.attacks.gpt_sentence_level",
        "document": "denmark.attacks.gpt_document_level",
        "parrot": "denmark.attacks.parrot",
        "dipper": "denmark.attacks.dipper",
        "backtranslation": "denmark.attacks.backtranslation",
        "token": "denmark.attacks.token_level",
        "mixed-length": "denmark.attacks.mixed_length",
        "variable-compression": "denmark.attacks.variable_compression",
    },
    "data": {
        "prompts": "denmark.data.prepare_waterbench",
        "filter": "denmark.data.filter",
        "negatives": "denmark.data.negatives",
    },
    "encoder": {
        "prepare": "denmark.encoder.prepare",
        "paraphrase": "denmark.encoder.paraphrase",
        "merge": "denmark.encoder.merge",
        "train": "denmark.encoder.train",
    },
    "evaluate": {
        "ppl": "denmark.evaluation.ppl",
        "ppl-delta": "denmark.evaluation.compare_ppl",
        "judge": "denmark.evaluation.judge",
        "collect": "denmark.evaluation.collect_metrics",
        "plot": "denmark.evaluation.plot_figures",
        "runtime": "denmark.evaluation.summarize_runtime",
    },
    "downstream": {
        "prepare": "denmark.experiments.downstream.prepare",
        "run": "denmark.experiments.downstream.run",
        "aggregate": "denmark.experiments.downstream.aggregate",
    },
}


def main() -> None:
    remaining = sys.argv[1:]
    node = COMMANDS
    program = "python -m denmark"
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
