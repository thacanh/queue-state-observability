"""Regenerate statistics or replay locked evaluations into a separate directory."""
import argparse
import shutil

import crossed_statistics
import replay_io as base


def compare_protocol(actual, expected, source_keys):
    """Require the original evaluation settings for each release replay."""
    def configuration(directory):
        return {key: value for key, value in base.load_json(directory / "protocol.json").items()
                if key not in source_keys}

    if configuration(actual) != configuration(expected):
        raise ValueError(f"Evaluation configuration differs from {expected.name}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("summaries", "crossed", "transfer"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    directory = base.ROOT / "regenerated"
    if args.task == "transfer":
        import replay_transfer as transfer

        transfer.EXT = directory / "azure2021"
        transfer.EXT.mkdir(parents=True, exist_ok=True)
        transfer.main()
        compare_protocol(transfer.EXT, base.OUT / "external", {"identity", "code"})
        return
    target = directory / "crossed_observations"
    target.mkdir(parents=True, exist_ok=True)
    if args.task == "crossed":
        import replay_crossed

        replay_crossed.OUT = target
        replay_crossed.main(["--workers", str(args.workers)])
        compare_protocol(target, crossed_statistics.OUT, {"identity", "files", "training"})
    else:
        for name in ("workload_by_trace.csv", "queue_by_trace_seed.csv", "factorial_pairs.csv", "status.json"):
            shutil.copyfile(crossed_statistics.OUT / name, target / name)
    crossed_statistics.OUT = target
    crossed_statistics.main()


if __name__ == "__main__":
    main()
