import logging
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logger(name: str, log_dir: str = "results", level: int = logging.INFO) -> logging.Logger:
    """Create a logger writing to both console and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results dict to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"Results saved to {output_path}")


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable objects for JSON dumping."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class ExperimentLogger:
    """Tracks experiment metadata, timing, and metrics, then saves to JSON."""

    def __init__(self, experiment_name: str, config: Dict, log_dir: str = "results"):
        self.name = experiment_name
        self.config = config
        self.log_dir = log_dir
        self.logger = setup_logger(experiment_name, log_dir)
        self._start_time: Optional[float] = None
        self._metrics: Dict[str, Any] = {}

    def start(self) -> None:
        self._start_time = time.time()
        self.logger.info(f"Experiment '{self.name}' started.")

    def log_metric(self, key: str, value: Any) -> None:
        self._metrics[key] = value
        self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def finish(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self.logger.info(f"Experiment '{self.name}' finished in {elapsed:.1f}s.")

        record = {
            "experiment": self.name,
            "config": self.config,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat()
            if self._start_time else None,
            "elapsed_seconds": elapsed,
            "metrics": self._metrics,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.log_dir, f"{self.name}_{timestamp}.json")
        save_results(record, out_path)
        return record
