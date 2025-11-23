from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, List, Optional

import modal


APP_NAME = "pii-ner-assignment"
REMOTE_PROJECT_DIR = PurePosixPath("/root/project")
REMOTE_PROJECT_DIR_PATH = Path("/root/project")
LOCAL_PROJECT_DIR = Path(__file__).parent
DEFAULT_TIMEOUT = 60 * 60  # 1 hour


app = modal.App(APP_NAME)
MODEL_VOLUME = modal.Volume.from_name("pii-ner-models", create_if_missing=True)


def _ignore_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".venv" in parts or "__pycache__" in parts:
        return True
    if "out" in parts:
        return True
    return path.suffix in {".pyc", ".pyo"}

def _base_image():
    image = modal.Image.debian_slim().apt_install("git").pip_install("uv")
    requirements_path = LOCAL_PROJECT_DIR / "requirements.txt"
    if requirements_path.exists():
        image = (
            image.add_local_file(
                str(requirements_path),
                "/tmp/requirements.txt",
                copy=True,
            ).run_commands(
                "uv pip install --system --no-cache-dir -r /tmp/requirements.txt"
            )
        )
    return image


image = _base_image().add_local_dir(
    str(LOCAL_PROJECT_DIR),
    str(REMOTE_PROJECT_DIR),
    copy=True,
    ignore=_ignore_path,
)


def _resolve_remote_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REMOTE_PROJECT_DIR_PATH / p).resolve()


def _format_cmd(cmd: Iterable[str]) -> List[str]:
    return [str(token) for token in cmd]


def _display_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in cmd)


def _run(cmd: Iterable[str], env: Optional[dict] = None):
    formatted = _format_cmd(cmd)
    print(f"[modal] running: {_display_cmd(formatted)}")
    subprocess.run(
        formatted,
        check=True,
        cwd=str(REMOTE_PROJECT_DIR),
        env={**os.environ, **(env or {})},
    )


def _maybe_install_deps(force_reinstall: bool):
    if getattr(_maybe_install_deps, "_done", False) and not force_reinstall:
        return
    _run(["uv", "pip", "install", "--system", "-r", "requirements.txt"])
    _maybe_install_deps._done = True


def modal_task(gpu: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
    return app.function(
        image=image,
        gpu=gpu,
        timeout=timeout,
        volumes={"/vol": MODEL_VOLUME},
    )


@modal_task(gpu="A100")
def train(
    model_name: str = "prajjwal1/bert-mini",
    out_dir: str = "out",
    train_path: str = "data/train.jsonl",
    dev_path: str = "data/dev.jsonl",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_length: int = 256,
    doc_stride: int = 0,
    eval_every: int = 0,
    volume_subdir: Optional[str] = None,
    force_reinstall: bool = False,
):
    """Remote training entrypoint. Invoke via `modal run modal_app.py::train -- --epochs 5`."""
    _maybe_install_deps(force_reinstall)
    cmd = [
        "python",
        "src/train.py",
        "--model_name",
        model_name,
        "--train",
        train_path,
        "--dev",
        dev_path,
        "--out_dir",
        out_dir,
        "--epochs",
        epochs,
        "--batch_size",
        batch_size,
        "--lr",
        lr,
        "--max_length",
        max_length,
        "--doc_stride",
        doc_stride,
    ]
    if eval_every:
        cmd += ["--eval_every", eval_every]
    _run(cmd)

    src = _resolve_remote_path(out_dir)
    persist_name = volume_subdir or src.name
    if persist_name and src.exists():
        dst = Path("/vol") / persist_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {src} to volume at /vol/{persist_name}")
    else:
        print(f"[warning] Skipped volume copy because {src} does not exist.")


@modal_task(gpu=None, timeout=15 * 60)
def download(out_dir: str = "out", archive_name: str = "model.zip"):
    """Streams a zipped model directory back to stdout."""
    src = _resolve_remote_path(out_dir)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist in the container.")
    archive_path = Path("/tmp") / archive_name
    subprocess.run(
        ["zip", "-r", str(archive_path), src.name],
        cwd=str(src.parent),
        check=True,
    )
    with open(archive_path, "rb") as fh:
        shutil.copyfileobj(fh, Path("/proc/self/fd/1").open("wb"))


@modal_task(gpu="A100")
def predict(
    model_dir: str = "out",
    input_path: str = "data/dev.jsonl",
    output_path: str = "out/dev_pred.json",
    max_length: int = 256,
    batch_size: int = 16,
    force_reinstall: bool = False,
):

    """Remote prediction entrypoint."""
    _maybe_install_deps(force_reinstall)
    cmd = [
        "python",
        "src/predict.py",
        "--model_dir",
        model_dir,
        "--input",
        input_path,
        "--output",
        output_path,
        "--max_length",
        max_length,
        "--batch_size",
        batch_size,
    ]
    _run(cmd)


@modal_task(gpu=None, timeout=15 * 60)
def evaluate(
    gold_path: str = "data/dev.jsonl",
    pred_path: str = "out/dev_pred.json",
    force_reinstall: bool = False,
):
    """Remote evaluation entrypoint."""
    _maybe_install_deps(force_reinstall)
    cmd = [
        "python",
        "src/eval_span_f1.py",
        "--gold",
        gold_path,
        "--pred",
        pred_path,
    ]
    _run(cmd)


@modal_task(gpu="A100")
def latency(
    model_dir: str = "out",
    input_path: str = "data/dev.jsonl",
    runs: int = 50,
    device: str = "cuda",
    max_length = 256,
    force_reinstall: bool = False,
):
    """Measure inference latency on Modal hardware."""
    _maybe_install_deps(force_reinstall)
    cmd = [
        "python",
        "src/measure_latency.py",
        "--model_dir",
        model_dir,
        "--input",
        input_path,
        "--runs",
        runs,
        "--device",
        device,
        "--max_length",
        max_length
    ]
    _run(cmd)


@app.local_entrypoint()
def main(task: str = "train"):
    """Helper so `python modal_app.py --task train` mirrors local invocations."""
    if task == "train":
        train.remote()
    elif task == "predict":
        predict.remote()
    elif task == "evaluate":
        evaluate.remote()
    elif task == "latency":
        latency.remote()
    else:
        raise ValueError(f"Unknown task: {task}")

