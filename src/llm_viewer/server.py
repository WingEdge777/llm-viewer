from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
from importlib import resources
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from llm_viewer.profiles import ProfileName, get_profile
from llm_viewer.registry import build_graph_bundle

STATIC_DIR = resources.files("llm_viewer").joinpath("static")


class GraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: ProfileName = ProfileName.PREFILL
    config: dict[str, Any]


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Viewer", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=_static_dir()), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_static_dir() / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/graph")
    def graph(payload: GraphRequest) -> dict[str, Any]:
        try:
            profile = get_profile(payload.profile)
            bundle = build_graph_bundle(config=payload.config, profile=profile)
            return bundle.to_dict()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def run_app(host: str, port: int, open_browser: bool = True) -> int:
    port = _pick_available_port(host, port)
    url = f"http://{host}:{port}/"
    print(f"llm_viewer listening on {url}", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: _open_browser(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0


def _static_dir() -> Path:
    return Path(str(STATIC_DIR))


def _open_browser(url: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # type: ignore[attr-defined]
            return

        candidates: list[list[str]] = []
        if sys.platform == "darwin":
            candidates.append(["open", url])
        else:
            candidates.extend(
                [
                    ["xdg-open", url],
                    ["gio", "open", url],
                ]
            )

        for command in candidates:
            if shutil.which(command[0]) is None:
                continue
            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return


def _pick_available_port(host: str, preferred_port: int, attempts: int = 32) -> int:
    for candidate in range(preferred_port, preferred_port + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((host, candidate))
                    return candidate
                except OSError:
                    continue
        except PermissionError:
            return preferred_port
    raise RuntimeError(
        f"No available port found in range {preferred_port}-{preferred_port + attempts - 1} for host {host}."
    )
