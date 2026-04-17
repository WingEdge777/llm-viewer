from llm_viewer.cli import main
from llm_viewer.server import _pick_available_port, _static_dir, create_app


def _route(app, path: str):
    for route in app.router.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"route not found: {path}")


def test_index_route_returns_static_html():
    app = create_app()
    route = _route(app, "/")

    response = route.endpoint()

    assert str(response.path).endswith("index.html")


def test_static_assets_exist():
    static_dir = _static_dir()

    assert (static_dir / "index.html").exists()
    assert (static_dir / "app.css").exists()
    assert (static_dir / "app.js").exists()

    html = (static_dir / "index.html").read_text(encoding="utf-8")
    css = (static_dir / "app.css").read_text(encoding="utf-8")

    assert 'id="sidebar" class="sidebar hidden"' in html
    assert "background-color: #ececec;" in css


def test_frontend_has_profile_reload_and_block_navigation():
    script = (_static_dir() / "app.js").read_text(encoding="utf-8")
    css = (_static_dir() / "app.css").read_text(encoding="utf-8")

    assert "reloadGraphForProfile" in script
    assert 'card.addEventListener("dblclick"' in script
    assert "edge.tensor_name || edge.shape" in script
    assert 'graphSurface.addEventListener("mousedown"' in script
    assert 'graphSurface.addEventListener("wheel"' in script
    assert "function fitView()" in script
    assert "function zoomAtCenter(nextZoom)" in script
    assert "function renderModelSidebar()" in script
    assert 'sidebar.classList.toggle("hidden", !loaded);' in script
    assert "function edgePortMaps(graph)" in script
    assert "const BASE_NODE_WIDTH = 220;" in script
    assert "function updateResponsiveScale()" in script
    assert "width: calc(220px * var(--ui-scale));" in css


def test_api_graph_endpoint_returns_bundle():
    app = create_app()
    route = _route(app, "/api/graph")

    response = route.endpoint(
        route.dependant.body_params[0].type_(
            profile="prefill",
            config={
                "model_type": "qwen3",
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 9728,
                "head_dim": 128,
                "vocab_size": 151936,
            },
        )
    )

    assert response["metadata"]["model_type"] == "qwen3"
    assert [graph["id"] for graph in response["graphs"]] == ["model", "block"]


def test_cli_no_args_starts_app(monkeypatch):
    captured = {}

    def fake_run_app(host: str, port: int, open_browser: bool) -> int:
        captured["host"] = host
        captured["port"] = port
        captured["open_browser"] = open_browser
        return 0

    monkeypatch.setattr("llm_viewer.cli.run_app", fake_run_app)

    result = main([])

    assert result == 0
    assert captured == {"host": "127.0.0.1", "port": 8000, "open_browser": True}


def test_pick_available_port_skips_busy_port():
    class FakeSocket:
        attempts = []

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, *_args):
            return None

        def bind(self, address):
            self.attempts.append(address[1])
            if address[1] == 8000:
                raise OSError("busy")
            return None

    import llm_viewer.server as server

    original = server.socket.socket
    server.socket.socket = FakeSocket
    try:
        picked = _pick_available_port("127.0.0.1", 8000, attempts=4)
    finally:
        server.socket.socket = original

    assert picked == 8001
    assert FakeSocket.attempts[:2] == [8000, 8001]
