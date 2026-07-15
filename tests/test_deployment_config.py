from pathlib import Path


def test_tencent_compose_preserves_the_production_runtime_contract() -> None:
    compose = (
        Path(__file__).parents[1] / "deploy" / "tencent" / "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert 'command: ["python", "/proxy/reverse_proxy.py"]' in compose
    assert "CAT_PSYCH_SESSION_BACKEND: json" in compose
    assert "CAT_PSYCH_SESSION_DIR: /app/data/sessions" in compose
    assert "./site/home.html:/app/web/home.html:ro" in compose
