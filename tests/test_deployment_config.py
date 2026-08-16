from pathlib import Path


def test_tencent_compose_preserves_the_production_runtime_contract() -> None:
    compose = (
        Path(__file__).parents[1] / "deploy" / "tencent" / "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert 'command: ["python", "/proxy/reverse_proxy.py"]' in compose
    assert "CAT_PSYCH_SESSION_BACKEND: json" in compose
    assert "CAT_PSYCH_SESSION_DIR: /app/data/sessions" in compose
    assert "./site/home.html:/app/web/home.html:ro" in compose
    assert "./site/home-assets:/app/web/home-assets:ro" in compose


def test_homepage_exposes_and_mounts_canonical_crawler_files() -> None:
    root = Path(__file__).parents[1]
    api_source = (root / "api" / "app.py").read_text(encoding="utf-8")
    compose = (root / "deploy" / "tencent" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert '@app.get("/robots.txt")' in api_source
    assert '@app.get("/sitemap.xml")' in api_source
    assert 'media_type="text/plain"' in api_source
    assert 'media_type="application/xml"' in api_source
    assert "./site/robots.txt:/app/web/robots.txt:ro" in compose
    assert "./site/sitemap.xml:/app/web/sitemap.xml:ro" in compose
