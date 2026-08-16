from pathlib import Path
import json
import re
import unittest
from html.parser import HTMLParser


ROOT = Path(__file__).parents[1]
HOME_HTML = ROOT / "web" / "home.html"
HOME_ASSETS = ROOT / "web" / "home-assets"
ROBOTS = ROOT / "web" / "robots.txt"
SITEMAP = ROOT / "web" / "sitemap.xml"
APP_SOURCE = ROOT / "homepage" / "src" / "App.jsx"
STYLES = ROOT / "homepage" / "src" / "styles.css"


class _HomeDocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1_count = 0
        self.semantic_tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1":
            self.h1_count += 1
        if tag in {"main", "nav", "section", "article", "footer"}:
            self.semantic_tags.add(tag)


class HomeDesktopTest(unittest.TestCase):
    def test_homepage_build_publishes_crawler_files_for_real_routes_only(self) -> None:
        robots = ROBOTS.read_text(encoding="utf-8")
        sitemap = SITEMAP.read_text(encoding="utf-8")

        self.assertEqual(
            robots,
            "User-agent: *\nAllow: /\nSitemap: https://shinonome.xyz/sitemap.xml\n",
        )
        self.assertIn("<loc>https://shinonome.xyz/</loc>", sitemap)
        for invented_route in ("/about", "/benchmark", "/contact", "/portfolio"):
            self.assertNotIn(f"shinonome.xyz{invented_route}", sitemap)

    def test_production_shell_contains_indexable_identity_and_projects(self) -> None:
        html = HOME_HTML.read_text(encoding="utf-8")
        parser = _HomeDocumentParser()
        parser.feed(html)

        self.assertIn(
            "<title>Xu Jianzhou / 东云 (Shinonome) — AI Agent, Machine Learning &amp; Systems Research</title>",
            html,
        )
        self.assertIn('<link rel="canonical" href="https://shinonome.xyz/" />', html)
        self.assertIn('property="og:image"', html)
        self.assertIn('name="twitter:card" content="summary_large_image"', html)
        self.assertIn('type="application/ld+json"', html)
        self.assertEqual(parser.h1_count, 1)
        self.assertEqual(
            parser.semantic_tags,
            {"main", "nav", "section", "article", "footer"},
        )
        for copy in (
            "Xu Jianzhou",
            "东云",
            "Shinonome",
            "DeepSeek Codex Adapter",
            "CAT-Psych",
            "PrepLoop",
            "Programming Visualization",
            "Xeon Max 9470C Benchmark",
        ):
            self.assertIn(copy, html)
        self.assertIn(
            'href="https://2409324124.github.io/xeon-max-9470c-benchmarks/report/"',
            html,
        )
        self.assertNotIn(
            'href="https://2409324124.github.io/xeon-max-9470c-benchmarks/"',
            html,
        )

        description = re.search(
            r'<meta name="description" content="([^"]+)"', html
        ).group(1)
        self.assertGreaterEqual(len(description), 120)
        self.assertLessEqual(len(description), 160)
        json_ld = re.search(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.S
        ).group(1)
        graph = json.loads(json_ld)["@graph"]
        self.assertEqual(
            {entry["@type"] for entry in graph},
            {"WebSite", "ProfilePage", "Person"},
        )

    def test_production_shell_loads_the_local_react_desktop(self) -> None:
        html = HOME_HTML.read_text(encoding="utf-8")

        self.assertIn('<div id="root">', html)
        self.assertRegex(
            html,
            re.compile(r'/static/home-assets/homepage\.css\?v=[0-9a-f]{12}'),
        )
        self.assertRegex(
            html,
            re.compile(r'/static/home-assets/homepage\.js\?v=[0-9a-f]{12}'),
        )
        self.assertLess(
            html.index("/static/home-assets/homepage.css"),
            html.index("</head>"),
        )
        self.assertNotIn("home-particles.js", html)
        self.assertTrue((HOME_ASSETS / "homepage.js").exists())
        self.assertTrue((HOME_ASSETS / "homepage.css").exists())

    def test_homepage_is_desktop_navigation_not_a_landing_page(self) -> None:
        source = APP_SOURCE.read_text(encoding="utf-8")
        shortcuts = [
            "About Me",
            "Portfolio",
            "Benchmark",
            "Xeon Max 9470C",
            "CAT-Psych",
            "PrepLoop",
            "GitHub",
            "Programming Visualization",
            "Qwen3 LaTeX",
            "Contact",
        ]

        for shortcut in shortcuts:
            self.assertIn(shortcut, source)
        self.assertIn("WELCOME.TXT", source)
        self.assertIn("东云 / Shinonome", source)
        self.assertRegex(source, re.compile(r"让想法变成现实，.*让想象力夺权。", re.S))
        self.assertIn("Agent · 深度学习 · 心理学", source)
        self.assertNotIn("Explore my work", source)
        self.assertNotIn("SELECTED ENGINEERING WORK", source)

    def test_win95_theme_and_mobile_accessibility_are_explicit(self) -> None:
        css = STYLES.read_text(encoding="utf-8")

        for token in (
            "--win-face",
            "--win-highlight",
            "--win-shadow",
            "--win-dark-shadow",
            "--win-title",
            "--desktop-teal",
        ):
            self.assertIn(token, css)
        self.assertIn("prefers-reduced-motion: reduce", css)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn("overflow: hidden", css)
        self.assertIn(".js .seo-fallback", css)
        self.assertIn(".seo-fallback", css)


if __name__ == "__main__":
    unittest.main()
