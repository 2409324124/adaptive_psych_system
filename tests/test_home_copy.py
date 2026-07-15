from pathlib import Path
import re
import unittest


HOME_HTML = Path(__file__).parents[1] / "web" / "home.html"


class HomeCopyTest(unittest.TestCase):
    def test_hero_distills_the_homepage_into_one_statement(self) -> None:
        html = HOME_HTML.read_text(encoding="utf-8")
        visible_copy = re.sub(r"<[^>]+>", " ", html)
        visible_copy = re.sub(r"\s+", " ", visible_copy)

        self.assertIn("Agent · 深度学习 · 心理学", visible_copy)
        self.assertIn("让想法变成现实，让想象力夺权。", visible_copy)
        self.assertNotIn(
            "把想法写成可以运行、可以验证、可以长期使用的东西。",
            visible_copy,
        )
        self.assertNotIn(
            "这里是入口，完整项目、研究记录与在线 Demo 收录在独立作品集。",
            visible_copy,
        )
        self.assertNotIn("Current signals", visible_copy)
        self.assertNotIn("Provider、router、tool schema", visible_copy)


if __name__ == "__main__":
    unittest.main()
