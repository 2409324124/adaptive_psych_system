import { describe, expect, it } from "vitest";

import { projects, renderSeoFallback, site } from "./seoShell";

describe("SEO fallback", () => {
  it("exposes the BBS as a real navigation and project link", () => {
    const html = renderSeoFallback();

    expect(site.bbs).toBe("https://bbs.shinonome.xyz/");
    expect(projects).toContainEqual(expect.objectContaining({
      name: "东云通信局",
      url: site.bbs,
    }));
    expect(html).toContain(`<a href="${site.bbs}">BBS</a>`);
    expect(html).toContain(`<a href="${site.bbs}">东云通信局</a>`);
  });
});
