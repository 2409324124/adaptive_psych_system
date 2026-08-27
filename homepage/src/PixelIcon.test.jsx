import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PixelIcon from "./PixelIcon";

describe("PixelIcon", () => {
  it("renders the BBS shortcut as two networked computers", () => {
    const { container } = render(<PixelIcon name="bbs" />);
    const icon = container.querySelector("svg");

    expect(icon.querySelectorAll('[fill="#000080"]').length).toBeGreaterThan(0);
    expect(icon.querySelectorAll('[fill="#008080"]').length).toBeGreaterThan(0);
    expect(icon.querySelectorAll('[fill="#d8c45a"]').length).toBeGreaterThan(0);
  });
});
