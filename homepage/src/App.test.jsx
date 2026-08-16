import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const BOOT_KEY = "shinonome95.boot.seen";

describe("Shinonome desktop", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(BOOT_KEY, "1");
    window.history.replaceState({}, "", "/");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("boots Shinonome 95 through hard-cut system phases on first visit", () => {
    vi.useFakeTimers();
    window.localStorage.removeItem(BOOT_KEY);

    render(<App />);
    expect(screen.queryByText("Shinonome Personal Computer")).toBeNull();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
    expect(window.localStorage.getItem(BOOT_KEY)).toBeNull();

    act(() => vi.advanceTimersByTime(100));
    expect(screen.getByText("Shinonome Personal Computer")).toBeTruthy();
    expect(screen.getByText("CPU .............. OK")).toBeTruthy();

    act(() => vi.advanceTimersByTime(600));
    expect(screen.getByText("Shinonome 95")).toBeTruthy();
    expect(screen.getByText("Starting Shinonome...")).toBeTruthy();

    act(() => vi.advanceTimersByTime(1_700));
    expect(screen.queryByText("Shinonome 95")).toBeNull();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();

    act(() => vi.advanceTimersByTime(120));
    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();
    expect(window.localStorage.getItem(BOOT_KEY)).toBe("1");

    act(() => vi.advanceTimersByTime(180));

    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
  });

  it("opens the desktop immediately after the Shinonome 95 boot has been seen", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
    expect(screen.queryByRole("status", { name: /Shinonome 95/ })).toBeNull();
  });

  it("opens with the welcome note as the secondary introduction", () => {
    render(<App />);

    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
    expect(screen.getByText("东云 / Shinonome")).toBeTruthy();
    expect(screen.getByText(/让想法变成现实/)).toBeTruthy();
    expect(screen.getByText("Agent · 深度学习 · 心理学")).toBeTruthy();
  });

  it("uses the ten desktop shortcuts as the homepage navigation", () => {
    render(<App />);

    const labels = [
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
    ];

    expect(screen.getByRole("navigation", { name: "桌面快捷方式" })).toBeTruthy();
    expect(screen.getAllByRole("button", { name: /打开/ })).toHaveLength(10);
    labels.forEach((label) => {
      expect(screen.getByRole("button", { name: `打开 ${label}` })).toBeTruthy();
    });
  });

  it("opens the Xeon shortcut in Shinonome System Profiler", async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    render(<App />);

    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    expect(screen.getByRole("dialog", { name: "Shinonome System Profiler" })).toBeTruthy();
    expect(openSpy).not.toHaveBeenCalled();
  });

  it("shows a dense workstation CPU profile with compact utility actions", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    expect(screen.getByRole("tab", { name: "CPU", selected: true })).toBeTruthy();
    expect(screen.getAllByText("Intel Xeon CPU Max 9470C")).toHaveLength(1);
    expect(screen.queryByText("Specification")).toBeNull();
    expect(screen.getByText("Instructions")).toBeTruthy();
    expect(screen.getByText(
      "MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA3, AVX-512F, AVX-512BW, AVX-512VL, AVX-512 VNNI, AVX-512 BF16, AVX-512 FP16, AMX-TILE, AMX-INT8, AMX-BF16",
    )).toBeTruthy();
    expect(screen.getByText("52")).toBeTruthy();
    expect(screen.getByText("104")).toBeTruthy();
    expect(screen.getByText("64 GiB HBM")).toBeTruthy();
    expect(screen.getByText("SNC4")).toBeTruthy();
    expect(screen.getByText("NUMA Nodes")).toBeTruthy();
    expect(screen.getByText("HBM-only")).toBeTruthy();
    expect(screen.getByText("Ubuntu 24.04")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Report..." })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Benchmarks..." })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Close" })).toBeTruthy();
  });

  it("uses the simplified CPU package as the only profiler preview", async () => {
    window.history.replaceState({}, "", "/?cpuPreview=full");
    const user = userEvent.setup();
    render(<App />);

    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    expect(screen.getByRole("img", { name: "Simplified CPU package preview" })).toBeTruthy();
    expect(screen.queryByRole("img", { name: "Full CPU package preview" })).toBeNull();
    expect(screen.queryByRole("img", { name: "Xeon Max identification badge" })).toBeNull();
  });

  it("switches directly between the Memory, Topology, Cache, and Links pages", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    await user.click(screen.getByRole("tab", { name: "Memory" }));
    expect(screen.getByText("HBM-only")).toBeTruthy();
    expect(screen.getByText("HBM / STREAM")).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Topology" }));
    expect(screen.getByText("NUMA Nodes")).toBeTruthy();
    expect(screen.getByText("4")).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Cache" }));
    expect(screen.getByText("Not loaded.")).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Links" }));
    expect(screen.getByText("Xeon Max 9470C Report")).toBeTruthy();
    expect(screen.getByText("Published benchmark report.")).toBeTruthy();
  });

  it("opens or focuses the existing Benchmark Explorer from the profiler", async () => {
    const user = userEvent.setup();
    render(<App />);
    const shortcut = screen.getByRole("button", { name: "打开 Xeon Max 9470C" });

    await user.dblClick(shortcut);
    await user.dblClick(shortcut);
    expect(screen.getAllByRole("dialog", { name: "Shinonome System Profiler" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "切换 System Profiler" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Benchmarks..." }));
    await user.click(screen.getByRole("button", { name: "Benchmarks..." }));

    expect(screen.getAllByRole("dialog", { name: "Benchmark Explorer" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "切换 Benchmark Explorer" })).toBeTruthy();
  });

  it("opens the published Xeon report from the profiler", async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    render(<App />);
    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    await user.click(screen.getByRole("button", { name: "Report..." }));

    expect(openSpy).toHaveBeenCalledWith(
      "https://2409324124.github.io/xeon-max-9470c-benchmarks/report/",
      "_blank",
      "noopener,noreferrer",
    );
  });

  it("closes the profiler from its compact utility footer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    await user.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.queryByRole("dialog", { name: "Shinonome System Profiler" })).toBeNull();
    expect(screen.queryByRole("button", { name: "切换 System Profiler" })).toBeNull();
  });

  it("controls the profiler through its existing window chrome and taskbar item", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.dblClick(screen.getByRole("button", { name: "打开 Xeon Max 9470C" }));

    await user.click(screen.getByRole("button", { name: "最大化 Shinonome System Profiler" }));
    expect(screen.getByRole("button", { name: "还原 Shinonome System Profiler" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "最小化 Shinonome System Profiler" }));
    expect(screen.queryByRole("dialog", { name: "Shinonome System Profiler" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "切换 System Profiler" }));
    expect(screen.getByRole("dialog", { name: "Shinonome System Profiler" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "关闭 Shinonome System Profiler" }));
    expect(screen.queryByRole("dialog", { name: "Shinonome System Profiler" })).toBeNull();
    expect(screen.queryByRole("button", { name: "切换 System Profiler" })).toBeNull();
  });

  it("opens Benchmark Explorer and filters evidence through its folder tree", async () => {
    const user = userEvent.setup();
    render(<App />);
    const shortcut = screen.getByRole("button", { name: "打开 Benchmark" });

    await user.click(shortcut);
    expect(shortcut.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("dialog", { name: "BENCHMARK" })).toBeNull();

    await user.dblClick(shortcut);
    expect(screen.getByRole("dialog", { name: "Benchmark Explorer" })).toBeTruthy();
    expect(screen.getByText("C:\\SHINONOME\\BENCHMARKS")).toBeTruthy();
    expect(screen.getByText("Xeon Max 9470C")).toBeTruthy();
    expect(screen.getByText("HBM / STREAM")).toBeTruthy();
    expect(screen.getByText("oneDNN AMX Kernels")).toBeTruthy();
    expect(screen.getByText("OpenVINO Qwen3-8B INT4")).toBeTruthy();
    expect(screen.getByText("Balanced INT4 Fidelity")).toBeTruthy();
    expect(screen.getByText("5 objects")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "浏览 Agent" }));
    expect(screen.getByText("C:\\SHINONOME\\BENCHMARKS\\AGENT")).toBeTruthy();
    expect(screen.getByText("0 objects")).toBeTruthy();
    expect(screen.queryByText("Xeon Max 9470C")).toBeNull();
  });

  it("opens measurement files in one reusable Notepad window", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.dblClick(screen.getByRole("button", { name: "打开 Benchmark" }));

    await user.dblClick(screen.getByRole("button", { name: "选择 OpenVINO Qwen3-8B INT4" }));
    expect(screen.getByRole("dialog", { name: "QWEN3_INT4.TXT" })).toBeTruthy();
    expect(screen.getByText(/203\.40 aggregate tok\/s/)).toBeTruthy();
    expect(screen.getByText(/1,320 measured requests/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Source" })).toBeTruthy();

    await user.dblClick(screen.getByRole("button", { name: "选择 Balanced INT4 Fidelity" }));
    expect(screen.queryByRole("dialog", { name: "QWEN3_INT4.TXT" })).toBeNull();
    expect(screen.getByRole("dialog", { name: "BALANCED_INT4.TXT" })).toBeTruthy();
    expect(screen.getByText(/Top-1 agreement\s+88\.038%/)).toBeTruthy();
    expect(screen.queryByText(/failed|rejected/i)).toBeNull();
  });

  it("minimizes and restores a window from its taskbar button", async () => {
    const user = userEvent.setup();
    render(<App />);
    const task = screen.getByRole("button", { name: "切换 WELCOME.TXT" });

    await user.click(task);
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();

    await user.click(task);
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
  });

  it("supports minimize, maximize, restore, and close controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "最大化 WELCOME.TXT" }));
    expect(screen.getByRole("button", { name: "还原 WELCOME.TXT" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "最小化 WELCOME.TXT" }));
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "切换 WELCOME.TXT" }));

    await user.click(screen.getByRole("button", { name: "关闭 WELCOME.TXT" }));
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();
    expect(screen.queryByRole("button", { name: "切换 WELCOME.TXT" })).toBeNull();
  });

  it("opens the classic Start menu with project and contact entries", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(screen.getByRole("menu", { name: "Start menu" })).toBeTruthy();
    expect(screen.getByText("Shinonome")).toBeTruthy();
    expect(screen.getByText("95")).toBeTruthy();
    expect(screen.queryByText("Windows")).toBeNull();
    expect(screen.getByRole("menuitem", { name: "Programs" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Benchmarks" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "GitHub" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Contact" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Restart..." })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Shut Down..." })).toBeTruthy();
  });

  it("restarts through the full Boot sequence without clearing the seen marker", () => {
    vi.useFakeTimers();
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Restart..." }));
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
    expect(window.localStorage.getItem(BOOT_KEY)).toBe("1");

    act(() => vi.advanceTimersByTime(100));
    expect(screen.getByText("Shinonome Personal Computer")).toBeTruthy();

    act(() => vi.advanceTimersByTime(2_420));
    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();

    act(() => vi.advanceTimersByTime(180));
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
    expect(window.localStorage.getItem(BOOT_KEY)).toBe("1");
  });

  it("skips an active Boot immediately with Escape and leaves no blocking overlay", () => {
    vi.useFakeTimers();
    window.localStorage.removeItem(BOOT_KEY);
    render(<App />);

    act(() => vi.advanceTimersByTime(700));
    expect(screen.getByText("Shinonome 95")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
    expect(screen.queryByRole("status", { name: /Shinonome 95/ })).toBeNull();
    expect(window.localStorage.getItem(BOOT_KEY)).toBe("1");

    act(() => vi.advanceTimersByTime(3_000));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    expect(screen.getByRole("menu", { name: "Start menu" })).toBeTruthy();
  });

  it.each(["Enter", " "])("also skips Boot with the %s key", (key) => {
    window.localStorage.removeItem(BOOT_KEY);
    render(<App />);

    fireEvent.keyDown(window, { key });

    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.queryByRole("status", { name: /Shinonome 95/ })).toBeNull();
  });

  it("skips Boot when its full-screen surface is tapped", () => {
    vi.useFakeTimers();
    window.localStorage.removeItem(BOOT_KEY);
    render(<App />);
    act(() => vi.advanceTimersByTime(100));

    fireEvent.pointerDown(screen.getByRole("status", { name: "Shinonome 95 POST" }));

    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
  });

  it("cancels the Boot timeline when the application unmounts", () => {
    vi.useFakeTimers();
    window.localStorage.removeItem(BOOT_KEY);
    const { unmount } = render(<App />);

    unmount();
    act(() => vi.advanceTimersByTime(3_000));

    expect(window.localStorage.getItem(BOOT_KEY)).toBeNull();
  });

  it("uses a safe power-off screen and can boot the desktop again", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start" }));
    await user.click(screen.getByRole("menuitem", { name: "Shut Down..." }));
    expect(screen.getByRole("dialog", { name: "Shut Down Shinonome 95" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Shut down" }));
    expect(screen.getByRole("status", { name: "Computer is off" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Power On" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Power On" }));
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
    expect(screen.queryByRole("status", { name: "Computer is off" })).toBeNull();
  });

  it("honors reduced motion during startup", () => {
    vi.useFakeTimers();
    window.localStorage.removeItem(BOOT_KEY);
    vi.stubGlobal("matchMedia", vi.fn((query) => ({
      matches: query === "(prefers-reduced-motion: reduce)",
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })));

    render(<App />);
    expect(screen.getByText("Starting...")).toBeTruthy();
    expect(screen.queryByText("Shinonome Personal Computer")).toBeNull();

    act(() => vi.advanceTimersByTime(280));
    expect(screen.queryByText("Starting...")).toBeNull();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();

    act(() => vi.advanceTimersByTime(100));
    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
    expect(screen.queryByRole("dialog", { name: "WELCOME.TXT" })).toBeNull();

    act(() => vi.advanceTimersByTime(120));
    expect(screen.getByRole("dialog", { name: "WELCOME.TXT" })).toBeTruthy();
  });
});
