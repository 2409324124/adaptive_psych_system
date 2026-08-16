import { useEffect, useReducer, useState } from "react";

import BenchmarkExplorer from "./BenchmarkExplorer";
import BootGate from "./BootGate";
import EvidenceNotepad from "./EvidenceNotepad";
import { desktopReducer, initialDesktopState } from "./desktopState";
import PixelIcon from "./PixelIcon";
import { PowerOff, ShutdownDialog } from "./PowerScreens";
import SystemProfiler from "./SystemProfiler";
import Win95Window from "./Win95Window";
import benchmarkData from "./data/benchmarks.json";
import { systemProfile } from "./data/systemProfile";

const shortcuts = [
  { id: "about", label: "About Me", lines: ["About Me"], windowId: "welcome" },
  { id: "portfolio", label: "Portfolio", lines: ["Portfolio"], url: "https://portfolio.shinonome.xyz" },
  { id: "benchmark", label: "Benchmark", lines: ["Benchmark"], windowId: "benchmark" },
  { id: "xeon", label: "Xeon Max 9470C", lines: ["Xeon Max", "9470C"], windowId: "system-profiler" },
  { id: "cat", label: "CAT-Psych", lines: ["CAT-Psych"], url: "/cat-psych" },
  { id: "preploop", label: "PrepLoop", lines: ["PrepLoop"], url: "https://preploop.shinonome.xyz" },
  { id: "github", label: "GitHub", lines: ["GitHub"], url: "https://github.com/2409324124" },
  {
    id: "visualization",
    label: "Programming Visualization",
    lines: ["Programming", "Visualization"],
    url: "https://2409324124.github.io/programming-visualization/examples/",
  },
  {
    id: "latex",
    label: "Qwen3 LaTeX",
    lines: ["Qwen3 LaTeX"],
    url: "https://github.com/2409324124/Qwen-Desktop-Assistant-Classic",
  },
  { id: "contact", label: "Contact", lines: ["Contact"], url: "mailto:jzhou2409324124@gmail.com" },
];

function formatClock(date) {
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function StartMenu({ onOpenShortcut, onClose, onRestart, onShutdown }) {
  const items = [
    { label: "Programs", suffix: "▶" },
    { label: "Documents", suffix: "▶" },
    { label: "Benchmarks", shortcut: shortcuts.find(({ id }) => id === "benchmark") },
    { separator: true },
    { label: "GitHub", shortcut: shortcuts.find(({ id }) => id === "github") },
    { label: "Contact", shortcut: shortcuts.find(({ id }) => id === "contact") },
    { separator: true },
    { label: "Restart...", action: onRestart },
    { label: "Shut Down...", action: onShutdown },
  ];

  return (
    <div className="start-menu" role="menu" aria-label="Start menu">
      <div className="start-menu-rail" aria-hidden="true">
        <strong>Shinonome</strong><span>95</span>
      </div>
      <div className="start-menu-items">
        {items.map((item, index) =>
          item.separator ? (
            <div className="start-separator" role="separator" key={`separator-${index}`} />
          ) : (
            <button
              role="menuitem"
              type="button"
              key={item.label}
              onClick={() => {
                if (item.shortcut) onOpenShortcut(item.shortcut);
                if (item.action) item.action();
                onClose();
              }}
            >
              <span className="start-menu-icon" aria-hidden="true">▣</span>
              <span>{item.label}</span>
              <span aria-hidden="true">{item.suffix}</span>
            </button>
          ),
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [state, dispatch] = useReducer(desktopReducer, initialDesktopState);
  const [clock, setClock] = useState(() => new Date());
  const [powerState, setPowerState] = useState("on");
  const evidenceItem = benchmarkData.items.find(
    ({ id }) => id === state.windows.evidence.documentId,
  );

  useEffect(() => {
    const timer = window.setInterval(() => setClock(new Date()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  function openShortcut(shortcut) {
    if (shortcut.windowId) {
      dispatch({ type: "open-window", id: shortcut.windowId });
      return;
    }
    window.open(shortcut.url, shortcut.url.startsWith("mailto:") ? "_self" : "_blank", "noopener,noreferrer");
  }

  function handleShortcutClick(shortcut) {
    dispatch({ type: "select-shortcut", id: shortcut.id });
    if (window.matchMedia?.("(pointer: coarse)").matches) {
      openShortcut(shortcut);
    }
  }

  return (
    <BootGate>
    {({ desktopReady, welcomeReady, restartSystem }) => (
    <>
    {powerState === "on" && desktopReady && <main className="desktop">
      <nav className="desktop-shortcuts" aria-label="桌面快捷方式">
        {shortcuts.map((shortcut) => (
          <button
            key={shortcut.id}
            type="button"
            aria-label={`打开 ${shortcut.label}`}
            aria-pressed={state.selectedShortcut === shortcut.id}
            className="desktop-shortcut"
            onClick={() => handleShortcutClick(shortcut)}
            onDoubleClick={() => openShortcut(shortcut)}
            onKeyDown={(event) => {
              if (event.key === "Enter") openShortcut(shortcut);
            }}
          >
            <span className="desktop-icon-wrap" aria-hidden="true">
              <PixelIcon name={shortcut.id} />
            </span>
            <span className="desktop-label">
              {shortcut.lines.map((line) => <span key={line}>{line}</span>)}
            </span>
          </button>
        ))}
      </nav>
      {welcomeReady && state.windows.welcome.open && !state.windows.welcome.minimized && (
        <Win95Window
          id="welcome"
          title="WELCOME.TXT"
          windowState={state.windows.welcome}
          dispatch={dispatch}
          isActive={state.activeWindow === "welcome"}
        >
          <div className="welcome-copy">
            <h1>东云 / Shinonome</h1>
            <p>
              让想法变成现实，<br />
              让想象力夺权。
            </p>
            <p>Agent · 深度学习 · 心理学</p>
          </div>
        </Win95Window>
      )}
      {state.windows.benchmark.open && !state.windows.benchmark.minimized && (
        <Win95Window
          id="benchmark"
          title="Benchmark Explorer"
          windowState={state.windows.benchmark}
          dispatch={dispatch}
          isActive={state.activeWindow === "benchmark"}
          menuItems={["File", "Edit", "View", "Help"]}
        >
          <BenchmarkExplorer onOpenItem={(item) => {
            if (item.action.type === "external") {
              window.open(item.action.target, "_blank", "noopener,noreferrer");
            } else if (item.action.type === "notepad") {
              dispatch({ type: "open-document", documentId: item.id });
            }
          }} />
        </Win95Window>
      )}
      {state.windows["system-profiler"].open && !state.windows["system-profiler"].minimized && (
        <Win95Window
          id="system-profiler"
          title="Shinonome System Profiler"
          windowState={state.windows["system-profiler"]}
          dispatch={dispatch}
          isActive={state.activeWindow === "system-profiler"}
          menuItems={["File", "Edit", "View", "Help"]}
        >
          <SystemProfiler
            onClose={() => dispatch({ type: "close-window", id: "system-profiler" })}
            onOpenBenchmarkExplorer={() => dispatch({
              type: "open-window",
              id: systemProfile.links.benchmarkExplorer,
            })}
            onOpenBenchmarkReport={() => window.open(
              systemProfile.links.benchmarkReport,
              "_blank",
              "noopener,noreferrer",
            )}
          />
        </Win95Window>
      )}
      {state.windows.evidence.open && !state.windows.evidence.minimized && evidenceItem && (
        <Win95Window
          id="evidence"
          title={evidenceItem.fileName}
          windowState={state.windows.evidence}
          dispatch={dispatch}
          isActive={state.activeWindow === "evidence"}
        >
          <EvidenceNotepad item={evidenceItem} />
        </Win95Window>
      )}
      {state.startOpen && (
        <StartMenu
          onOpenShortcut={openShortcut}
          onClose={() => dispatch({ type: "close-start" })}
          onRestart={restartSystem}
          onShutdown={() => setPowerState("shutdown-dialog")}
        />
      )}
      <footer className="taskbar" aria-label="任务栏">
        <button
          className={`start-button${state.startOpen ? " is-pressed" : ""}`}
          type="button"
          aria-label="Start"
          aria-expanded={state.startOpen}
          onClick={() => dispatch({ type: "toggle-start" })}
        >
          <span className="start-logo" aria-hidden="true">▰</span>
          <strong>Start</strong>
        </button>
        <div className="taskbar-divider" aria-hidden="true" />
        <div className="taskbar-windows">
          {Object.entries(state.windows)
            .filter(([, windowState]) => windowState.open)
            .map(([id, windowState]) => {
              const title = id === "welcome"
                ? "WELCOME.TXT"
                : id === "benchmark"
                  ? "Benchmark Explorer"
                  : id === "system-profiler"
                    ? "System Profiler"
                    : evidenceItem?.fileName ?? "EVIDENCE.TXT";
              return (
                <button
                  className={state.activeWindow === id && !windowState.minimized ? "is-active" : ""}
                  key={id}
                  type="button"
                  aria-label={`切换 ${title}`}
                  onClick={() => dispatch({ type: "toggle-taskbar", id })}
                >
                  <span className="task-icon" aria-hidden="true">▤</span>
                  {title}
                </button>
              );
            })}
        </div>
        <div className="system-tray">
          <span className="speaker-icon" aria-hidden="true">◖))</span>
          <time dateTime={clock.toISOString()}>{formatClock(clock)}</time>
        </div>
      </footer>
    </main>}
    {powerState === "shutdown-dialog" && (
      <ShutdownDialog onCancel={() => setPowerState("on")} onConfirm={() => setPowerState("off")} />
    )}
    {powerState === "off" && <PowerOff onPowerOn={() => {
      setPowerState("on");
      restartSystem();
    }} />}
    </>
    )}
    </BootGate>
  );
}
