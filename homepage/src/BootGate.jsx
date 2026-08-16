import { useCallback, useEffect, useRef, useState } from "react";

import { BlackScreen, PostScreen, ShinonomeBootScreen } from "./BootScreen";

export const BOOT_STORAGE_KEY = "shinonome95.boot.seen";

function hasSeenBoot() {
  try {
    return window.localStorage.getItem(BOOT_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function markBootSeen() {
  try {
    window.localStorage.setItem(BOOT_STORAGE_KEY, "1");
  } catch {
    // Storage is optional; the current startup can still complete.
  }
}

export default function BootGate({ children }) {
  const shouldBoot = useRef(!hasSeenBoot()).current;
  const initialPhase = shouldBoot ? "idle" : "desktop";
  const [phase, setPhase] = useState(initialPhase);
  const phaseRef = useRef(initialPhase);
  const [welcomeReady, setWelcomeReady] = useState(!shouldBoot);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [runId, setRunId] = useState(shouldBoot ? 1 : 0);
  const cancelRunRef = useRef(null);

  const setBootPhase = useCallback((nextPhase) => {
    phaseRef.current = nextPhase;
    setPhase(nextPhase);
  }, []);

  const skipBoot = useCallback(() => {
    if (phaseRef.current === "desktop") return;
    cancelRunRef.current?.();
    cancelRunRef.current = null;
    markBootSeen();
    setBootPhase("desktop");
    setWelcomeReady(true);
  }, [setBootPhase]);

  const restartSystem = useCallback(() => {
    cancelRunRef.current?.();
    cancelRunRef.current = null;
    setWelcomeReady(false);
    setBootPhase("idle");
    setRunId((current) => current + 1);
  }, [setBootPhase]);

  useEffect(() => {
    if (runId === 0) return undefined;

    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    setReducedMotion(reducedMotion);
    const timers = [];
    let cancelled = false;

    const removeKeyListener = () => window.removeEventListener("keydown", handleKeyDown);
    const cancelRun = () => {
      if (cancelled) return;
      cancelled = true;
      timers.forEach((timer) => window.clearTimeout(timer));
      removeKeyListener();
    };
    const schedule = (delay, callback) => {
      timers.push(window.setTimeout(() => {
        if (!cancelled) callback();
      }, delay));
    };
    function handleKeyDown(event) {
      if (!["Escape", "Enter", " "].includes(event.key)) return;
      if (event.key === " ") event.preventDefault();
      skipBoot();
    }

    cancelRunRef.current = cancelRun;
    window.addEventListener("keydown", handleKeyDown);

    if (reducedMotion) {
      setBootPhase("boot");
      schedule(280, () => setBootPhase("blackout"));
      schedule(380, () => {
        markBootSeen();
        removeKeyListener();
        setBootPhase("desktop");
      });
      schedule(500, () => {
        setWelcomeReady(true);
        cancelRun();
      });
    } else {
      setBootPhase("idle");
      schedule(100, () => setBootPhase("post"));
      schedule(700, () => setBootPhase("boot"));
      schedule(2_400, () => setBootPhase("blackout"));
      schedule(2_520, () => {
        markBootSeen();
        removeKeyListener();
        setBootPhase("desktop");
      });
      schedule(2_700, () => {
        setWelcomeReady(true);
        cancelRun();
      });
    }

    return cancelRun;
  }, [runId, setBootPhase, skipBoot]);

  return (
    <>
      {children({
        desktopReady: phase === "desktop",
        welcomeReady,
        restartSystem,
      })}
      {phase === "idle" && <BlackScreen variant="idle" onSkip={skipBoot} />}
      {phase === "post" && <PostScreen onSkip={skipBoot} />}
      {phase === "boot" && (
        <ShinonomeBootScreen reducedMotion={reducedMotion} onSkip={skipBoot} />
      )}
      {phase === "blackout" && <BlackScreen variant="blackout" onSkip={skipBoot} />}
    </>
  );
}
