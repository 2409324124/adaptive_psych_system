function StartupSurface({ className, children, onSkip, ...props }) {
  return (
    <section
      className={`boot-screen ${className}`}
      onPointerDown={onSkip}
      {...props}
    >
      {children}
    </section>
  );
}

export function BlackScreen({ variant, onSkip }) {
  return (
    <StartupSurface
      className={`boot-screen--black boot-screen--${variant}`}
      onSkip={onSkip}
      aria-hidden="true"
    />
  );
}

export function PostScreen({ onSkip }) {
  return (
    <StartupSurface
      className="boot-screen--post"
      onSkip={onSkip}
      role="status"
      aria-label="Shinonome 95 POST"
      aria-live="polite"
    >
      <div className="boot-post">
        <strong>Shinonome Personal Computer</strong>
        <span>CPU .............. OK</span>
        <span>Memory ........... OK</span>
        <span>Display .......... OK</span>
        <span>System Disk ...... OK</span>
        <span className="boot-post-starting">Starting Shinonome 95...</span>
      </div>
    </StartupSurface>
  );
}

export function ShinonomeBootScreen({ onSkip, reducedMotion = false }) {
  return (
    <StartupSurface
      className={`boot-screen--system${reducedMotion ? " is-reduced" : ""}`}
      onSkip={onSkip}
      role="status"
      aria-label="Shinonome 95 startup"
      aria-live="polite"
    >
      <div className="boot-system">
        <strong lang="ja">東雲</strong>
        <h1>Shinonome 95</h1>
        <p>{reducedMotion ? "Starting..." : "Starting Shinonome..."}</p>
        {!reducedMotion && (
          <div className="boot-meter" role="img" aria-label="Starting">
            <span className="boot-meter-blocks" aria-hidden="true">
              <i /><i /><i />
            </span>
          </div>
        )}
      </div>
    </StartupSurface>
  );
}
