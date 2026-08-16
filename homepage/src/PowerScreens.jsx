export function ShutdownDialog({ onCancel, onConfirm }) {
  return (
    <div className="power-overlay">
      <section className="shutdown-dialog" role="dialog" aria-label="Shut Down Shinonome 95">
        <div className="shutdown-titlebar">Shut Down Shinonome 95</div>
        <div className="shutdown-body">
          <span className="shutdown-computer" aria-hidden="true">▣</span>
          <p>Are you sure you want to shut down your computer?</p>
        </div>
        <div className="shutdown-actions">
          <button type="button" onClick={onConfirm}>Shut down</button>
          <button type="button" onClick={onCancel}>Cancel</button>
        </div>
      </section>
    </div>
  );
}

export function PowerOff({ onPowerOn }) {
  return (
    <section className="power-off" role="status" aria-label="Computer is off">
      <p>It is now safe to turn off your computer.</p>
      <button type="button" onClick={onPowerOn}>Power On</button>
    </section>
  );
}
