import { useRef } from "react";

const TASKBAR_HEIGHT = 44;

export default function Win95Window({
  id,
  title,
  windowState,
  dispatch,
  children,
  menu = true,
  menuItems = ["File", "Edit", "Search", "Help"],
  isActive = true,
}) {
  const windowRef = useRef(null);
  const dragRef = useRef(null);

  function startDrag(event) {
    if (windowState.maximized || event.button !== 0) return;
    const rect = windowRef.current.getBoundingClientRect();
    dragRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    dispatch({ type: "focus-window", id });
  }

  function dragWindow(event) {
    if (!dragRef.current || windowState.maximized) return;
    const rect = windowRef.current.getBoundingClientRect();
    const maxX = Math.max(4, window.innerWidth - rect.width - 4);
    const maxY = Math.max(4, window.innerHeight - TASKBAR_HEIGHT - 30);
    dispatch({
      type: "move-window",
      id,
      x: Math.min(maxX, Math.max(4, event.clientX - dragRef.current.offsetX)),
      y: Math.min(maxY, Math.max(4, event.clientY - dragRef.current.offsetY)),
    });
  }

  function stopDrag(event) {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const position = windowState.hasMoved
    ? { left: `${windowState.x}px`, top: `${windowState.y}px` }
    : undefined;

  return (
    <section
      ref={windowRef}
      role="dialog"
      aria-label={title}
      className={`win-window win-window--${id}${windowState.maximized ? " is-maximized" : ""}${isActive ? "" : " is-inactive"}`}
      style={{ ...position, zIndex: windowState.z }}
      onPointerDown={() => dispatch({ type: "focus-window", id })}
    >
      <div
        className="win-titlebar"
        onDoubleClick={() => dispatch({ type: "toggle-maximize", id })}
        onPointerDown={startDrag}
        onPointerMove={dragWindow}
        onPointerUp={stopDrag}
        onPointerCancel={stopDrag}
      >
        <span className="win-title-icon" aria-hidden="true">
          ▤
        </span>
        <strong>{title}</strong>
        <div className="win-controls">
          <button
            type="button"
            aria-label={`最小化 ${title}`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => dispatch({ type: "minimize-window", id })}
          >
            <span aria-hidden="true">_</span>
          </button>
          <button
            type="button"
            aria-label={`${windowState.maximized ? "还原" : "最大化"} ${title}`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => dispatch({ type: "toggle-maximize", id })}
          >
            <span aria-hidden="true">{windowState.maximized ? "◫" : "□"}</span>
          </button>
          <button
            type="button"
            aria-label={`关闭 ${title}`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => dispatch({ type: "close-window", id })}
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>
      </div>
      {menu && (
        <nav className="win-menubar" aria-label={`${title} 菜单`}>
          {menuItems.map((item) => <button type="button" key={item}>{item}</button>)}
        </nav>
      )}
      <div className="win-content">{children}</div>
    </section>
  );
}
