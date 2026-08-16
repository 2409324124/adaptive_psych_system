export const initialDesktopState = {
  selectedShortcut: null,
  activeWindow: "welcome",
  nextZ: 3,
  startOpen: false,
  windows: {
    welcome: {
      open: true,
      minimized: false,
      maximized: false,
      z: 2,
      x: 0,
      y: 0,
      hasMoved: false,
    },
    "system-profiler": {
      open: false,
      minimized: false,
      maximized: false,
      z: 1,
      x: 0,
      y: 0,
      hasMoved: false,
    },
    benchmark: {
      open: false,
      minimized: false,
      maximized: false,
      z: 1,
      x: 0,
      y: 0,
      hasMoved: false,
    },
    evidence: {
      open: false,
      minimized: false,
      maximized: false,
      z: 1,
      x: 0,
      y: 0,
      hasMoved: false,
      documentId: null,
    },
  },
};

function activateWindow(state, id, overrides = {}) {
  return {
    ...state,
    activeWindow: id,
    nextZ: state.nextZ + 1,
    startOpen: false,
    windows: {
      ...state.windows,
      [id]: {
        ...state.windows[id],
        open: true,
        minimized: false,
        z: state.nextZ,
        ...overrides,
      },
    },
  };
}

export function desktopReducer(state, action) {
  switch (action.type) {
    case "select-shortcut":
      return { ...state, selectedShortcut: action.id, startOpen: false };
    case "open-window":
      return activateWindow(state, action.id);
    case "open-document":
      return activateWindow(state, "evidence", { documentId: action.documentId });
    case "focus-window":
      return activateWindow(state, action.id);
    case "toggle-taskbar": {
      const windowState = state.windows[action.id];
      if (!windowState.open || windowState.minimized) {
        return activateWindow(state, action.id);
      }
      if (state.activeWindow === action.id) {
        return {
          ...state,
          activeWindow: null,
          windows: {
            ...state.windows,
            [action.id]: { ...windowState, minimized: true },
          },
        };
      }
      return activateWindow(state, action.id);
    }
    case "minimize-window":
      return {
        ...state,
        activeWindow: state.activeWindow === action.id ? null : state.activeWindow,
        windows: {
          ...state.windows,
          [action.id]: { ...state.windows[action.id], minimized: true },
        },
      };
    case "toggle-maximize":
      return activateWindow(state, action.id, {
        maximized: !state.windows[action.id].maximized,
      });
    case "close-window":
      return {
        ...state,
        activeWindow: state.activeWindow === action.id ? null : state.activeWindow,
        windows: {
          ...state.windows,
          [action.id]: {
            ...state.windows[action.id],
            open: false,
            minimized: false,
          },
        },
      };
    case "move-window":
      return {
        ...state,
        windows: {
          ...state.windows,
          [action.id]: {
            ...state.windows[action.id],
            x: action.x,
            y: action.y,
            hasMoved: true,
          },
        },
      };
    case "toggle-start":
      return { ...state, startOpen: !state.startOpen, selectedShortcut: null };
    case "close-start":
      return { ...state, startOpen: false };
    default:
      return state;
  }
}
