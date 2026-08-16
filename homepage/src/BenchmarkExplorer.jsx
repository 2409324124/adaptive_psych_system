import { useMemo, useState } from "react";

import benchmarkData from "./data/benchmarks.json";

const BASE_PATH = "C:\\SHINONOME\\BENCHMARKS";

function descendantsOf(categoryId) {
  const ids = new Set([categoryId]);
  let changed = true;
  while (changed) {
    changed = false;
    benchmarkData.categories.forEach((category) => {
      if (category.parentId && ids.has(category.parentId) && !ids.has(category.id)) {
        ids.add(category.id);
        changed = true;
      }
    });
  }
  return ids;
}

function pathFor(categoryId) {
  const segments = [];
  let current = benchmarkData.categories.find(({ id }) => id === categoryId);
  while (current && current.id !== "root") {
    segments.unshift(current.pathSegment);
    current = benchmarkData.categories.find(({ id }) => id === current.parentId);
  }
  return segments.length ? `${BASE_PATH}\\${segments.join("\\")}` : BASE_PATH;
}

function TreeBranch({ parentId, currentId, onNavigate, depth = 0 }) {
  const children = benchmarkData.categories.filter((category) => category.parentId === parentId);
  return children.map((category) => (
    <div className="explorer-tree-branch" key={category.id}>
      <button
        className={currentId === category.id ? "is-current" : ""}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
        type="button"
        aria-label={`浏览 ${category.label}`}
        onClick={() => onNavigate(category.id)}
      >
        <span className="tree-twist" aria-hidden="true">−</span>
        <span className="tree-folder-icon" aria-hidden="true" />
        <span>{category.label}</span>
      </button>
      <TreeBranch
        parentId={category.id}
        currentId={currentId}
        onNavigate={onNavigate}
        depth={depth + 1}
      />
    </div>
  ));
}

export default function BenchmarkExplorer({ onOpenItem }) {
  const [categoryId, setCategoryId] = useState("root");
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [foldersVisible, setFoldersVisible] = useState(
    () => !window.matchMedia?.("(max-width: 760px)").matches,
  );

  const visibleItems = useMemo(() => {
    const categoryIds = descendantsOf(categoryId);
    return benchmarkData.items.filter((item) => categoryIds.has(item.categoryId));
  }, [categoryId]);

  function navigate(nextId) {
    if (nextId === categoryId) return;
    setHistory((entries) => [...entries, categoryId]);
    setCategoryId(nextId);
    setSelectedId(null);
  }

  function goBack() {
    const previous = history.at(-1);
    if (!previous) return;
    setHistory((entries) => entries.slice(0, -1));
    setCategoryId(previous);
    setSelectedId(null);
  }

  function goUp() {
    const category = benchmarkData.categories.find(({ id }) => id === categoryId);
    if (category?.parentId) navigate(category.parentId);
  }

  return (
    <div className="benchmark-explorer">
      <div className="explorer-toolbar" aria-label="Explorer toolbar">
        <button type="button" onClick={goBack} disabled={history.length === 0} aria-label="Back">←</button>
        <button type="button" onClick={goUp} disabled={categoryId === "root"} aria-label="Up">↑</button>
        <button type="button" className="folders-toggle" onClick={() => setFoldersVisible((value) => !value)}>
          Folders
        </button>
      </div>
      <div className="explorer-address">
        <strong>Address:</strong>
        <span>{pathFor(categoryId)}</span>
        <button type="button" aria-label="Go">▼</button>
      </div>
      <div className={`explorer-workspace${foldersVisible ? " has-folders" : ""}`}>
        <nav className="explorer-tree" aria-label="Benchmark folders">
          <button
            type="button"
            aria-label="浏览 Benchmarks"
            className={categoryId === "root" ? "is-current" : ""}
            onClick={() => navigate("root")}
          >
            <span className="tree-twist" aria-hidden="true">−</span>
            <span className="tree-computer-icon" aria-hidden="true" />
            <span>Benchmarks</span>
          </button>
          <TreeBranch parentId="root" currentId={categoryId} onNavigate={navigate} />
        </nav>
        <div className="explorer-details">
          <table>
            <thead>
              <tr><th>Name</th><th>Type</th><th>Status</th></tr>
            </thead>
            <tbody>
              {visibleItems.map((item) => (
                <tr className={selectedId === item.id ? "is-selected" : ""} key={item.id}>
                  <td>
                    <button
                      type="button"
                      aria-label={`选择 ${item.name}`}
                      onClick={() => setSelectedId(item.id)}
                      onDoubleClick={() => onOpenItem(item)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") onOpenItem(item);
                      }}
                    >
                      <span aria-hidden="true">{item.action.type === "external" ? "▣" : "▤"}</span> {item.name}
                    </button>
                  </td>
                  <td>{item.type}</td>
                  <td>{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {visibleItems.length === 0 && <p className="explorer-empty">This folder is empty.</p>}
        </div>
      </div>
      <div className="explorer-statusbar">
        <span>{visibleItems.length} {visibleItems.length === 1 ? "object" : "objects"}</span>
        <span>{selectedId ? benchmarkData.items.find(({ id }) => id === selectedId)?.type : "Select an item"}</span>
        {selectedId && (
          <button
            type="button"
            onClick={() => onOpenItem(benchmarkData.items.find(({ id }) => id === selectedId))}
          >
            Open
          </button>
        )}
      </div>
    </div>
  );
}
