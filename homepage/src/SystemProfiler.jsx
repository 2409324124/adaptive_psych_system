import { useState } from "react";

import { systemProfile } from "./data/systemProfile";

const TABS = ["CPU", "Memory", "Topology", "Cache", "Links"];

function CpuPackagePreview() {
  return (
    <div className="cpu-preview-simple" role="img" aria-label="Simplified CPU package preview">
      <div className="cpu-preview-simple-plate">
        <strong>XEON</strong>
        <span>MAX</span>
        <small>9470C</small>
      </div>
    </div>
  );
}

function CpuPreview() {
  return (
    <div className="cpu-preview-pane">
      <CpuPackagePreview />
      <span className="cpu-preview-caption">Package</span>
    </div>
  );
}

function ProfileFields({ rows }) {
  return (
    <dl className="profile-fields">
      {rows.map(([label, value, multiline = false]) => (
        <div className={multiline ? "is-multiline" : ""} key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function CompactCpuFields({ rows }) {
  return (
    <dl className="cpu-facts">
      {rows.map(([label, value, wide = false]) => (
        <div className={wide ? "is-wide" : ""} key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function CpuPanel() {
  const { processor, memory, topology, system } = systemProfile;
  return (
    <fieldset className="profile-group">
      <legend>Processor</legend>
      <div className="cpu-profile-layout">
        <ProfileFields rows={[
          ["Name", processor.name],
          ["Instructions", processor.instructions.join(", "), true],
        ]} />
        <div className="cpu-profile-body">
          <CompactCpuFields rows={[
            ["Cores", processor.cores],
            ["Threads", processor.threads],
            ["NUMA Mode", topology.numaMode],
            ["NUMA Nodes", topology.numaNodes],
            ["Memory", `${memory.capacity} ${memory.type}`],
            ["Memory Mode", memory.configuration],
            ["OS", system.os, true],
          ]} />
          <CpuPreview />
        </div>
      </div>
    </fieldset>
  );
}

function MemoryPanel() {
  const { memory } = systemProfile;
  return (
    <fieldset className="profile-group">
      <legend>Memory</legend>
      <ProfileFields rows={[
        ["Memory Type", memory.type],
        ["Capacity", memory.capacity],
        ["Configuration", memory.configuration],
        ["Related benchmark", memory.relatedBenchmark],
      ]} />
    </fieldset>
  );
}

function TopologyPanel() {
  const { processor, topology } = systemProfile;
  return (
    <fieldset className="profile-group">
      <legend>CPU / NUMA Topology</legend>
      <ProfileFields rows={[
        ["Cores", processor.cores],
        ["Threads", processor.threads],
        ["NUMA Mode", topology.numaMode],
        ["NUMA Nodes", topology.numaNodes],
      ]} />
    </fieldset>
  );
}

function CachePanel() {
  return (
    <fieldset className="profile-group">
      <legend>Cache</legend>
      <ProfileFields rows={[["Detailed cache information", systemProfile.cache.status]]} />
    </fieldset>
  );
}

function LinksPanel({ onOpenBenchmarkExplorer, onOpenBenchmarkReport }) {
  return (
    <fieldset className="profile-group profile-links">
      <legend>Links</legend>
      <div>
        <strong>Benchmark Explorer</strong>
        <p>Browse published benchmark entries.</p>
        <button
          type="button"
          className="profile-button"
          aria-label="Open Benchmark Explorer link"
          onClick={onOpenBenchmarkExplorer}
        >
          Open...
        </button>
      </div>
      <div>
        <strong>Xeon Max 9470C Report</strong>
        <p>Published benchmark report.</p>
        <button
          type="button"
          className="profile-button"
          aria-label="Open Xeon Max 9470C Report link"
          onClick={onOpenBenchmarkReport}
        >
          Open...
        </button>
      </div>
    </fieldset>
  );
}

export default function SystemProfiler({
  onClose,
  onOpenBenchmarkExplorer,
  onOpenBenchmarkReport,
}) {
  const [activeTab, setActiveTab] = useState("CPU");

  function handleTabKeyDown(event) {
    const currentIndex = TABS.indexOf(activeTab);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % TABS.length;
    if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + TABS.length) % TABS.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = TABS.length - 1;
    if (nextIndex === currentIndex) return;
    event.preventDefault();
    setActiveTab(TABS[nextIndex]);
  }

  return (
    <div className="system-profiler">
      <div className="profile-tabs" role="tablist" aria-label="System profile sections">
        {TABS.map((tab) => (
          <button
            id={`profile-tab-${tab.toLowerCase()}`}
            className={activeTab === tab ? "is-active" : ""}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls="system-profile-panel"
            tabIndex={activeTab === tab ? 0 : -1}
            key={tab}
            onClick={() => setActiveTab(tab)}
            onKeyDown={handleTabKeyDown}
          >
            {tab}
          </button>
        ))}
      </div>
      <div
        id="system-profile-panel"
        className="profile-panel"
        role="tabpanel"
        aria-labelledby={`profile-tab-${activeTab.toLowerCase()}`}
      >
        {activeTab === "CPU" && <CpuPanel />}
        {activeTab === "Memory" && <MemoryPanel />}
        {activeTab === "Topology" && <TopologyPanel />}
        {activeTab === "Cache" && <CachePanel />}
        {activeTab === "Links" && (
          <LinksPanel
            onOpenBenchmarkExplorer={onOpenBenchmarkExplorer}
            onOpenBenchmarkReport={onOpenBenchmarkReport}
          />
        )}
      </div>
      <div className="profile-actions">
        <button type="button" className="profile-button" onClick={onOpenBenchmarkReport}>
          Report...
        </button>
        <button type="button" className="profile-button" onClick={onOpenBenchmarkExplorer}>
          Benchmarks...
        </button>
        <button type="button" className="profile-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
