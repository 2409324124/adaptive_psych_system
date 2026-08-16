export default function EvidenceNotepad({ item }) {
  return (
    <article className="evidence-note">
      <pre>{item.summary.join("\n")}</pre>
      <div className="evidence-note-source">
        <a href={item.sourceUrl} target="_blank" rel="noreferrer">Source</a>
        <span>Read-only evidence file</span>
      </div>
    </article>
  );
}
