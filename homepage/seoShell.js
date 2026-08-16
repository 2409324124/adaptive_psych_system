const site = {
  title: "Xu Jianzhou / 东云 (Shinonome) — AI Agent, Machine Learning & Systems Research",
  description:
    "Xu Jianzhou (东云 / Shinonome) builds and studies AI agents, machine learning systems, runtime tools, hardware benchmarks, and model evaluation.",
  canonical: "https://shinonome.xyz/",
  ogImage: "https://shinonome.xyz/static/home-assets/og/shinonome-desktop.png",
  email: "jzhou2409324124@gmail.com",
  github: "https://github.com/2409324124",
  portfolio: "https://portfolio.shinonome.xyz/",
  orcid: "https://orcid.org/0009-0008-6625-0148",
};

const projects = [
  {
    name: "CAT-Psych",
    description: "Chinese-first adaptive Big Five personality assessment using MIRT / CAT.",
    url: "https://shinonome.xyz/cat-psych",
  },
  {
    name: "Programming Visualization",
    description: "Interactive algorithm execution, variable state, and call-stack visualization.",
    url: "https://2409324124.github.io/programming-visualization/examples/",
  },
  {
    name: "PrepLoop",
    description: "Daily spaced review for algorithms, agent harnesses, and system design.",
    url: "https://preploop.shinonome.xyz/",
  },
  {
    name: "Xeon Max 9470C Benchmark",
    description: "CPU inference, HBM, AMX, NUMA, OpenVINO, and low-bit quantization evidence.",
    url: "https://2409324124.github.io/xeon-max-9470c-benchmarks/report/",
  },
  {
    name: "DeepSeek Codex Adapter",
    description: "OpenAI Responses-compatible adapter for DeepSeek tool-calling workflows.",
    url: "https://github.com/2409324124/deepseek-codex-adapter",
  },
];

const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": `${site.canonical}#website`,
      url: site.canonical,
      name: "Shinonome",
      alternateName: ["东云", "Xu Jianzhou"],
      inLanguage: ["zh-CN", "en"],
    },
    {
      "@type": "ProfilePage",
      "@id": `${site.canonical}#profile-page`,
      url: site.canonical,
      name: site.title,
      description: site.description,
      isPartOf: { "@id": `${site.canonical}#website` },
      mainEntity: { "@id": `${site.canonical}#person` },
    },
    {
      "@type": "Person",
      "@id": `${site.canonical}#person`,
      name: "Xu Jianzhou",
      alternateName: ["东云", "Shinonome"],
      url: site.canonical,
      email: `mailto:${site.email}`,
      sameAs: [site.github, site.portfolio, site.orcid],
      knowsAbout: [
        "AI Agents",
        "Machine Learning",
        "Runtime Systems",
        "Hardware Benchmarking",
        "LLM Evaluation",
      ],
    },
  ],
};

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function renderSeoHead() {
  const jsonLd = JSON.stringify(structuredData).replaceAll("<", "\\u003c");
  return `
    <meta name="description" content="${escapeHtml(site.description)}" />
    <meta name="robots" content="index,follow,max-image-preview:large" />
    <link rel="canonical" href="${site.canonical}" />
    <meta property="og:type" content="profile" />
    <meta property="og:site_name" content="Shinonome" />
    <meta property="og:title" content="${escapeHtml(site.title)}" />
    <meta property="og:description" content="${escapeHtml(site.description)}" />
    <meta property="og:image" content="${site.ogImage}" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:url" content="${site.canonical}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${escapeHtml(site.title)}" />
    <meta name="twitter:description" content="${escapeHtml(site.description)}" />
    <meta name="twitter:image" content="${site.ogImage}" />
    <title>${escapeHtml(site.title)}</title>
    <script type="application/ld+json">${jsonLd}</script>`;
}

export function renderSeoFallback() {
  const projectArticles = projects
    .map(
      ({ name, description, url }) => `
          <article>
            <h3><a href="${url}">${escapeHtml(name)}</a></h3>
            <p>${escapeHtml(description)}</p>
          </article>`,
    )
    .join("");

  return `<div class="seo-fallback">
      <header>
        <p>Shinonome Personal Research Workstation</p>
        <nav aria-label="Primary navigation">
          <a href="${site.portfolio}">Projects</a>
          <a href="https://2409324124.github.io/xeon-max-9470c-benchmarks/report/">Benchmark</a>
          <a href="#about">About</a>
          <a href="mailto:${site.email}">Contact</a>
        </nav>
      </header>
      <main>
        <section id="about" aria-labelledby="seo-home-title">
          <h1 id="seo-home-title">Xu Jianzhou / 东云 / Shinonome</h1>
          <p>AI Agent · Machine Learning · Systems Research</p>
          <p>让想法变成现实，让想象力夺权。</p>
        </section>
        <section aria-labelledby="seo-projects-title">
          <h2 id="seo-projects-title">Projects</h2>${projectArticles}
        </section>
      </main>
      <footer>
        <a href="${site.github}">GitHub</a>
        <a href="${site.orcid}">ORCID</a>
        <a href="mailto:${site.email}">Email</a>
      </footer>
    </div>`;
}

export function injectSeoShell(template) {
  return template
    .replace("    <!--seo-head-->", renderSeoHead())
    .replace("<!--seo-fallback-->", renderSeoFallback());
}

export function renderRobots() {
  return "User-agent: *\nAllow: /\nSitemap: https://shinonome.xyz/sitemap.xml\n";
}

export function renderSitemap() {
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${site.canonical}</loc>
  </url>
</urlset>
`;
}

export { projects, site, structuredData };
