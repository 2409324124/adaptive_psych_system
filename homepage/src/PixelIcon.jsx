const P = {
  black: "#101010",
  dark: "#404040",
  shadow: "#808080",
  face: "#c0c0c0",
  light: "#dfdfdf",
  white: "#ffffff",
  blue: "#000080",
  cyan: "#00a8c0",
  teal: "#008080",
  yellow: "#d8c45a",
  yellowDark: "#887526",
  purple: "#800080",
  pink: "#e2a5a5",
  green: "#00a040",
  red: "#c02020",
};

function AboutIcon() {
  return (
    <>
      <rect x="5" y="2" width="22" height="18" fill={P.dark} />
      <rect x="4" y="3" width="22" height="17" fill={P.white} />
      <rect x="6" y="4" width="18" height="13" fill={P.face} />
      <rect x="8" y="6" width="14" height="9" fill={P.black} />
      <rect x="9" y="7" width="12" height="7" fill={P.blue} />
      <rect x="12" y="8" width="6" height="5" fill={P.cyan} />
      <rect x="14" y="7" width="3" height="1" fill={P.white} />
      <rect x="11" y="10" width="9" height="1" fill={P.white} />
      <rect x="9" y="20" width="16" height="3" fill={P.shadow} />
      <rect x="7" y="23" width="20" height="5" fill={P.face} />
      <rect x="6" y="24" width="20" height="4" fill={P.white} />
      <rect x="8" y="25" width="16" height="2" fill={P.dark} />
      <rect x="9" y="25" width="2" height="1" fill={P.red} />
      <rect x="13" y="25" width="2" height="1" fill={P.green} />
      <rect x="18" y="25" width="5" height="1" fill={P.black} />
      <rect x="5" y="28" width="22" height="2" fill={P.black} />
    </>
  );
}

function PortfolioIcon() {
  return (
    <>
      <rect x="10" y="4" width="12" height="4" fill={P.black} />
      <rect x="11" y="3" width="10" height="4" fill={P.yellow} />
      <rect x="4" y="8" width="24" height="19" fill={P.black} />
      <rect x="3" y="9" width="24" height="16" fill={P.yellow} />
      <rect x="5" y="10" width="20" height="2" fill="#f4e58b" />
      <rect x="5" y="13" width="20" height="11" fill="#b7a33e" />
      <rect x="6" y="14" width="18" height="9" fill={P.yellow} />
      <rect x="14" y="8" width="4" height="5" fill={P.black} />
      <rect x="15" y="9" width="2" height="3" fill={P.white} />
      <rect x="3" y="25" width="24" height="2" fill={P.yellowDark} />
      <rect x="7" y="7" width="2" height="20" fill={P.yellowDark} />
      <rect x="22" y="7" width="2" height="20" fill={P.yellowDark} />
      <rect x="4" y="27" width="22" height="2" fill={P.black} />
    </>
  );
}

function BenchmarkIcon() {
  return (
    <>
      <rect x="3" y="6" width="13" height="4" fill={P.black} />
      <rect x="4" y="5" width="10" height="4" fill="#f1df74" />
      <rect x="2" y="9" width="27" height="19" fill={P.black} />
      <rect x="3" y="8" width="25" height="18" fill={P.yellow} />
      <rect x="5" y="11" width="21" height="14" fill="#f2df72" />
      <rect x="7" y="21" width="3" height="3" fill={P.blue} />
      <rect x="11" y="17" width="3" height="7" fill="#273ad4" />
      <rect x="15" y="13" width="3" height="11" fill={P.blue} />
      <rect x="19" y="9" width="3" height="15" fill="#15208b" />
      <rect x="6" y="23" width="18" height="1" fill={P.black} />
      <path d="M7 19L12 14L16 16L21 10" fill="none" stroke={P.red} strokeWidth="1" />
      <rect x="3" y="26" width="25" height="2" fill={P.yellowDark} />
    </>
  );
}

function XeonIcon() {
  return (
    <>
      {[5, 9, 13, 17, 21, 25].map((x) => <rect key={`t${x}`} x={x} y="1" width="2" height="4" fill={P.light} />)}
      {[5, 9, 13, 17, 21, 25].map((x) => <rect key={`b${x}`} x={x} y="27" width="2" height="4" fill={P.light} />)}
      {[5, 9, 13, 17, 21, 25].map((y) => <rect key={`l${y}`} x="1" y={y} width="4" height="2" fill={P.light} />)}
      {[5, 9, 13, 17, 21, 25].map((y) => <rect key={`r${y}`} x="27" y={y} width="4" height="2" fill={P.light} />)}
      <rect x="4" y="4" width="24" height="24" fill={P.black} />
      <rect x="6" y="6" width="20" height="20" fill={P.dark} />
      <rect x="8" y="8" width="16" height="16" fill="#282828" />
      <rect x="9" y="9" width="14" height="1" fill={P.shadow} />
      <text x="16" y="15" textAnchor="middle" fill={P.white} fontSize="5" fontFamily="monospace">x86</text>
      <text x="16" y="21" textAnchor="middle" fill={P.white} fontSize="5" fontFamily="monospace">AI</text>
    </>
  );
}

function CatPsychIcon() {
  return (
    <>
      <path d="M8 4H20V6H24V9H27V18H24V21H20V29H10V21H7V18H5V9H8Z" fill={P.black} />
      <path d="M9 5H19V7H23V10H25V17H22V20H18V27H11V20H8V17H6V10H9Z" fill={P.white} />
      <rect x="10" y="8" width="11" height="8" fill={P.pink} />
      <rect x="11" y="7" width="7" height="2" fill={P.purple} />
      <rect x="9" y="10" width="3" height="4" fill={P.purple} />
      <rect x="13" y="9" width="2" height="6" fill={P.purple} />
      <rect x="16" y="8" width="2" height="6" fill={P.purple} />
      <rect x="19" y="10" width="3" height="4" fill={P.purple} />
      <rect x="11" y="15" width="9" height="1" fill={P.dark} />
    </>
  );
}

function PrepLoopIcon() {
  return (
    <>
      <path d="M6 8H19V4L27 11L19 18V14H10V19H5V11Z" fill={P.black} />
      <path d="M7 7H18V3L25 10L18 16V12H9V18H4V10Z" fill={P.cyan} />
      <path d="M26 23H13V28L5 21L13 14V18H22V13H27V21Z" fill={P.black} />
      <path d="M25 22H14V27L7 21L14 15V19H23V14H28V22Z" fill="#b8f1f1" />
      <rect x="7" y="8" width="2" height="8" fill={P.white} />
      <rect x="23" y="16" width="2" height="7" fill={P.shadow} />
    </>
  );
}

function GitHubIcon() {
  return (
    <>
      <path d="M8 7L11 9H21L24 7L25 12C28 14 29 18 27 22C25 25 22 27 18 27H14C10 27 7 25 5 22C3 18 4 14 7 12Z" fill={P.black} />
      <rect x="8" y="13" width="16" height="10" fill={P.black} />
      <rect x="10" y="15" width="12" height="7" fill={P.pink} />
      <rect x="12" y="17" width="2" height="2" fill={P.black} />
      <rect x="18" y="17" width="2" height="2" fill={P.black} />
      <rect x="15" y="20" width="2" height="1" fill={P.black} />
      <rect x="4" y="19" width="5" height="1" fill={P.white} />
      <rect x="23" y="19" width="5" height="1" fill={P.white} />
      <rect x="13" y="26" width="2" height="5" fill={P.black} />
      <rect x="18" y="26" width="2" height="5" fill={P.black} />
    </>
  );
}

function VisualizationIcon() {
  return (
    <>
      <rect x="4" y="3" width="24" height="20" fill={P.black} />
      <rect x="3" y="4" width="24" height="18" fill={P.white} />
      <rect x="6" y="6" width="18" height="13" fill="#181840" />
      <rect x="7" y="16" width="2" height="2" fill={P.cyan} />
      <rect x="9" y="13" width="2" height="2" fill={P.purple} />
      <rect x="11" y="14" width="2" height="2" fill={P.cyan} />
      <rect x="13" y="10" width="2" height="2" fill={P.purple} />
      <rect x="15" y="11" width="2" height="2" fill={P.cyan} />
      <rect x="17" y="7" width="2" height="2" fill={P.purple} />
      <rect x="19" y="9" width="2" height="2" fill={P.cyan} />
      <rect x="21" y="8" width="2" height="2" fill={P.red} />
      <rect x="13" y="22" width="5" height="4" fill={P.shadow} />
      <rect x="8" y="26" width="16" height="3" fill={P.black} />
      <rect x="7" y="25" width="16" height="3" fill={P.face} />
      <rect x="8" y="25" width="14" height="1" fill={P.white} />
    </>
  );
}

function LatexIcon() {
  return (
    <>
      <path d="M7 2H22L27 7V30H7Z" fill={P.black} />
      <path d="M6 1H21L26 6V29H6Z" fill={P.white} />
      <path d="M21 1V7H26" fill={P.face} />
      <path d="M10 8H20L15 15L21 22H9V19H16L11 14L16 10H10Z" fill={P.blue} />
      <text x="16" y="27" textAnchor="middle" fill={P.black} fontSize="5" fontFamily="serif">LaTeX</text>
    </>
  );
}

function ContactIcon() {
  return (
    <>
      <rect x="3" y="8" width="27" height="18" fill={P.black} />
      <rect x="2" y="7" width="27" height="18" fill={P.white} />
      <path d="M3 8L15 18L28 8" fill={P.face} stroke={P.dark} strokeWidth="1" />
      <path d="M3 24L11 16M28 24L20 16" fill="none" stroke={P.shadow} strokeWidth="1" />
      <rect x="25" y="21" width="3" height="3" fill={P.blue} />
      <rect x="3" y="8" width="1" height="16" fill={P.light} />
    </>
  );
}

function BBSIcon() {
  return (
    <>
      <rect x="4" y="3" width="24" height="20" fill={P.black} />
      <rect x="3" y="2" width="24" height="20" fill={P.light} />
      <rect x="5" y="4" width="20" height="16" fill={P.dark} />
      <rect x="7" y="6" width="16" height="11" fill="#061d2b" />
      <rect x="9" y="8" width="5" height="1" fill={P.cyan} />
      <rect x="9" y="11" width="9" height="1" fill={P.cyan} />
      <rect x="9" y="14" width="3" height="1" fill={P.cyan} />
      <rect x="14" y="14" width="6" height="1" fill={P.white} />
      <rect x="13" y="22" width="5" height="3" fill={P.shadow} />
      <rect x="8" y="25" width="15" height="3" fill={P.black} />
      <rect x="7" y="24" width="15" height="3" fill={P.face} />
      <rect x="24" y="6" width="5" height="2" fill={P.cyan} />
      <rect x="26" y="10" width="4" height="2" fill={P.cyan} />
      <rect x="27" y="14" width="3" height="2" fill={P.cyan} />
      <rect x="24" y="18" width="5" height="2" fill={P.cyan} />
    </>
  );
}

const ICONS = {
  about: AboutIcon,
  portfolio: PortfolioIcon,
  benchmark: BenchmarkIcon,
  xeon: XeonIcon,
  cat: CatPsychIcon,
  preploop: PrepLoopIcon,
  github: GitHubIcon,
  visualization: VisualizationIcon,
  latex: LatexIcon,
  contact: ContactIcon,
  bbs: BBSIcon,
};

export default function PixelIcon({ name, className = "" }) {
  const Icon = ICONS[name] ?? AboutIcon;
  return (
    <svg
      className={`pixel-icon ${className}`}
      viewBox="0 0 32 32"
      role="presentation"
      aria-hidden="true"
      shapeRendering="crispEdges"
    >
      <Icon />
    </svg>
  );
}
