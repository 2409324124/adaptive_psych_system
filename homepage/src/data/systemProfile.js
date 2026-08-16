import benchmarkData from "./benchmarks.json";

const reportItem = benchmarkData.items.find(({ id }) => id === "xeon-9470c");

export const systemProfile = {
  processor: {
    name: "Intel Xeon CPU Max 9470C",
    cores: 52,
    threads: 104,
    instructions: [
      "MMX",
      "SSE",
      "SSE2",
      "SSE3",
      "SSSE3",
      "SSE4.1",
      "SSE4.2",
      "AVX",
      "AVX2",
      "FMA3",
      "AVX-512F",
      "AVX-512BW",
      "AVX-512VL",
      "AVX-512 VNNI",
      "AVX-512 BF16",
      "AVX-512 FP16",
      "AMX-TILE",
      "AMX-INT8",
      "AMX-BF16",
    ],
  },
  memory: {
    type: "HBM",
    capacity: "64 GiB",
    configuration: "HBM-only",
    relatedBenchmark: "HBM / STREAM",
  },
  topology: {
    numaMode: "SNC4",
    numaNodes: 4,
  },
  cache: {
    status: "Not loaded.",
  },
  system: {
    os: "Ubuntu 24.04",
  },
  thumbnail: {
    kind: "placeholder",
    label: "Xeon Max 9470C CPU package",
  },
  links: {
    benchmarkExplorer: "benchmark",
    benchmarkReport: reportItem?.action.target ?? "",
  },
};
