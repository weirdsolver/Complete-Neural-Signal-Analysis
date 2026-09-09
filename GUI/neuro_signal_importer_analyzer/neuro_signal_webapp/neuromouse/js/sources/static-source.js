let dataCache = null;

export function createStaticSource() {
  return {
    meta() {
      return dataCache?.meta ?? null;
    },
    async start(onFrame, onStatus) {
      onStatus?.("static");
      onFrame?.(await loadStaticData());
    },
    stop() {},
  };
}

export async function loadStaticData() {
  if (dataCache) return dataCache;

  const failures = [];
  for (const candidate of resolveStaticDataCandidates()) {
    try {
      const response = await fetch(candidate.url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      validateData(data);
      dataCache = data;
      if (candidate.persist) {
        rememberBackendDataset(candidate.url);
      }
      return dataCache;
    } catch (error) {
      failures.push(`${candidate.label}: ${error.message}`);
      if (candidate.clearOnFail) clearRememberedBackendDataset();
    }
  }

  throw new Error(`Failed to load a NeuroMouse data.json. Tried: ${failures.join("; ")}`);
}


function resolveStaticDataCandidates() {
  // backend-dataset mode: bind importer/analyzer generated data into original NeuroMouse.
  // v0.11.9: the original NeuroMouse advanced panels only become visible after
  // the dataset loads and contains advanced-analysis objects. Previous builds
  // could get stuck on a stale localStorage URL from an older job; when that
  // fetch failed, the app never mounted and the panels stayed hidden. This
  // function now tries backend-generated data first, falls back safely, and
  // clears stale remembered URLs.
  const candidates = [];
  const add = (url, label, options = {}) => {
    if (!url) return;
    const text = String(url);
    if (candidates.some((item) => String(item.url) === text)) return;
    candidates.push({ url, label, ...options });
  };

  const params = new URLSearchParams(globalThis.window?.location?.search ?? "");
  const injectedUrl = globalThis.window?.NEURO_SIGNAL_BACKEND_DATASET?.datasetUrl;
  add(injectedUrl, "backend-injected dataset", { persist: true, clearOnFail: false });
  add(params.get("dataset") || params.get("data") || params.get("data_json"), "query dataset", { persist: true, clearOnFail: false });

  try {
    const remembered = globalThis.window?.localStorage?.getItem("NEURO_SIGNAL_LAST_BACKEND_DATASET_URL");
    add(remembered, "remembered backend dataset", { persist: false, clearOnFail: true });
  } catch (_error) {}

  // This endpoint scans the app workspace for the newest generated NeuroMouse
  // data.json. It is what users expect after they click Convert or Analyze in
  // NeuroMouse. If none exists, it will 404 and we continue to the bundled demo.
  add("/api/neuromouse/latest/data.json", "latest backend dataset", { persist: true, clearOnFail: false });

  // Plain /neuromouse/?demo=1 and /neuromouse/ still have an advanced-analysis demo dataset so users can
  // verify that Polar Alpha, Kuramoto, Channel Network, and TDA render even
  // before importing their own file.
  add(new URL("../../data/data.json", import.meta.url), "bundled NeuroMouse demo", { persist: false, clearOnFail: false });
  return candidates;
}

function rememberBackendDataset(url) {
  if (!url) return;
  try {
    const text = String(url);
    if (text.startsWith("/api/") || text.includes("/api/")) {
      globalThis.window?.localStorage?.setItem("NEURO_SIGNAL_LAST_BACKEND_DATASET_URL", text);
    }
  } catch (_error) {}
}

function clearRememberedBackendDataset() {
  try {
    globalThis.window?.localStorage?.removeItem("NEURO_SIGNAL_LAST_BACKEND_DATASET_URL");
    globalThis.window?.localStorage?.removeItem("NEURO_SIGNAL_LAST_BACKEND_JOB_ID");
    globalThis.window?.localStorage?.removeItem("NEURO_SIGNAL_LAST_NEUROMOUSE_URL");
  } catch (_error) {}
}

export function validateData(data, { maxChannels = 4096 } = {}) {
  const isPositiveInteger = (value) => Number.isInteger(value) && value > 0;
  const requireFiniteNumbers = (values, message) => {
    for (const value of values) {
      if (!Number.isFinite(value)) {
        throw new Error(message);
      }
    }
  };
  const requireMatrixRows = (rows, expectedWidth, label, widthLabel) => {
    rows.forEach((row, index) => {
      if (!Array.isArray(row)) {
        throw new Error(`${label} row ${index} must be an array`);
      }
      if (row.length !== expectedWidth) {
        throw new Error(`${label} row ${index} length must equal ${widthLabel}`);
      }
      requireFiniteNumbers(row, `${label} row ${index} must contain only finite numbers`);
    });
  };

  if (!isPositiveInteger(maxChannels)) {
    throw new Error("maxChannels must be a positive integer");
  }

  const channels = data?.meta?.channels;
  if (!Array.isArray(channels) || channels.length === 0) {
    throw new Error("data.json must contain a non-empty meta.channels array");
  }
  const channelCount = channels.length;
  if (channelCount > maxChannels) {
    throw new Error(`meta.channels length must be at most ${maxChannels}`);
  }

  if (Object.hasOwn(data.meta, "n_channels")) {
    const declaredChannelCount = data.meta.n_channels;
    if (!isPositiveInteger(declaredChannelCount)) {
      throw new Error("meta.n_channels must be a positive integer");
    }
    if (declaredChannelCount !== channelCount) {
      throw new Error("meta.n_channels must equal meta.channels length");
    }
  }

  if (!Array.isArray(data?.welch_psd?.frequencies) || !Array.isArray(data?.welch_psd?.psd)) {
    throw new Error("data.json is missing welch_psd arrays");
  }
  if (data.welch_psd.frequencies.length === 0) {
    throw new Error("welch_psd.frequencies must be a non-empty array");
  }
  requireFiniteNumbers(
    data.welch_psd.frequencies,
    "welch_psd.frequencies must contain only finite numbers",
  );
  if (data.welch_psd.psd.length !== channelCount) {
    throw new Error(`welch_psd.psd has ${data.welch_psd.psd.length} channel rows but meta.channels lists ${channelCount}`);
  }
  requireMatrixRows(
    data.welch_psd.psd,
    data.welch_psd.frequencies.length,
    "welch_psd.psd",
    "welch_psd.frequencies length",
  );

  if (!Array.isArray(data?.centroid?.time_relative) || !Array.isArray(data?.centroid?.values)) {
    throw new Error("data.json is missing centroid arrays");
  }
  if (data.centroid.time_relative.length === 0) {
    throw new Error("centroid.time_relative must be a non-empty array");
  }
  if (data.centroid.values.length !== channelCount) {
    throw new Error(`centroid.values has ${data.centroid.values.length} channel rows but meta.channels lists ${channelCount}`);
  }
  requireMatrixRows(
    data.centroid.values,
    data.centroid.time_relative.length,
    "centroid.values",
    "centroid.time_relative length",
  );

  if (!Array.isArray(data?.geometry?.time)) {
    throw new Error("data.json is missing geometry.time");
  }
  if (data.geometry.time.length === 0) {
    throw new Error("geometry.time must be a non-empty array");
  }
  requireFiniteNumbers(
    data.geometry.time,
    "geometry.time must contain only finite numbers",
  );

  if (Object.hasOwn(data, "mea")) {
    if (!data.mea || typeof data.mea !== "object") {
      throw new Error("mea must be an object when present");
    }
    if (typeof data.mea.sampling_rate_hz !== "number" || !Number.isFinite(data.mea.sampling_rate_hz) || data.mea.sampling_rate_hz <= 0) {
      throw new Error("mea.sampling_rate_hz must be a positive number");
    }
    if (!Array.isArray(data.mea.traces) || data.mea.traces.length === 0) {
      throw new Error("mea.traces must be a non-empty array when present");
    }
    if (!Array.isArray(data.mea.traces[0]) || data.mea.traces[0].length === 0) {
      throw new Error("mea.traces[0] must be a non-empty array");
    }
    const meaHasSamples = Object.hasOwn(data.mea, "n_samples");
    let expectedTraceWidth = data.mea.traces[0]?.length;
    if (meaHasSamples) {
      if (!isPositiveInteger(data.mea.n_samples)) {
        throw new Error("mea.n_samples must be a positive integer");
      }
      expectedTraceWidth = data.mea.n_samples;
    }
    requireMatrixRows(
      data.mea.traces,
      expectedTraceWidth,
      "mea.traces",
      meaHasSamples ? "mea.n_samples" : "trace length",
    );
    if (data.mea.traces.length !== channelCount) {
      throw new Error(`mea.traces has ${data.mea.traces.length} channel rows but meta.channels lists ${channelCount}`);
    }
  }
}
