(function() {
  const BASE_NODE_WIDTH = 220;
  const BASE_NODE_HEIGHT = 124;

  const state = {
    bundle: null,
    graphId: "model",
    selectedNodeId: null,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
    graphWidth: 0,
    graphHeight: 0,
    fileName: "llm_viewer",
    config: null,
    uiScale: 1,
  };

  const fileInput = document.getElementById("file-input");
  const openButton = document.getElementById("open-button");
  const openButtonTop = document.getElementById("open-button-top");
  const dropzone = document.getElementById("dropzone");
  const titlebar = document.getElementById("titlebar");
  const titleText = document.getElementById("title-text");
  const profileSelectTop = document.getElementById("profile-select-top");
  const errorBanner = document.getElementById("error-banner");
  const graphSurface = document.getElementById("graph-surface");
  const nodeLayer = document.getElementById("node-layer");
  const edgeLayer = document.getElementById("edge-layer");
  const sidebar = document.getElementById("sidebar");
  const sidebarObject = document.getElementById("sidebar-object");
  const sidebarClose = document.getElementById("sidebar-close");
  const graphModelButton = document.getElementById("graph-model");
  const graphBlockButton = document.getElementById("graph-block");
  const topToolbar = document.getElementById("top-toolbar");
  const bottomToolbar = document.getElementById("bottom-toolbar");
  const zoomInButton = document.getElementById("zoom-in");
  const zoomOutButton = document.getElementById("zoom-out");
  const zoomResetButton = document.getElementById("zoom-reset");
  const fitViewButton = document.getElementById("fit-view");
  const toggleGraphButton = document.getElementById("toggle-graph");
  const dragState = {
    active: false,
    startX: 0,
    startY: 0,
    baseOffsetX: 0,
    baseOffsetY: 0,
  };

  function activeGraph() {
    if (!state.bundle) {
      return null;
    }
    return state.bundle.graphs.find((graph) => graph.id === state.graphId) || null;
  }

  function nodeWidth() {
    return BASE_NODE_WIDTH * state.uiScale;
  }

  function nodeHeight() {
    return BASE_NODE_HEIGHT * state.uiScale;
  }

  function updateResponsiveScale() {
    const width = graphSurface.clientWidth || window.innerWidth || 1440;
    const height = graphSurface.clientHeight || window.innerHeight || 900;
    const scale = Math.max(1.08, Math.min(1.5, Math.min(width / 1440, height / 900) * 1.14));
    state.uiScale = Number(scale.toFixed(3));
    document.documentElement.style.setProperty("--ui-scale", String(state.uiScale));
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.add("hidden");
  }

  function setProfile(value) {
    profileSelectTop.value = value;
  }

  async function fetchBundle(config, profile) {
    const response = await fetch("/api/graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile,
        config,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }
    return payload;
  }

  async function uploadFile(file) {
    clearError();
    const rawText = await file.text();
    let config;
    try {
      config = JSON.parse(rawText);
    } catch (error) {
      showError("Invalid JSON: " + error.message);
      return;
    }

    try {
      const payload = await fetchBundle(config, profileSelectTop.value);
      state.bundle = payload;
      state.config = config;
      state.selectedNodeId = null;
      state.fileName = file.name;
      titleText.textContent = file.name + " · " + profileSelectTop.value;
      dropzone.classList.add("hidden");
      render();
      fitView();
    } catch (error) {
      showError(error.message);
    }
  }

  async function reloadGraphForProfile(profile) {
    setProfile(profile);
    if (!state.config) {
      return;
    }
    clearError();
    try {
      const payload = await fetchBundle(state.config, profile);
      state.bundle = payload;
      titleText.textContent = state.fileName + " · " + profile;
      const nextGraph = payload.graphs.find((graph) => graph.id === state.graphId);
      state.graphId = nextGraph ? nextGraph.id : payload.graphs[0].id;
      state.selectedNodeId = null;
      render();
      fitView();
    } catch (error) {
      showError(error.message);
    }
  }

  function openBlockGraph(node) {
    if (!state.bundle || !node || node.attrs.block_graph_id !== "block") {
      return;
    }
    state.graphId = "block";
    state.selectedNodeId = null;
    render();
  }

  function familyName(node) {
    const value = String(node.op_family || "").toLowerCase();
    if (value === "lmhead") {
      return "lmhead";
    }
    return value || "default";
  }

  function formatValue(value) {
    if (value === null || value === undefined) {
      return "null";
    }
    if (typeof value === "string") {
      return value;
    }
    if (Array.isArray(value) && value.every((item) => Array.isArray(item))) {
      if (value.length === 1) {
        return JSON.stringify(value[0]);
      }
      return value.map((item) => JSON.stringify(item)).join("\n");
    }
    return JSON.stringify(value);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function shapeText(value) {
    if (!value) {
      return "[]";
    }
    if (Array.isArray(value) && value.length === 1 && Array.isArray(value[0])) {
      return JSON.stringify(value[0]);
    }
    return JSON.stringify(value);
  }

  function edgeLabelLines(edge) {
    if (edge.shape) {
      return [shapeText(edge.shape)];
    }
    if (edge.tensor_name) {
      return [edge.tensor_name];
    }
    return [];
  }

  function edgeOffsetMaps(graph) {
    const outgoing = new Map();
    const incoming = new Map();

    for (const edge of graph.edges) {
      if (!outgoing.has(edge.source)) {
        outgoing.set(edge.source, []);
      }
      if (!incoming.has(edge.target)) {
        incoming.set(edge.target, []);
      }
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    }

    const offsetByEdge = new Map();

    for (const edges of outgoing.values()) {
      edges.forEach((edge, index) => {
        const center = (edges.length - 1) / 2;
        offsetByEdge.set(edge, (index - center) * 34);
      });
    }

    for (const edges of incoming.values()) {
      edges.forEach((edge, index) => {
        const center = (edges.length - 1) / 2;
        const current = offsetByEdge.get(edge) || 0;
        offsetByEdge.set(edge, current + (index - center) * 18);
      });
    }

    return offsetByEdge;
  }

  function edgeLabelMaps(graph) {
    const outgoing = new Map();
    const incoming = new Map();

    for (const edge of graph.edges) {
      if (!outgoing.has(edge.source)) {
        outgoing.set(edge.source, []);
      }
      if (!incoming.has(edge.target)) {
        incoming.set(edge.target, []);
      }
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    }

    const labelByEdge = new Map();

    for (const edges of outgoing.values()) {
      const sorted = [...edges].sort((left, right) => left.target.localeCompare(right.target));
      sorted.forEach((edge, index) => {
        labelByEdge.set(edge, {
          sourceIndex: index,
          sourceCount: sorted.length,
          targetIndex: 0,
          targetCount: 1,
        });
      });
    }

    for (const edges of incoming.values()) {
      const sorted = [...edges].sort((left, right) => left.source.localeCompare(right.source));
      sorted.forEach((edge, index) => {
        const current = labelByEdge.get(edge) || {
          sourceIndex: 0,
          sourceCount: 1,
          targetIndex: 0,
          targetCount: 1,
        };
        current.targetIndex = index;
        current.targetCount = sorted.length;
        labelByEdge.set(edge, current);
      });
    }

    return labelByEdge;
  }

  function edgePortMaps(graph) {
    const outgoing = new Map();
    const incoming = new Map();

    for (const edge of graph.edges) {
      if (!outgoing.has(edge.source)) {
        outgoing.set(edge.source, []);
      }
      if (!incoming.has(edge.target)) {
        incoming.set(edge.target, []);
      }
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    }

    const portByEdge = new Map();

    for (const edges of outgoing.values()) {
      const sorted = [...edges].sort((left, right) => left.target.localeCompare(right.target));
      sorted.forEach((edge, index) => {
        const center = (sorted.length - 1) / 2;
        portByEdge.set(edge, { source: (index - center) * 26, target: 0 });
      });
    }

    for (const edges of incoming.values()) {
      const sorted = [...edges].sort((left, right) => left.source.localeCompare(right.source));
      sorted.forEach((edge, index) => {
        const center = (sorted.length - 1) / 2;
        const current = portByEdge.get(edge) || { source: 0, target: 0 };
        current.target = (index - center) * 34;
        portByEdge.set(edge, current);
      });
    }

    return portByEdge;
  }

  function layoutGraph(graph) {
    const incoming = new Map(graph.nodes.map((node) => [node.id, []]));
    const outgoing = new Map(graph.nodes.map((node) => [node.id, []]));
    for (const edge of graph.edges) {
      if (!incoming.has(edge.target)) {
        incoming.set(edge.target, []);
      }
      if (!outgoing.has(edge.source)) {
        outgoing.set(edge.source, []);
      }
      incoming.get(edge.target).push(edge);
      outgoing.get(edge.source).push(edge);
    }

    const depthMemo = new Map();
    function depth(nodeId) {
      if (depthMemo.has(nodeId)) {
        return depthMemo.get(nodeId);
      }
      const parents = incoming.get(nodeId) || [];
      if (!parents.length) {
        depthMemo.set(nodeId, 0);
        return 0;
      }
      const result = Math.max(...parents.map((edge) => depth(edge.source))) + 1;
      depthMemo.set(nodeId, result);
      return result;
    }

    for (const node of graph.nodes) {
      depth(node.id);
    }

    const groups = new Map();
    for (const node of graph.nodes) {
      const layer = depthMemo.get(node.id) || 0;
      if (!groups.has(layer)) {
        groups.set(layer, []);
      }
      groups.get(layer).push(node);
    }

    const layerKeys = Array.from(groups.keys()).sort((a, b) => a - b);
    const positions = new Map();
    let maxWidth = 1080;
    let maxHeight = 640;
    const columnGap = 34 * state.uiScale;

    for (const layer of layerKeys) {
      const nodes = groups.get(layer);
      nodes.sort((left, right) => left.name.localeCompare(right.name));
      const nodeWidths = nodes.map((node) => {
        const fanIn = (incoming.get(node.id) || []).length;
        const fanOut = (outgoing.get(node.id) || []).length;
        const edgeTextWidth = Math.max(
          ...((incoming.get(node.id) || []).concat(outgoing.get(node.id) || []).map((edge) => {
            const text = edgeLabelLines(edge).join(" ");
            return Math.min(420, Math.max(nodeWidth(), text.length * (6 * state.uiScale)));
          })),
          nodeWidth()
        );
        return Math.max(nodeWidth(), edgeTextWidth, nodeWidth() + Math.max(fanIn, fanOut) * 28 * state.uiScale);
      });
      const rowWidth = nodeWidths.reduce((total, width) => total + width, 0) + Math.max(0, nodes.length - 1) * columnGap;
      const startX = Math.max(34, Math.floor((Math.max(maxWidth, rowWidth + 68) - rowWidth) / 2));
      let cursorX = startX;
      nodes.forEach((node, index) => {
        const width = nodeWidths[index];
        const x = cursorX;
        const y = 64 + layer * (178 * state.uiScale);
        positions.set(node.id, { x, y });
        maxWidth = Math.max(maxWidth, x + width);
        maxHeight = Math.max(maxHeight, y + nodeHeight() + 24 * state.uiScale);
        cursorX += width + columnGap;
      });
    }

    return { positions, maxWidth, maxHeight };
  }

  function appendArrowDefinitions() {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    marker.setAttribute("id", "arrowhead");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", String(6 * state.uiScale));
    marker.setAttribute("markerHeight", String(6 * state.uiScale));
    marker.setAttribute("orient", "auto-start-reverse");
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "path");
    polygon.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    polygon.setAttribute("fill", "#000000");
    marker.appendChild(polygon);
    defs.appendChild(marker);
    edgeLayer.appendChild(defs);
  }

  function shapeLines(value) {
    if (!value) {
      return [];
    }
    if (Array.isArray(value) && value.every((item) => Array.isArray(item))) {
      return value.map((item) => JSON.stringify(item));
    }
    return [shapeText(value)];
  }

  function nodeShapeRows(node) {
    const rows = [];
    for (const line of shapeLines(node.output_shapes)) {
      rows.push({ label: "output", value: line });
    }
    for (const line of shapeLines(node.param_shapes)) {
      rows.push({ label: "params", value: line });
    }
    return rows;
  }

  function applyViewport() {
    nodeLayer.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.zoom})`;
    edgeLayer.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.zoom})`;
  }

  function centerGraph() {
    const width = graphSurface.clientWidth || 0;
    const height = graphSurface.clientHeight || 0;
    state.offsetX = Math.max(24, (width - state.graphWidth * state.zoom) / 2);
    state.offsetY = Math.max(48, (height - state.graphHeight * state.zoom) / 2);
    applyViewport();
  }

  function renderGraph() {
    const graph = activeGraph();
    nodeLayer.innerHTML = "";
    edgeLayer.innerHTML = "";

    if (!graph) {
      return;
    }

    const { positions, maxWidth, maxHeight } = layoutGraph(graph);
    const edgeOffsets = edgeOffsetMaps(graph);
    const edgePorts = edgePortMaps(graph);
    const edgeLabels = edgeLabelMaps(graph);
    state.graphWidth = maxWidth;
    state.graphHeight = maxHeight;
    nodeLayer.style.width = maxWidth + "px";
    nodeLayer.style.height = maxHeight + "px";
    edgeLayer.setAttribute("width", String(maxWidth));
    edgeLayer.setAttribute("height", String(maxHeight));
    edgeLayer.setAttribute("viewBox", `0 0 ${maxWidth} ${maxHeight}`);
    applyViewport();
    appendArrowDefinitions();

    for (const edge of graph.edges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) {
        continue;
      }
      const spread = edgeOffsets.get(edge) || 0;
      const ports = edgePorts.get(edge) || { source: 0, target: 0 };
      const labelMeta = edgeLabels.get(edge) || {
        sourceIndex: 0,
        sourceCount: 1,
        targetIndex: 0,
        targetCount: 1,
      };
      const sourceCenterX = source.x + nodeWidth() / 2 + ports.source + spread * 0.25;
      const targetCenterX = target.x + nodeWidth() / 2 + ports.target;
      const x1 = sourceCenterX;
      const y1 = source.y + nodeHeight();
      const x2 = targetCenterX;
      const y2 = target.y;
      const midY = y1 + (y2 - y1) / 2;
      const lines = edgeLabelLines(edge);

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      let pathData = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
      let labelX = (x1 + x2) / 2 + spread * 0.2;
      let labelY = midY - 6;
      let labelAnchor = "middle";
      if (edge.edge_kind === "residual") {
        const laneX = Math.min(source.x, target.x) - (110 * state.uiScale + Math.abs(spread) * 0.2);
        const sourceSideX = source.x - 10 * state.uiScale;
        const targetSideX = target.x - 10 * state.uiScale;
        const midLaneY = y1 + (y2 - y1) * 0.58;
        pathData = [
          `M ${sourceSideX} ${y1 - 2 * state.uiScale}`,
          `C ${sourceSideX} ${y1 + 18 * state.uiScale}, ${laneX} ${y1 + 8 * state.uiScale}, ${laneX} ${midLaneY}`,
          `S ${targetSideX} ${y2 - 16 * state.uiScale}, ${targetSideX} ${y2}`
        ].join(" ");
        labelX = laneX - 8 * state.uiScale;
        labelY = midLaneY - 8 * state.uiScale;
        labelAnchor = "end";
      } else if (labelMeta.sourceCount > 1) {
        const center = (labelMeta.sourceCount - 1) / 2;
        const labelSpread = labelMeta.sourceIndex - center;
        labelX = x1 + (x2 - x1) * 0.74 + labelSpread * 72 * state.uiScale;
        labelY = y1 + 20 * state.uiScale + Math.abs(labelSpread) * 18 * state.uiScale;
        labelAnchor = labelSpread < 0 ? "end" : labelSpread > 0 ? "start" : "middle";
      } else if (labelMeta.targetCount > 1) {
        const center = (labelMeta.targetCount - 1) / 2;
        const labelSpread = labelMeta.targetIndex - center;
        labelX = x2 - (x2 - x1) * 0.22 + labelSpread * 62 * state.uiScale;
        labelY = y2 - 22 * state.uiScale - Math.abs(labelSpread) * 13 * state.uiScale;
        labelAnchor = labelSpread < 0 ? "end" : labelSpread > 0 ? "start" : "middle";
      }
      path.setAttribute("class", `edge-path${edge.edge_kind === "residual" ? " residual" : ""}`);
      path.setAttribute("d", pathData);
      path.setAttribute("marker-end", "url(#arrowhead)");
      edgeLayer.appendChild(path);

      if ((edge.tensor_name || edge.shape) && lines.length) {
        const longest = lines.reduce((max, line) => Math.max(max, line.length), 0);
        const labelWidth = Math.min(220 * state.uiScale, Math.max(42, longest * 6.2 * state.uiScale));
        const labelHeight = Math.max(14, lines.length * 11) * state.uiScale;
        const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        box.setAttribute("class", "edge-label-box");
        const boxX = labelAnchor === "start" ? labelX - 4 * state.uiScale : labelAnchor === "end" ? labelX - labelWidth + 4 * state.uiScale : labelX - labelWidth / 2;
        box.setAttribute("x", String(boxX));
        box.setAttribute("y", String(labelY - 9 * state.uiScale));
        box.setAttribute("width", String(labelWidth));
        box.setAttribute("height", String(labelHeight + 4 * state.uiScale));
        edgeLayer.appendChild(box);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "edge-label");
        label.setAttribute("x", String(labelX));
        label.setAttribute("y", String(labelY));
        label.setAttribute("text-anchor", labelAnchor);
        lines.forEach((line, index) => {
          const span = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
          span.setAttribute("x", String(labelX));
          if (index === 0) {
            span.setAttribute("dy", "0");
          } else {
            span.setAttribute("dy", String(10 * state.uiScale));
          }
          span.textContent = line;
          label.appendChild(span);
        });
        edgeLayer.appendChild(label);
      }
    }

    for (const node of graph.nodes) {
      const pos = positions.get(node.id);
      if (!pos) {
        continue;
      }
      const card = document.createElement("button");
      card.type = "button";
      card.className = `node-card ${familyName(node)}${state.selectedNodeId === node.id ? " selected" : ""}`;
      card.style.left = pos.x + "px";
      card.style.top = pos.y + "px";
      const bodyRows = nodeShapeRows(node);
      card.innerHTML = `
        <div class="node-card-header">
          <div class="node-card-title">${escapeHtml(node.name)}</div>
        </div>
        <div class="node-card-body">
          ${bodyRows.map((row) => `<div><strong>${escapeHtml(row.label)}</strong> ${escapeHtml(row.value)}</div>`).join("")}
        </div>
      `;
      card.addEventListener("click", () => {
        state.selectedNodeId = node.id;
        sidebar.classList.remove("hidden");
        renderGraph();
        renderSidebar();
      });
      card.addEventListener("dblclick", () => {
        openBlockGraph(node);
      });
      nodeLayer.appendChild(card);
    }
  }

  function sidebarRows(items) {
    return items.map(([name, value]) => `
      <div class="sidebar-item">
        <span class="sidebar-item-name">${escapeHtml(name)}</span>
        <span class="sidebar-item-value-list">${formatLines(value).map((line) => `
          <div class="sidebar-item-value">
            <div class="sidebar-item-value-line">${escapeHtml(line)}</div>
          </div>
        `).join("")}</span>
      </div>
    `).join("");
  }

  function formatLines(value) {
    if (value === null || value === undefined) {
      return ["n/a"];
    }
    if (typeof value === "string") {
      return value.split("\n");
    }
    if (Array.isArray(value) && value.every((item) => Array.isArray(item))) {
      return value.length ? value.map((item) => JSON.stringify(item)) : ["[]"];
    }
    return [formatValue(value)];
  }

  function modelSummaryRows() {
    if (!state.config || !state.bundle) {
      return "";
    }

    const config = state.config;
    const metadata = state.bundle.metadata || {};
    const rows = [
      ["file", state.fileName],
      ["profile", metadata.profile || profileSelectTop.value],
      ["architecture", Array.isArray(config.architectures) ? config.architectures.join(", ") : (config.architectures || "n/a")],
      ["model_type", config.model_type || "n/a"],
      ["hidden_size", config.hidden_size],
      ["num_hidden_layers", config.num_hidden_layers],
      ["num_attention_heads", config.num_attention_heads],
      ["num_key_value_heads", config.num_key_value_heads],
      ["intermediate_size", config.intermediate_size],
      ["vocab_size", config.vocab_size],
      ["max_position_embeddings", config.max_position_embeddings],
      ["torch_dtype", config.torch_dtype],
      ["use_cache", config.use_cache],
      ["tie_word_embeddings", config.tie_word_embeddings],
    ];
    return sidebarRows(rows.filter(([, value]) => value !== undefined));
  }

  function renderModelSidebar() {
    sidebarObject.innerHTML = `
      <div class="sidebar-header">${escapeHtml(state.fileName)}</div>
      <div class="sidebar-subtitle">Model config summary</div>
      <div class="sidebar-section">Overview</div>
      <div class="sidebar-paragraph">Useful fields from model_config.json and backend metadata for current graph profile.</div>
      ${modelSummaryRows()}
    `;
  }

  function renderSidebar() {
    const graph = activeGraph();
    if (!state.bundle) {
      sidebarObject.innerHTML = '<div class="empty-state">Select a node.</div>';
      return;
    }
    if (!graph || !state.selectedNodeId) {
      renderModelSidebar();
      return;
    }
    const node = graph.nodes.find((item) => item.id === state.selectedNodeId);
    if (!node) {
      renderModelSidebar();
      return;
    }
    sidebarObject.innerHTML = `
      <div class="sidebar-header">${escapeHtml(node.name)}</div>
      <div class="sidebar-subtitle">${escapeHtml(node.op_family)} / ${escapeHtml(node.kind)}</div>
      <div class="sidebar-section">Shapes</div>
      ${sidebarRows([
        ["input", node.input_shapes],
        ["output", node.output_shapes],
        ["params", node.param_shapes],
      ])}
      <div class="sidebar-section">Attributes</div>
      <div class="sidebar-paragraph">Computed by backend graph extraction for current runtime profile.</div>
      ${sidebarRows(Object.entries(node.attrs || {}))}
      <div class="sidebar-section">Source</div>
      ${sidebarRows([
        ["module_path", node.module_path || "n/a"],
        ["source_file", node.source_file || "n/a"],
        ["source_line", node.source_line || "n/a"],
      ])}
    `;
  }

  function renderTabs() {
    graphModelButton.classList.toggle("active", state.graphId === "model");
    graphBlockButton.classList.toggle("active", state.graphId === "block");
  }

  function render() {
    updateResponsiveScale();
    const loaded = Boolean(state.bundle);
    dropzone.classList.toggle("hidden", loaded);
    titlebar.classList.toggle("hidden", !loaded);
    topToolbar.classList.toggle("hidden", !loaded);
    bottomToolbar.classList.toggle("hidden", !loaded);
    sidebar.classList.toggle("hidden", !loaded);
    renderTabs();
    renderGraph();
    renderSidebar();
  }

  function toggleModelSidebar() {
    if (!state.bundle) {
      return;
    }
    if (!sidebar.classList.contains("hidden") && state.selectedNodeId === null) {
      sidebar.classList.add("hidden");
      return;
    }
    state.selectedNodeId = null;
    sidebar.classList.remove("hidden");
    renderGraph();
    renderSidebar();
  }

  function fitView() {
    if (!state.graphWidth || !state.graphHeight) {
      return;
    }
    const width = graphSurface.clientWidth || 1;
    const height = graphSurface.clientHeight || 1;
    const zoomX = (width - 80) / state.graphWidth;
    const zoomY = (height - 96) / state.graphHeight;
    state.zoom = Math.max(0.45, Math.min(1.2, zoomX, zoomY));
    centerGraph();
  }

  function resetView() {
    state.zoom = 1;
    centerGraph();
  }

  function zoomAtCenter(nextZoom) {
    if (!state.graphWidth || !state.graphHeight) {
      return;
    }
    const width = graphSurface.clientWidth || 0;
    const height = graphSurface.clientHeight || 0;
    const centerGraphX = (width / 2 - state.offsetX) / state.zoom;
    const centerGraphY = (height / 2 - state.offsetY) / state.zoom;
    state.zoom = nextZoom;
    state.offsetX = width / 2 - centerGraphX * state.zoom;
    state.offsetY = height / 2 - centerGraphY * state.zoom;
    applyViewport();
  }

  function openPicker() {
    fileInput.click();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) {
      uploadFile(file).catch((error) => showError(error.message));
    }
  });

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
    if (file) {
      uploadFile(file).catch((error) => showError(error.message));
    }
  });

  openButton.addEventListener("click", openPicker);
  openButtonTop.addEventListener("click", openPicker);

  profileSelectTop.addEventListener("change", () => {
    reloadGraphForProfile(profileSelectTop.value).catch((error) => showError(error.message));
  });

  graphModelButton.addEventListener("click", () => {
    state.graphId = "model";
    state.selectedNodeId = null;
    render();
    fitView();
  });
  graphBlockButton.addEventListener("click", () => {
    state.graphId = "block";
    state.selectedNodeId = null;
    render();
    fitView();
  });
  toggleGraphButton.addEventListener("click", () => {
    toggleModelSidebar();
  });

  zoomInButton.addEventListener("click", () => {
    zoomAtCenter(Math.min(1.8, state.zoom + 0.1));
  });
  zoomOutButton.addEventListener("click", () => {
    zoomAtCenter(Math.max(0.45, state.zoom - 0.1));
  });
  zoomResetButton.addEventListener("click", resetView);
  fitViewButton.addEventListener("click", fitView);
  sidebarClose.addEventListener("click", () => {
    sidebar.classList.add("hidden");
  });

  graphSurface.addEventListener("mousedown", (event) => {
    if (!state.bundle) {
      return;
    }
    if (event.target.closest(".node-card")) {
      return;
    }
    dragState.active = true;
    dragState.startX = event.clientX;
    dragState.startY = event.clientY;
    dragState.baseOffsetX = state.offsetX;
    dragState.baseOffsetY = state.offsetY;
    graphSurface.classList.add("dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragState.active) {
      return;
    }
    state.offsetX = dragState.baseOffsetX + (event.clientX - dragState.startX);
    state.offsetY = dragState.baseOffsetY + (event.clientY - dragState.startY);
    applyViewport();
  });

  window.addEventListener("mouseup", () => {
    dragState.active = false;
    graphSurface.classList.remove("dragging");
  });

  graphSurface.addEventListener("wheel", (event) => {
    if (!state.bundle) {
      return;
    }
    event.preventDefault();
    const direction = event.deltaY < 0 ? 0.08 : -0.08;
    zoomAtCenter(Math.max(0.45, Math.min(1.8, state.zoom + direction)));
  }, { passive: false });

  window.addEventListener("resize", () => {
    updateResponsiveScale();
    if (state.bundle) {
      render();
      fitView();
    }
  });

  setProfile("prefill");
})();
