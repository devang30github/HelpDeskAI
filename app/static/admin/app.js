(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const S = { key: sessionStorage.getItem("hd_admin_key"), signedIn: false, meta: null, selected: null, convo: [], poll: null, searchTimer: null };

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  // Tiny DOM helper: strings become text nodes (never HTML).
  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c !== null && c !== undefined && c !== false) node.append(c instanceof Node ? c : String(c));
    }
    return node;
  }

  async function api(path, { method = "GET", json, form } = {}) {
    const headers = { "X-Admin-Key": S.key || "" };
    let body;
    if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
    else if (form) { body = form; }
    let res;
    try { res = await fetch(path, { method, headers, body }); }
    catch (e) { throw new ApiError(0, "Cannot reach the server."); }
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch (e) { /* no JSON body */ }
    if (!res.ok) {
      if (res.status === 401 && S.signedIn) signOut("Admin key rejected. Please sign in again.");
      const d = data && data.detail;
      const msg = typeof d === "string" ? d
        : Array.isArray(d) ? d.map((x) => x.msg).join("; ")
        : "Request failed (" + res.status + ").";
      throw new ApiError(res.status, msg);
    }
    return data;
  }

  // ---------- formatting ----------
  const title = (s) => String(s).replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const label = (v) => (S.meta.categories.find((c) => c.value === v) || {}).label || v;
  function parseUtc(iso) {
    if (!iso) return null;
    return new Date(/[zZ]$|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z"); // API timestamps are UTC
  }
  const fmt = (iso) => { const d = parseUtc(iso); return d ? d.toLocaleString() : "-"; };
  const badge = (cls, text) => h("span", { class: "badge " + cls }, text);
  const kb = (n) => (n === null || n === undefined ? "-" : (n / 1024).toFixed(1) + " KB");

  // ---------- sign in / out ----------
  function showGate(message) {
    $("app").hidden = true;
    $("gate").hidden = false;
    $("gate-error").textContent = message || "";
    $("admin-key").focus();
  }

  function signOut(message) {
    S.signedIn = false;
    S.key = null;
    sessionStorage.removeItem("hd_admin_key");
    stopPolling();
    $("detail").hidden = true;
    showGate(message);
  }

  async function enter() {
    S.signedIn = true;
    $("gate").hidden = true;
    $("app").hidden = false;
    if (!S.meta) {
      S.meta = await api("/api/meta");
      fillFilters();
    }
    await Promise.all([loadStats(), loadTickets()]);
  }

  $("gate-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    $("gate-error").textContent = "";
    S.key = $("admin-key").value.trim();
    try {
      await api("/api/admin/stats"); // verifies the key
      sessionStorage.setItem("hd_admin_key", S.key);
      $("admin-key").value = "";
      await enter();
    } catch (e) {
      S.key = null;
      $("gate-error").textContent = e.status === 401 ? "That admin key was not accepted." : e.message;
    }
  });
  $("signout").addEventListener("click", () => signOut());

  // ---------- tickets ----------
  function fillSelect(sel, placeholder, options) {
    sel.replaceChildren(h("option", { value: "" }, placeholder), ...options.map(([v, t]) => h("option", { value: v }, t)));
  }

  function fillFilters() {
    fillSelect($("f-category"), "All categories", S.meta.categories.map((c) => [c.value, c.label]));
    fillSelect($("f-priority"), "All priorities", S.meta.priorities.map((p) => [p, title(p)]));
    fillSelect($("f-status"), "All statuses", S.meta.statuses.map((s) => [s, title(s)]));
    fillSelect($("f-assignee"), "Any assignee", [["unassigned", "Unassigned"], ...S.meta.team_members.map((m) => [m, m])]);
  }

  const stat = (text, value, cls = "") =>
    h("div", { class: "stat " + cls }, h("div", { class: "stat-value" }, String(value)), h("div", { class: "stat-label" }, text));

  async function loadStats() {
    const s = await api("/api/admin/stats");
    $("stats").replaceChildren(
      stat("Total", s.total),
      stat("Open", s.by_status.open),
      stat("In progress", s.by_status.in_progress),
      stat("Resolved", s.by_status.resolved),
      stat("Closed", s.by_status.closed),
      stat("Overdue", s.overdue, s.overdue ? "alert" : ""),
      stat("Unassigned (active)", s.unassigned_open, s.unassigned_open ? "warn" : ""),
    );
    $("prio-summary").textContent = "By priority: " + S.meta.priorities.map((p) => title(p) + " " + s.by_priority[p]).join("  |  ");
  }

  async function loadTickets() {
    const params = new URLSearchParams();
    const filters = {
      q: $("f-q").value.trim(), category: $("f-category").value, priority: $("f-priority").value,
      status: $("f-status").value, assignee: $("f-assignee").value,
    };
    Object.entries(filters).forEach(([k, v]) => { if (v) params.set(k, v); });
    try {
      renderTickets(await api("/api/tickets?" + params.toString()));
    } catch (e) {
      $("ticket-list").replaceChildren(h("p", { class: "error" }, e.message));
    }
  }

  function renderTickets(tickets) {
    $("ticket-count").textContent = tickets.length + (tickets.length === 1 ? " ticket" : " tickets");
    $("ticket-list").replaceChildren(
      ...(tickets.length ? tickets.map(ticketCard) : [h("p", { class: "muted" }, "No tickets match these filters.")]),
    );
  }

  function ticketCard(t) {
    const open = () => openTicket(t.ticket_id);
    return h("article", {
      class: "card prio-" + t.priority + (t.is_overdue ? " overdue" : ""), tabindex: "0", role: "button",
      onclick: open, onkeydown: (e) => { if (e.key === "Enter") open(); },
    },
      h("div", { class: "card-top" },
        h("strong", {}, t.ticket_id),
        h("span", { class: "badges" },
          badge("prio-" + t.priority, title(t.priority)),
          badge("status-" + t.status, title(t.status)),
          t.is_overdue ? badge("overdue", "OVERDUE") : null)),
      h("div", { class: "card-title" }, t.title),
      h("div", { class: "card-meta" }, label(t.category) + " | " + fmt(t.created_at) + " | " + (t.assignee || "Unassigned")));
  }

  ["f-category", "f-priority", "f-status", "f-assignee"].forEach((id) => $(id).addEventListener("change", loadTickets));
  $("f-q").addEventListener("input", () => { clearTimeout(S.searchTimer); S.searchTimer = setTimeout(loadTickets, 300); });
  $("refresh").addEventListener("click", () => Promise.all([loadStats(), loadTickets()]));

  // ---------- ticket detail ----------
  async function openTicket(id) {
    S.selected = id;
    $("detail").hidden = false;
    $("detail-id").textContent = id;
    $("detail-body").replaceChildren(h("p", { class: "muted" }, "Loading..."));
    try {
      const [ticket, convo] = await Promise.all([
        api("/api/tickets/" + encodeURIComponent(id)),
        api("/api/admin/tickets/" + encodeURIComponent(id) + "/conversation"),
      ]);
      renderDetail(ticket, convo);
    } catch (e) {
      $("detail-body").replaceChildren(h("p", { class: "error" }, e.message));
    }
  }

  const fact = (k, v) => h("div", { class: "fact" }, h("dt", {}, k), h("dd", {}, v));
  const field = (text, control) => h("label", { class: "field" }, text, control);

  function renderDetail(t, convo) {
    S.convo = convo;
    const next = S.meta.status_transitions[t.status] || [];
    const statusSel = h("select", {}, [t.status, ...next].map((s) => h("option", { value: s, selected: s === t.status }, title(s))));
    const prioSel = h("select", {}, S.meta.priorities.map((p) => h("option", { value: p, selected: p === t.priority }, title(p))));
    const assignSel = h("select", {}, [
      ...(t.assignee ? [] : [h("option", { value: "" }, "Select assignee...")]),
      ...S.meta.team_members.map((m) => h("option", { value: m, selected: m === t.assignee }, m)),
    ]);
    const notes = h("textarea", { rows: "4", placeholder: "Resolution notes (required to resolve a ticket)" });
    notes.value = t.resolution_notes || "";
    const msg = h("p", { class: "muted", role: "status" });

    const save = h("button", { class: "primary", type: "button" }, "Save changes");
    save.addEventListener("click", async () => {
      const payload = {};
      if (statusSel.value !== t.status) payload.status = statusSel.value;
      if (prioSel.value !== t.priority) payload.priority = prioSel.value;
      if (assignSel.value && assignSel.value !== (t.assignee || "")) payload.assignee = assignSel.value;
      const text = notes.value.trim();
      if (text !== (t.resolution_notes || "")) payload.resolution_notes = text;
      if (!Object.keys(payload).length) { msg.className = "muted"; msg.textContent = "No changes to save."; return; }
      save.disabled = true;
      try {
        const updated = await api("/api/tickets/" + encodeURIComponent(t.ticket_id), { method: "PATCH", json: payload });
        renderDetail(updated, S.convo);
        $("detail-body").appendChild(h("p", { class: "muted", role: "status" }, "Saved."));
        await Promise.all([loadStats(), loadTickets()]);
      } catch (e) {
        msg.className = "error";
        msg.textContent = e.message;
        save.disabled = false;
      }
    });

    const transcript = convo.length
      ? convo.map((m) => h("div", { class: "line" },
          h("strong", {}, m.role === "user" ? "Employee" : "HelpDeskAI"),
          h("span", {}, m.content),
          m.meta && m.meta.sources && m.meta.sources.length ? h("small", {}, "Sources: " + m.meta.sources.join(", ")) : null))
      : [h("p", { class: "muted" }, "No linked conversation.")];

    $("detail-body").replaceChildren(
      h("div", { class: "badges" },
        badge("prio-" + t.priority, title(t.priority)), badge("status-" + t.status, title(t.status)),
        t.is_overdue ? badge("overdue", "OVERDUE") : null),
      h("h2", {}, t.title),
      h("dl", { class: "facts" },
        fact("Requester", t.requester), fact("Category", label(t.category)), fact("Created", fmt(t.created_at)),
        fact("Updated", fmt(t.updated_at)), fact("Due", fmt(t.due_at)), fact("Assignee", t.assignee || "Unassigned")),
      h("h3", {}, "Description"),
      h("p", { class: "pre" }, t.description),
      h("h3", {}, "Manage"),
      h("div", { class: "form-grid" }, field("Status", statusSel), field("Priority", prioSel), field("Assignee", assignSel)),
      field("Resolution notes", notes), save, msg,
      h("h3", {}, "Conversation"),
      h("div", { class: "transcript" }, transcript));
  }

  $("detail-close").addEventListener("click", () => { $("detail").hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("detail").hidden = true; });

  // ---------- knowledge base ----------
  function stopPolling() { if (S.poll) { clearInterval(S.poll); S.poll = null; } }

  function startPolling() {
    stopPolling();
    S.poll = setInterval(async () => {
      try {
        const st = await api("/api/admin/kb/reindex");
        if (st.state !== "running") { stopPolling(); await loadKb(); }
      } catch (e) { stopPolling(); }
    }, 2000);
  }

  function renderReindex(st) {
    const btn = $("reindex-btn");
    btn.disabled = st.state === "running";
    btn.textContent = st.state === "running" ? "Rebuilding..." : "Rebuild index";
    const msg = $("kb-msg");
    if (st.state === "failed") { msg.className = "error"; msg.textContent = "Rebuild failed: " + st.error; }
    else if (st.state === "succeeded" && st.result) {
      msg.className = "muted";
      msg.textContent = "Index rebuilt: " + st.result.num_documents + " documents, " + st.result.num_chunks + " chunks.";
    }
  }

  function renderKb(data) {
    const idx = data.index;
    const banner = $("kb-banner");
    banner.className = "banner " + (data.stale ? "warn" : "ok");
    banner.replaceChildren(
      h("strong", {}, idx.built ? (data.stale ? "Index needs a rebuild" : "Index is up to date") : "Index not built yet"),
      h("div", { class: "muted" }, idx.built
        ? "Built " + fmt(idx.built_at) + " | " + idx.num_documents + " documents | " + idx.num_chunks + " chunks"
        : "Upload documents, then click 'Rebuild index'."),
      data.stale && data.stale_reasons.length ? h("div", { class: "muted" }, data.stale_reasons.join("; ")) : null,
      h("div", { class: "muted" }, "Allowed: " + data.limits.extensions.join(", ") + " | max " + data.limits.max_upload_mb + " MB"));

    const rows = data.documents.map((d) => h("tr", {},
      h("td", {}, d.filename), h("td", {}, d.title || "-"), h("td", {}, d.category ? label(d.category) : "-"),
      h("td", {}, d.num_chunks === null ? "-" : d.num_chunks), h("td", {}, kb(d.size_bytes)),
      h("td", {}, badge(d.state, title(d.state))),
      h("td", {}, d.state === "removed" ? "" : h("button", {
        type: "button",
        onclick: async () => {
          if (!confirm("Delete " + d.filename + "? You will need to rebuild the index afterwards.")) return;
          try { await api("/api/admin/kb/documents/" + encodeURIComponent(d.filename), { method: "DELETE" }); await loadKb(); }
          catch (e) { $("kb-msg").className = "error"; $("kb-msg").textContent = e.message; }
        },
      }, "Delete"))));
    $("kb-table").replaceChildren(rows.length
      ? h("table", {}, h("thead", {}, h("tr", {}, ...["File", "Title", "Category", "Chunks", "Size", "Status", ""].map((x) => h("th", {}, x)))), h("tbody", {}, rows))
      : h("p", { class: "muted" }, "No documents yet."));
    renderReindex(data.reindex);
  }

  async function loadKb() {
    try {
      const data = await api("/api/admin/kb");
      renderKb(data);
      if (data.reindex.state === "running") startPolling();
    } catch (e) { $("kb-msg").className = "error"; $("kb-msg").textContent = e.message; }
  }

  $("upload-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = $("kb-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("replace", $("kb-replace").checked ? "true" : "false");
    const msg = $("kb-msg");
    msg.className = "muted";
    msg.textContent = "Uploading...";
    try {
      const info = await api("/api/admin/kb/documents", { method: "POST", form });
      msg.textContent = (info.replaced ? "Replaced " : "Uploaded ") + info.filename + " (" + info.words + " words). Rebuild the index to include it."
        + (info.warnings.length ? " Warnings: " + info.warnings.join(" ") : "");
      $("upload-form").reset();
      await loadKb();
    } catch (e) { msg.className = "error"; msg.textContent = e.message; }
  });

  $("reindex-btn").addEventListener("click", async () => {
    try {
      renderReindex(await api("/api/admin/kb/reindex", { method: "POST" }));
      startPolling();
    } catch (e) { $("kb-msg").className = "error"; $("kb-msg").textContent = e.message; }
  });

  // ---------- tabs and startup ----------
  function switchTab(name) {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $("tab-tickets").hidden = name !== "tickets";
    $("tab-kb").hidden = name !== "kb";
    if (name === "kb") loadKb(); else stopPolling();
  }
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

  (async function init() {
    if (S.key) {
      try { await api("/api/admin/stats"); await enter(); return; }
      catch (e) { S.key = null; sessionStorage.removeItem("hd_admin_key"); }
    }
    showGate();
  })();
})();