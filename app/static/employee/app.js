(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const els = {
    login: $("login-view"), chat: $("chat-view"), loginForm: $("login-form"), loginError: $("login-error"),
    username: $("username"), password: $("password"), who: $("who"), messages: $("messages"),
    quick: $("quick"), composer: $("composer"), input: $("input"), send: $("send"),
    newChat: $("new-chat"), logout: $("logout"),
  };
  // Token lives in sessionStorage (cleared when the tab closes). A production deployment
  // should use httpOnly, secure cookies instead.
  const app = { token: sessionStorage.getItem("hd_token"), user: null, conversationId: null, busy: false, categories: {} };

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  async function api(path, { method = "GET", body, auth = true } = {}) {
    const headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (auth && app.token) headers["Authorization"] = "Bearer " + app.token;
    let res;
    try {
      res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    } catch (e) {
      throw new ApiError(0, "Cannot reach the server. Please check your connection.");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON body */ }
    if (!res.ok) {
      const detail = data && data.detail;
      const message = typeof detail === "string" ? detail
        : res.status === 422 ? "Please check your input and try again."
        : "Something went wrong (" + res.status + ").";
      throw new ApiError(res.status, message);
    }
    return data;
  }

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text; // textContent only: never innerHTML
    return node;
  };
  const convKey = () => "hd_conv_" + app.user.username;
  const scrollDown = () => { els.messages.scrollTop = els.messages.scrollHeight; };

  function addMessage(role, text, meta = {}) {
    const wrap = el("div", "msg " + role);
    wrap.appendChild(el("div", "bubble", text));

    if (role === "assistant") {
      const sources = meta.sources || [];
      const cited = sources.filter((s) => s.cited);
      const shown = cited.length ? cited : meta.answerable ? sources : [];
      const names = [...new Set(shown.map((s) => s.source_name))];
      if (names.length) {
        const row = el("div", "sources");
        row.appendChild(el("span", "", cited.length ? "Sources:" : "Related documents:"));
        names.forEach((n) => row.appendChild(el("span", "chip", n)));
        wrap.appendChild(row);
      }
      if (meta.ticket) {
        const t = meta.ticket;
        const card = el("div", "ticket-card");
        card.appendChild(el("strong", "", "Ticket " + t.ticket_id));
        card.appendChild(el("div", "", t.title));
        card.appendChild(el("div", "muted", (app.categories[t.category] || t.category) + " | " + t.priority + " priority | " + t.status));
        wrap.appendChild(card);
      }
    }
    els.messages.appendChild(wrap);
    scrollDown();
  }

  function setQuick(options) {
    els.quick.replaceChildren();
    (options || []).forEach((label) => {
      const b = el("button", "", label);
      b.type = "button";
      b.addEventListener("click", () => send(label));
      els.quick.appendChild(b);
    });
  }

  function setBusy(busy) {
    app.busy = busy;
    els.send.disabled = busy;
    els.input.disabled = busy;
  }

  function greet() {
    addMessage("assistant", "Hi " + app.user.full_name + "! Ask me an IT support or facilities question, or tell me if you'd like to raise a ticket.");
  }

  async function send(text) {
    text = text.trim();
    if (app.busy || !text) return;
    setBusy(true);
    setQuick([]);
    addMessage("user", text);
    els.input.value = "";
    els.input.style.height = "auto";
    const typing = el("div", "msg typing");
    typing.appendChild(el("div", "bubble", "Thinking..."));
    els.messages.appendChild(typing);
    scrollDown();
    try {
      const data = await api("/api/chat", { method: "POST", body: { message: text, conversation_id: app.conversationId } });
      app.conversationId = data.conversation_id;
      localStorage.setItem(convKey(), data.conversation_id);
      typing.remove();
      addMessage("assistant", data.reply, data);
      setQuick(data.quick_replies);
    } catch (e) {
      typing.remove();
      if (e.status === 401) { logout("Your session expired. Please sign in again."); return; }
      if (e.status === 404) { app.conversationId = null; localStorage.removeItem(convKey()); }
      addMessage("error", e.message);
    } finally {
      setBusy(false);
      els.input.focus();
    }
  }

  async function restore() {
    els.messages.replaceChildren();
    setQuick([]);
    app.conversationId = null;
    const saved = localStorage.getItem(convKey());
    if (!saved) { greet(); return; }
    try {
      const conv = await api("/api/chat/" + encodeURIComponent(saved));
      app.conversationId = conv.conversation_id;
      if (!conv.messages.length) { greet(); return; }
      conv.messages.forEach((m) => addMessage(m.role, m.content, m.meta || {}));
      const last = conv.messages[conv.messages.length - 1];
      if (conv.mode !== "idle" && last.meta) setQuick(last.meta.quick_replies);
    } catch (e) {
      if (e.status === 401) throw e;
      localStorage.removeItem(convKey());
      greet();
    }
  }

  function showLogin(message) {
    els.chat.hidden = true;
    els.login.hidden = false;
    els.loginError.textContent = message || "";
    els.username.focus();
  }

  async function showChat() {
    els.login.hidden = true;
    els.chat.hidden = false;
    els.who.textContent = app.user.full_name;
    await restore();
    els.input.focus();
  }

  function logout(message) {
    sessionStorage.removeItem("hd_token");
    app.token = null;
    app.user = null;
    app.conversationId = null;
    showLogin(message);
  }

  els.loginForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    els.loginError.textContent = "";
    try {
      const data = await api("/api/auth/login", {
        method: "POST", auth: false,
        body: { username: els.username.value.trim(), password: els.password.value },
      });
      app.token = data.access_token;
      app.user = data.user;
      sessionStorage.setItem("hd_token", app.token);
      els.password.value = "";
      await showChat();
    } catch (e) {
      els.loginError.textContent = e.message;
    }
  });

  els.composer.addEventListener("submit", (ev) => { ev.preventDefault(); send(els.input.value); });
  els.input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(els.input.value); }
  });
  els.input.addEventListener("input", () => {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 140) + "px";
  });
  els.newChat.addEventListener("click", () => {
    if (app.busy) return;
    localStorage.removeItem(convKey());
    restore();
  });
  els.logout.addEventListener("click", () => logout());

  (async function init() {
    try {
      const meta = await api("/api/meta", { auth: false });
      meta.categories.forEach((c) => { app.categories[c.value] = c.label; });
    } catch (e) { /* labels fall back to raw values */ }
    if (app.token) {
      try {
        app.user = await api("/api/auth/me");
        await showChat();
        return;
      } catch (e) {
        sessionStorage.removeItem("hd_token");
        app.token = null;
      }
    }
    showLogin();
  })();
})();