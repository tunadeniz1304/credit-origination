/* Anil2 SPA — Alpine.js (CSP build) component. User data is rendered only
   through x-text / textContent (never innerHTML), which prevents stored XSS.
   The CSP build evaluates directive expressions without eval/new Function, so
   templates may only use simple expressions: anything with arrow functions,
   globals (Math, window) or several statements lives in a method below. */
"use strict";

const EMPTY_FORM = () => ({
  name: "", identity_no: "", birth_date: "1988-04-12", phone: "05321234567", email: "ornek@example.com",
  address: "Moda Cad. No:5 Kadıköy İstanbul", iban: "TR330006100519786457841326", gender: null, province: "İstanbul",
  monthly_income: 45000, employment_type: "MAASLI", employer_name: "Anadolu Bilişim Ltd. Şti.", product: "IHTIYAC",
  requested_amount: 200000, requested_term_months: 36,
  consents: { kvkk_aydinlatma: false, acik_riza: false, kkb_sorgu: false, edevlet_sorgu: true, acik_bankacilik: true },
});

// Chart.js instances must stay outside Alpine's reactive proxies.
const CHARTS = new Map();

// Codes the backend does not label in /api/v1/labels (UI-only vocabularies).
const UI_LABELS = {
  role: { basvuran: "Başvuran", uzman: "Krediler Uzmanı", kidemli_uzman: "Kıdemli Krediler Uzmanı", komite: "Kredi Komitesi", model_yoneticisi: "Model Yöneticisi", admin: "Sistem Yöneticisi" },
  document_status: { YUKLENDI: "Yüklendi", ISLENDI: "İşlendi", SUPHELI: "Şüpheli", OCR_GEREKLI: "Uzman incelemesi gerekli (OCR yok)" },
  drift: { STABIL: "Stabil", ORTA: "Orta kayma", ANLAMLI_KAYMA: "Anlamlı kayma" },
  rule_action: { decline: "Ret", refer: "Uzmana yönlendir" },
  fairness_attr: { gender: "Cinsiyet", age_band: "Yaş bandı", province: "İl", SEX: "Cinsiyet", AGE_BAND: "Yaş bandı", EDUCATION: "Eğitim", MARRIAGE: "Medeni durum", FOREIGN_WORKER: "Yabancı işçi" },
  model_role: { champion: "Şampiyon", challenger: "Aday (challenger)" },
  model_family: { lightgbm: "Monotonik LightGBM", logistic: "Lojistik regresyon", scorecard: "WoE skor kartı", lightgbm_uncalibrated: "LightGBM (kalibrasyonsuz)" },
  llm_mode: { live: "canlı", demo: "demo" },
  node_kind: { application: "Başvuru", phone: "Telefon", iban: "IBAN", device: "Cihaz", address: "Adres" },
  persona: { temiz: "Temiz dosya", ince_dosya: "İnce dosya", yuksek_dsr: "Yüksek borç/gelir", asiri_borclu: "Aşırı borçlu", kurcalanmis: "Kurcalanmış belge", halka: "Olası dolandırıcılık halkası", halka_2: "Olası dolandırıcılık halkası", halka_3: "Olası dolandırıcılık halkası", gri: "Gri bölge", gecikmeli: "Gecikme geçmişi", takipte: "Takipte kayıt" },
  dataset: { uci_taiwan: "UCI Tayvan kredi kartı verisi", german_credit: "German Credit verisi" },
};

document.addEventListener("alpine:init", () => { window.Alpine.data("platform", platform); });

function platform() {
  return {
    user: null, view: "portal", authConfig: { demo_mode: false, registration_enabled: false, captcha: false },
    registerForm: { username: "", full_name: "", password: "", captcha_token: "" }, theme: "light", llm: null, error: "", toast: "", busy: false,
    loginForm: { username: "", password: "" },
    demoUsers: [["basvuran", "Başvuran"], ["uzman", "Uzman"], ["kidemli", "Kıdemli uzman"], ["komite", "Komite"], ["modelyon", "Model yöneticisi"], ["admin", "Admin"]],
    demoTckn: { temiz: "68846908942", ince_dosya: "29551411288", gri: "24377158546", gecikmeli: "21134086364" },
    applications: [], current: null, wizard: { open: false, step: 1 }, form: EMPTY_FORM(), quote: null,
    upload: { code: "IDENTITY", over: false }, objection: "",
    queue: [], queuePage: { total: 0, limit: 25, offset: 0, sla_breached: 0 }, queueLoading: false, queueSeq: 0, searchTimer: null,
    queueFilter: { q: "", state: "", product: "", sla: "", mine: false, limit: 25, offset: 0 }, fieldEdit: null,
    reviews: [], staff: null, decisionForm: { action: "ONAY", justification: "", amount: null, term_months: null },
    policyQ: "Borç servis oranı sınırı nedir?", policyA: null, dash: {}, dashTab: "genel", promoteError: "",
    validation: null, validationError: "", valSet: "", valCalModel: "", seedResult: "",
    staffTabs: [["ozet", "Özet"], ["belgeler", "Belgeler"], ["kkb", "KKB"], ["nakit", "Nakit akışı"], ["karar", "Karar + SHAP"], ["fiyat", "Fiyatlama"], ["halka", "Halka"], ["memo", "AI memorandum"], ["politika", "Politika sor"], ["karar_ver", "Karar ver"]],
    bureauKeys: [["bureau_score", "KKB notu"], ["bureau_hit", "KKB kaydı"], ["active_loans", "Aktif kredi"], ["bureau_utilisation", "Limit kullanımı"], ["delinquency_count_24m", "24 ay gecikme"], ["max_dpd_24m", "Azami gecikme günü"], ["inquiries_6m", "6 ay sorgu"], ["existing_debt_service", "Mevcut aylık taksit"], ["existing_dsr", "Mevcut DSR"], ["dsr", "DSR (talep)"], ["employment_months", "Çalışma süresi (ay)"]],

    labels: {}, health: null,

    // ------------------------------------------------------------ helpers
    /** Turkish label for an enum code: never show the raw code if a label exists. */
    lbl(kind, code) {
      if (code === null || code === undefined || code === "") return "—";
      const server = this.labels[kind] || {};
      const local = UI_LABELS[kind] || {};
      return server[code] || local[code] || String(code);
    },
    lblList(kind, codes) { return (codes || []).map((c) => this.lbl(kind, c)).join(", "); },
    get ocrMissing() { return !!(this.health && this.health.ocr && this.health.ocr.available === false); },
    labelEntries(kind) { return Object.entries(this.labels[kind] || {}); },
    tl(v) { return v === null || v === undefined ? "—" : Number(v).toLocaleString("tr-TR", { style: "currency", currency: "TRY" }); },
    pct(v) { return v === null || v === undefined ? "—" : "%" + (Number(v) * 100).toLocaleString("tr-TR", { maximumFractionDigits: 2 }); },
    date(v) { return v ? new Date(v).toLocaleString("tr-TR") : ""; },
    stateClass(s) {
      if (["OTOMATIK_ONAY", "TEKLIF_SUNULDU", "TEKLIF_KABUL", "SOZLESME_HAZIR", "KULLANDIRILDI", "ONAYLANDI"].includes(s)) return "ok";
      if (["OTOMATIK_RET", "REDDEDILDI", "IPTAL"].includes(s)) return "bad";
      return "warn";
    },
    get pricingKeys() {
      const t = (v) => this.tl(v), p = (v) => this.pct(v), n = (v) => v;
      return [["amount", "Tutar", t], ["term_months", "Vade (ay)", n], ["annual_rate", "Yıllık akdi faiz", p], ["instalment", "Aylık taksit", t], ["apr", "Yıllık maliyet oranı", p], ["total_payment", "Toplam geri ödeme", t], ["total_taxes", "BSMV + KKDF", t], ["upfront_fee", "Tahsis ücreti", t], ["expected_loss_annual", "Beklenen kayıp", t], ["capital_ratio", "Ekonomik sermaye oranı", p], ["raroc", "RAROC", p], ["legal_cap_annual", "Yasal tavan", p]];
    },
    round(v) { return Math.round(Number(v) || 0); },
    pricingValue(k) { return k[2](this.staff.decision.pricing[k[0]]); },
    firedRules() { return ((this.staff && this.staff.decision && this.staff.decision.rule_results) || []).filter((x) => x.fired); },
    memoSteps() { return ((this.staff && this.staff.memo && this.staff.memo.steps) || []).map((s) => s.tool).join(" → "); },
    notify(msg) { this.toast = msg; setTimeout(() => { this.toast = ""; }, 4000); },
    csrfToken() {
      // The session itself lives in an HttpOnly cookie the page cannot read; only
      // the CSRF double-submit value is readable and echoed on unsafe requests.
      const hit = document.cookie.split("; ").find((c) => c.startsWith("anil2_csrf="));
      return hit ? decodeURIComponent(hit.slice("anil2_csrf=".length)) : "";
    },
    async api(path, opts = {}) {
      const method = opts.method || "GET";
      const headers = Object.assign({}, opts.headers || {});
      if (method !== "GET" && method !== "HEAD") headers["X-CSRF-Token"] = this.csrfToken();
      let body = opts.body;
      if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
      const res = await fetch(path, { method, headers, body, credentials: "same-origin" });
      if (res.status === 401 && this.user && !opts.quiet401) { this.clearSession(); throw new Error("Oturum süresi doldu, lütfen yeniden giriş yapın."); }
      const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
      if (!res.ok) {
        const detail = data && data.detail;
        throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : (detail || res.statusText));
      }
      return data;
    },
    async download(path, name) {
      try {
        const res = await fetch(path, { credentials: "same-origin" });
        if (!res.ok) throw new Error("dosya alınamadı");
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement("a"); a.href = url; a.download = name; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      } catch (e) { this.notify(e.message); }
    },

    // ------------------------------------------------------------ session
    async init() {
      try { this.theme = localStorage.getItem("anil2-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (e) { /* ignore */ }
      this.api("/api/v1/auth/config").then((c) => { this.authConfig = c; }).catch(() => {});
      if (!this.csrfToken()) return; // no session cookie pair: show the login form
      try {
        const me = await this.api("/api/v1/auth/me", { quiet401: true });
        this.user = { full_name: me.full_name, role: me.role, role_label: me.role_label, username: me.username };
        await this.afterLogin();
      } catch (e) { this.user = null; }
    },
    toggleTheme() { this.theme = this.theme === "dark" ? "light" : "dark"; try { localStorage.setItem("anil2-theme", this.theme); } catch (e) { /* ignore */ } },
    async login() {
      this.error = "";
      try {
        const r = await this.api("/api/v1/auth/login", { method: "POST", json: this.loginForm });
        // r.access_token is for API clients; the browser relies on the HttpOnly cookie only.
        this.user = { full_name: r.full_name, role: r.role, role_label: r.role_label, username: r.username };
        this.loginForm = { username: "", password: "" };
        await this.afterLogin();
      } catch (e) { this.error = e.message; }
    },
    async register() {
      this.error = "";
      const body = { username: this.registerForm.username, full_name: this.registerForm.full_name, password: this.registerForm.password };
      if (this.authConfig.captcha) body.captcha_token = this.registerForm.captcha_token;
      try {
        const r = await this.api("/api/v1/auth/register", { method: "POST", json: body });
        this.user = { full_name: r.full_name, role: r.role, role_label: r.role_label, username: r.username };
        this.registerForm = { username: "", full_name: "", password: "", captcha_token: "" };
        await this.afterLogin();
      } catch (e) { this.error = e.message; }
    },
    demoLogin(username) { this.loginForm = { username, password: "Demo123!" }; return this.login(); },
    clearSession() { this.user = null; this.current = null; this.staff = null; this.applications = []; this.queue = []; this.reviews = []; this.dash = {}; this.dashTab = "genel"; this.validation = null; this.promoteError = ""; },
    async logout() {
      try { await this.api("/api/v1/auth/logout", { method: "POST" }); } catch (e) { /* cookies may already be gone */ }
      this.clearSession();
    },
    views() {
      if (!this.user) return [];
      const r = this.user.role;
      const v = [];
      if (r === "basvuran") v.push({ id: "portal", label: "Başvurularım" });
      if (["uzman", "kidemli_uzman", "komite", "admin"].includes(r)) v.push({ id: "workbench", label: "Çalışma masası" });
      if (r !== "basvuran") v.push({ id: "dashboard", label: "Yönetim panosu" });
      return v;
    },
    async afterLogin() {
      if (!Object.keys(this.labels).length) {
        try { this.labels = await this.api("/api/v1/labels"); } catch (e) { this.labels = {}; }
      }
      this.api("/health").then((h) => { this.health = h; }).catch(() => {});
      this.view = this.views()[0].id;
      this.api("/api/v1/llm/status").then((s) => { this.llm = s; }).catch(() => {});
      await this.go(this.view);
    },
    async go(view) {
      this.view = view;
      try {
        if (view === "portal") await this.loadApplications();
        if (view === "workbench") { await this.loadQueue(); await this.loadApplications(); }
        if (view === "dashboard") await this.loadDashboard();
      } catch (e) { this.notify(e.message); }
    },

    // ------------------------------------------------------------ applicant
    async loadApplications() { this.applications = (await this.api("/api/v1/applications?limit=100")).applications; },
    startWizard() { this.form = EMPTY_FORM(); this.wizard = { open: true, step: 1 }; this.current = null; this.error = ""; },
    wizardBack() { if (this.wizard.step > 1) this.wizard.step--; else this.wizard.open = false; },
    wizardNext() { this.wizard.step++; if (this.wizard.step === 3) this.loadQuote(); },
    onDrop(event) { this.upload.over = false; this.uploadFile(event.dataTransfer.files[0]); },
    pickDemo(key) { if (key) { this.form.identity_no = this.demoTckn[key]; } },
    async loadQuote() {
      try { this.quote = await this.api("/api/v1/pricing/quote", { method: "POST", json: { amount: this.form.requested_amount, term_months: this.form.requested_term_months, product: this.form.product, risk_band: "B" } }); } catch (e) { this.quote = null; }
    },
    async submitApplication() {
      this.error = ""; this.busy = true;
      try {
        const r = await this.api("/api/v1/applications", { method: "POST", json: this.form });
        this.wizard.open = false; this.notify("Başvurunuz alındı: " + r.application_id);
        await this.loadApplications(); await this.openApp(r.application_id);
      } catch (e) { this.error = e.message; } finally { this.busy = false; }
    },
    async openApp(id) {
      this.current = await this.api("/api/v1/applications/" + id);
      if (this.current.missing_documents.length) this.upload.code = this.current.missing_documents[0];
    },
    async uploadFile(file) {
      if (!file) return;
      const body = new FormData(); body.append("code", this.upload.code); body.append("file", file);
      try {
        const r = await this.api("/api/v1/applications/" + this.current.application_id + "/documents", { method: "POST", body });
        this.notify(r.complete ? "Belgeler tamamlandı, değerlendirme başladı." : "Belge yüklendi. Eksik: " + this.lblList("document", r.missing_documents));
        setTimeout(() => this.openApp(this.current.application_id), r.processing_resumed ? 1500 : 10);
      } catch (e) { this.notify(e.message); }
    },
    async act(path) {
      try { await this.api("/api/v1/applications/" + this.current.application_id + "/" + path, { method: "POST" }); await this.openApp(this.current.application_id); this.notify("İşlem tamamlandı."); } catch (e) { this.notify(e.message); }
    },
    async submitObjection() {
      try { await this.api("/api/v1/applications/" + this.current.application_id + "/objection", { method: "POST", json: { reason: this.objection } }); this.objection = ""; await this.openApp(this.current.application_id); this.notify("İtirazınız alındı; bir uzman inceleyecek."); } catch (e) { this.notify(e.message); }
    },

    // ------------------------------------------------------------ workbench
    queueQuery() {
      const f = this.queueFilter;
      const params = new URLSearchParams({ limit: String(f.limit), offset: String(f.offset) });
      const q = (f.q || "").trim();
      if (q) params.set("q", q.slice(0, 64));
      if (f.state) params.set("state", f.state);
      if (f.product) params.set("product", f.product);
      if (f.sla) params.set("sla_breached", f.sla);
      if (f.mine) params.set("mine", "true");
      return params.toString();
    },
    async loadQueue() {
      const seq = ++this.queueSeq;
      this.queueLoading = true;
      try {
        const [page, reviews] = await Promise.all([
          this.api("/api/v1/workbench/queue?" + this.queueQuery()),
          this.api("/api/v1/workbench/reviews"),
        ]);
        if (seq !== this.queueSeq) return; // a newer filter/page request superseded this one
        if (page.total > 0 && page.offset >= page.total) { this.queueFilter.offset = Math.max(0, page.total - this.queueFilter.limit); return this.loadQueue(); }
        this.queue = page.items;
        this.queuePage = { total: page.total, limit: page.limit, offset: page.offset, sla_breached: page.sla_breached || 0 };
        this.reviews = reviews.reviews;
      } finally { if (seq === this.queueSeq) this.queueLoading = false; }
    },
    applyFilters() { this.queueFilter.offset = 0; return this.loadQueue().catch((e) => this.notify(e.message)); },
    onSearchInput() {
      clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => this.applyFilters(), 300);
    },
    pageQueue(direction) {
      const f = this.queueFilter;
      f.offset = Math.max(0, f.offset + direction * f.limit);
      return this.loadQueue().catch((e) => this.notify(e.message));
    },
    hasPrevPage() { return this.queuePage.offset > 0; },
    hasNextPage() { return this.queuePage.offset + this.queue.length < this.queuePage.total; },
    pageRange() {
      const p = this.queuePage;
      if (!p.total) return "0 / 0";
      return (p.offset + 1) + "–" + (p.offset + this.queue.length) + " / " + p.total;
    },
    queueStatus() { return this.queueLoading ? "Kuyruk yükleniyor" : "Kuyruk: " + this.pageRange() + " başvuru gösteriliyor"; },
    isSelected(id) { return !!(this.staff && this.staff.detail.application_id === id); },
    slaClass(q) { return q.sla_breached ? "bad" : (q.sla_remaining_hours !== null && q.sla_remaining_hours < 4 ? "warn" : "ok"); },
    slaText(q) {
      if (q.sla_remaining_hours === null || q.sla_remaining_hours === undefined) return "—";
      return q.sla_breached ? "aşıldı (" + Math.abs(q.sla_remaining_hours) + " sa)" : q.sla_remaining_hours + " sa";
    },
    focusRow(event, direction) {
      const buttons = Array.from(event.target.closest("tbody").querySelectorAll("button.queue-open"));
      const next = buttons[buttons.indexOf(event.target) + direction];
      if (next) next.focus();
    },
    tabKey(direction) {
      const ids = this.staffTabs.map((t) => t[0]);
      const next = ids[(ids.indexOf(this.staff.tab) + direction + ids.length) % ids.length];
      this.staffTab(next);
      this.$nextTick(() => { const el = document.getElementById("tab-" + next); if (el) el.focus(); });
    },
    shapRows() { return ((this.staff && this.staff.decision && this.staff.decision.shap) || []).slice(0, 10).map((x) => ({ label: x.label, value: Number(x.value).toFixed(3) })); },
    async openStaff(id) {
      try { await this.loadStaff(id); } catch (e) { this.notify(e.message); }
    },
    async loadStaff(id) {
      const base = "/api/v1/applications/" + id;
      const [detail, docs, cash, net, authority] = await Promise.all([
        this.api(base), this.api(base + "/documents"), this.api(base + "/cashflow"), this.api(base + "/network"),
        this.api("/api/v1/workbench/" + id + "/authority").catch(() => null)]);
      const memo = (detail.letters && detail.letters.ai_memo) || null; // stored by POST /agent/{id}/memo
      const tab = this.staff && this.staff.detail.application_id === id ? this.staff.tab : "ozet";
      this.staff = { detail, decision: detail.decision, documents: docs.documents, cashflow: cash, network: net, memo, authority, tab, replay: "" };
      this.fieldEdit = null;
      if (tab !== "ozet") this.staffTab(tab);
      this.decisionForm = { action: "ONAY", justification: "", amount: null, term_months: null };
    },
    staffTab(tab) {
      this.staff.tab = tab;
      this.$nextTick(() => {
        if (tab === "nakit") this.drawCash();
        if (tab === "karar") this.drawShap();
        if (tab === "halka") this.drawRing();
      });
    },
    chart(id, config) {
      if (typeof Chart === "undefined") return;
      if (CHARTS.has(id)) { CHARTS.get(id).destroy(); CHARTS.delete(id); }
      const el = document.getElementById(id); if (!el) return;
      CHARTS.set(id, new Chart(el, Object.assign({ options: { responsive: true, maintainAspectRatio: false } }, config)));
    },
    drawCash() {
      const m = (this.staff.cashflow && this.staff.cashflow.monthly) || [];
      this.chart("cashChart", { type: "bar", data: { labels: m.map((x) => x.month), datasets: [
        { label: "Gelir", data: m.map((x) => x.income), backgroundColor: "#2f6690" },
        { label: "Harcama", data: m.map((x) => x.spending), backgroundColor: "#b3261e" },
        { label: "Ay sonu bakiye", data: m.map((x) => x.end_balance), type: "line", borderColor: "#1e7b34" }] } });
    },
    drawShap() {
      const s = (this.staff.decision && this.staff.decision.shap || []).slice(0, 10);
      this.chart("shapChart", { type: "bar", data: { labels: s.map((x) => x.label), datasets: [{ label: "Katkı", data: s.map((x) => x.value), backgroundColor: s.map((x) => (x.value > 0 ? "#b3261e" : "#1e7b34")) }] }, options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } } });
    },
    drawRing() {
      const svg = document.getElementById("ringSvg"); if (!svg) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const nodes = (this.staff.network && this.staff.network.nodes) || [];
      const edges = (this.staff.network && this.staff.network.edges) || [];
      const ns = "http://www.w3.org/2000/svg", pos = {};
      nodes.forEach((n, i) => { const a = (2 * Math.PI * i) / Math.max(nodes.length, 1); pos[n.id] = [300 + 120 * Math.cos(a), 160 + 120 * Math.sin(a)]; });
      edges.forEach((e) => { const l = document.createElementNS(ns, "line"); const [x1, y1] = pos[e.source]; const [x2, y2] = pos[e.target]; l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2); l.setAttribute("stroke", "#9aa5b1"); svg.appendChild(l); });
      nodes.forEach((n) => { const [x, y] = pos[n.id]; const c = document.createElementNS(ns, "circle"); c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", n.kind === "application" ? 14 : 9); c.setAttribute("fill", n.kind === "application" ? "#2f6690" : "#9a6700"); svg.appendChild(c);
        const t = document.createElementNS(ns, "text"); t.setAttribute("x", x + 16); t.setAttribute("y", y + 4); t.textContent = this.lbl("node_kind", n.kind) + ": " + n.label; svg.appendChild(t); });
      if (!nodes.length) { const t = document.createElementNS(ns, "text"); t.setAttribute("x", 200); t.setAttribute("y", 160); t.textContent = "Ortak tanımlayıcı bulunmadı"; svg.appendChild(t); }
    },
    async assign() { try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/assign", { method: "POST" }); this.notify("Başvuru üzerinize atandı."); await this.loadQueue(); } catch (e) { this.notify(e.message); } },
    async replay() {
      try { const r = await this.api("/api/v1/decisions/" + this.staff.decision.decision_id + "/replay", { method: "POST" }); this.staff.replay = r.identical ? "✓ aynı sonuç" : "✗ farklı sonuç"; } catch (e) { this.notify(e.message); }
    },
    correctField(f) {
      this.fieldEdit = { field_id: f.field_id, name: f.name, value: f.value || "" };
      this.$nextTick(() => { const el = document.getElementById("field-value"); if (el) el.focus(); });
    },
    async saveField() {
      const edit = this.fieldEdit;
      if (!edit || !String(edit.value).trim()) return;
      try {
        await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/fields/" + edit.field_id, { method: "POST", json: { value: String(edit.value), note: "çalışma masası" } });
        this.fieldEdit = null; await this.openStaff(this.staff.detail.application_id); this.notify("Alan düzeltildi.");
      } catch (e) { this.notify(e.message); }
    },
    async generateMemo() {
      this.busy = true;
      try { this.staff.memo = await this.api("/api/v1/agent/" + this.staff.detail.application_id + "/memo", { method: "POST" }); } catch (e) { this.notify(e.message); } finally { this.busy = false; }
    },
    async askPolicy() { try { this.policyA = await this.api("/api/v1/policy/ask", { method: "POST", json: { question: this.policyQ } }); } catch (e) { this.notify(e.message); } },
    async submitDecision() {
      const body = { action: this.decisionForm.action, justification: this.decisionForm.justification };
      if (this.decisionForm.amount) body.amount = this.decisionForm.amount;
      if (this.decisionForm.term_months) body.term_months = this.decisionForm.term_months;
      try {
        const r = await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/decision", { method: "POST", json: body });
        this.notify(r.review.status === "ONAY_BEKLIYOR" ? "Dört göz onayı bekleniyor." : "Karar kaydedildi: " + this.lbl("state", r.state));
        await this.loadQueue(); await this.openStaff(this.staff.detail.application_id);
      } catch (e) { this.notify(e.message); }
    },
    async checkReview(r, approve) {
      try { const res = await this.api("/api/v1/workbench/reviews/" + r.review_id + "/check", { method: "POST", json: { approve, note: approve ? "Uygundur." : "Uygun değildir." } }); this.notify("Durum: " + this.lbl("state", res.state)); await this.loadQueue(); if (this.staff && this.staff.detail.application_id === r.application_id) await this.loadStaff(r.application_id); } catch (e) { this.notify(e.message); }
    },
    async resolveObjection(upheld) {
      try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/objection/resolve", { method: "POST", json: { upheld, note: this.decisionForm.justification || "İnceleme tamamlandı, gerekçe dosyada." } }); await this.loadQueue(); await this.openStaff(this.staff.detail.application_id); } catch (e) { this.notify(e.message); }
    },
    async disburse() { try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/disburse", { method: "POST" }); await this.openStaff(this.staff.detail.application_id); this.notify("Kredi kullandırıldı."); } catch (e) { this.notify(e.message); } },

    // ------------------------------------------------------------ dashboard
    // ------------------------------------------------------------ number formatting
    num(v, digits) { return v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits }); },
    ci(pair, digits) { return Array.isArray(pair) && pair.length === 2 ? "[" + this.num(pair[0], digits) + " – " + this.num(pair[1], digits) + "]" : "—"; },
    pval(p) { if (p === null || p === undefined) return "—"; return p < 0.001 ? "< 0,001" : this.num(p, 3); },
    pairs(obj) { return Object.entries(obj || {}); },

    // ------------------------------------------------------------ model validation (lane A, real public data)
    canValidate() { return !!this.user && (this.user.role === "model_yoneticisi" || this.user.role === "admin"); },
    setDashTab(tab, focus) {
      this.dashTab = tab;
      if (tab === "dogrulama" && !this.validation) this.loadValidation();
      else if (tab === "dogrulama") this.$nextTick(() => this.drawCalibration());
      if (focus) this.$nextTick(() => { const el = document.getElementById("dtab-" + tab); if (el) el.focus(); });
    },
    async loadValidation() {
      this.validationError = "";
      try {
        const v = await this.api("/api/v1/models/validation");
        const names = Object.keys(v.sets || {}).filter((n) => v.sets[n]);
        if (!names.length) { this.validationError = "Doğrulama çıktısı bulunamadı (scripts/run_validation.py çalıştırılmalı)."; return; }
        this.validation = v;
        if (!names.includes(this.valSet)) this.valSet = names.includes("uci_taiwan") ? "uci_taiwan" : names[0];
        this.valCalModel = this.vs().champion.model;
        this.$nextTick(() => this.drawCalibration());
      } catch (e) { this.validationError = e.message; }
    },
    valSetNames() { return this.validation ? Object.keys(this.validation.sets).filter((n) => this.validation.sets[n]) : []; },
    vs() { return this.validation && this.valSet ? this.validation.sets[this.valSet] : null; },
    onValSetChange() { this.valCalModel = this.vs().champion.model; this.$nextTick(() => this.drawCalibration()); },
    modelLabel(key) {
      const m = this.vs();
      if (key === "lightgbm_uncalibrated") return UI_LABELS.model_family.lightgbm_uncalibrated;
      return (m && m.models && m.models[key]) || this.lbl("model_family", key);
    },
    valDesign() {
      const d = this.vs().design || {};
      return this.num(d.rows, 0) + " kayıt · hold-out " + this.num(d.holdout_rows, 0) + " · temerrüt oranı " + this.pct(d.default_rate) + " · " + (d.cv_folds || "?") + " katlı CV · " + this.date(this.vs().generated_at);
    },
    holdoutRows() {
      const m = this.vs();
      return Object.keys(m.holdout).map((key) => {
        const h = m.holdout[key], cv = (m.cv || {})[key];
        return {
          key, label: this.modelLabel(key), champion: key === m.champion.model,
          auc: this.num(h.auc, 4), ci: this.ci(h.auc_ci, 4), ciDelong: this.ci(h.auc_ci_delong, 4),
          gini: this.num(h.gini, 4), ks: this.num(h.ks, 4), brier: this.num(h.brier, 4), logloss: this.num(h.log_loss, 4),
          cv: cv ? this.num(cv.mean_auc, 4) + " ± " + this.num(cv.std_auc, 4) : "—",
        };
      });
    },
    delongRows() {
      const d = this.vs().delong || {};
      return Object.keys(d).map((key) => {
        const [a, b] = key.split("_vs_");
        const r = d[key];
        return { key, label: this.modelLabel(a) + " − " + this.modelLabel(b), diff: this.num(r.auc_diff, 4), ci: this.ci(r.diff_ci, 4), z: this.num(r.z, 2), p: this.pval(r.p_value) };
      });
    },
    championOrder() { return (this.vs().champion.simplicity_order || []).map((k) => this.modelLabel(k)).join(" → "); },
    calModels() { return Object.keys(this.vs().calibration || {}); },
    calRows() {
      const c = (this.vs().calibration || {})[this.valCalModel];
      if (!c) return [];
      const low = (c.low_risk && c.low_risk.deciles) || 0;
      return c.deciles.map((d) => ({
        decile: d.decile, n: this.num(d.n, 0), range: this.num(d.pd_min, 3) + " – " + this.num(d.pd_max, 3),
        predicted: this.num(d.predicted, 4), observed: this.num(d.observed, 4), ratio: this.num(d.ratio_observed_to_predicted, 2),
        lowRisk: d.decile <= low,
      }));
    },
    calSummary() {
      const c = (this.vs().calibration || {})[this.valCalModel];
      if (!c) return "";
      const hl = c.hosmer_lemeshow || {}, lr = c.low_risk || {};
      return "ECE " + this.num(c.ece, 4) + " · Hosmer-Lemeshow χ²=" + this.num(hl.statistic, 2) + " (sd " + hl.dof + ", p " + this.pval(hl.p_value) + ")"
        + " · düşük riskli ilk " + (lr.deciles || 0) + " dilim: tahmini " + this.pct(lr.predicted) + ", gözlenen " + this.pct(lr.observed) + " (oran " + this.num(lr.ratio_observed_to_predicted, 2) + ")";
    },
    drawCalibration() {
      if (!this.vs() || this.dashTab !== "dogrulama") return;
      const c = (this.vs().calibration || {})[this.valCalModel];
      if (!c) return;
      const labels = c.deciles.map((d) => String(d.decile));
      const low = (c.low_risk && c.low_risk.deciles) || 0;
      this.chart("calChart", { type: "line", data: { labels, datasets: [
        { label: "Tahmini PD", data: c.deciles.map((d) => d.predicted), borderColor: "#2f6690", backgroundColor: "#2f6690", pointStyle: "circle" },
        { label: "Gözlenen temerrüt", data: c.deciles.map((d) => d.observed), borderColor: "#b3261e", backgroundColor: c.deciles.map((d) => (d.decile <= low ? "#9a6700" : "#b3261e")), pointStyle: "rectRot", pointRadius: 5, borderDash: [5, 3] }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: "Ondalık dilim (1 = en düşük risk)" } }, y: { title: { display: true, text: "Temerrüt oranı" } } } } });
    },
    fairnessTables() {
      const f = this.vs().fairness || {}, byModel = f.by_model || {};
      const models = Object.keys(byModel);
      if (!models.length) return [];
      return Object.keys(byModel[models[0]]).map((attr) => {
        const groups = Object.keys(byModel[models[0]][attr].selection_rate || {});
        return {
          attr, groups, title: this.lbl("fairness_attr", attr),
          rows: models.map((m) => {
            const a = byModel[m][attr];
            return { model: m, label: this.modelLabel(m), cells: groups.map((g) => this.pct(a.selection_rate[g])), minAir: this.num(a.min_air, 3), pass: a.passes_four_fifths, tpr: this.num(a.tpr_gap, 3), fpr: this.num(a.fpr_gap, 3) };
          }),
        };
      });
    },
    ldaRows() {
      const l = this.vs().lda || {};
      const row = (r, i, reference) => ({ key: (reference ? "ref-" : "") + i, model: r.model, reference, caveat: !!r.legal_caveat, auc: this.num(r.auc, 4), approval: this.pct(r.approval_rate), badRate: this.pct(r.bad_rate_approved), minAir: this.num(r.min_air, 3), tpr: this.num(r.tpr_gap, 3), fpr: this.num(r.fpr_gap, 3), pass: r.passes_four_fifths });
      return (l.rows || []).map((r, i) => row(r, i, false)).concat((l.reference_rows || []).map((r, i) => row(r, i, true)));
    },
    ldaCaveat() {
      const l = this.vs().lda || {};
      const hit = (l.rows || []).concat(l.reference_rows || []).find((r) => r.legal_caveat);
      return hit ? hit.legal_caveat : "";
    },
    proxyList() { return Object.entries((this.vs().lda || {}).proxy_strength || {}).slice(0, 5).map((e) => e[0] + " (" + this.num(e[1], 2) + ")").join(", ") || "—"; },
    laneBRows() {
      const b = this.validation && this.validation.lane_b;
      if (!b) return [];
      return Object.entries(b).filter((e) => e[1] === null || typeof e[1] !== "object").map((e) => [e[0], String(e[1])]);
    },
    ccEvidence() {
      const v = this.dash.cc && this.dash.cc.validation;
      if (!v || !v.available) return null;
      const row = (role, x) => ({ role, family: this.lbl("model_family", x.family), auc: this.num(x.auc, 4), ci: this.ci(x.auc_ci, 4), brier: this.num(x.brier, 4), ece: this.num(x.ece, 4) });
      const diff = v.auc_diff_challenger_minus_champion;
      return {
        rows: [row("Şampiyon", v.champion), row("Challenger", v.challenger)],
        summary: this.lbl("dataset", v.dataset) + " · ΔAUC (challenger − şampiyon) " + this.num(diff, 4) + " · DeLong p " + this.pval(v.delong_p_value)
          + " · gerçek veride seçilen şampiyon: " + this.lbl("model_family", v.lane_a_champion),
      };
    },

    async loadDashboard() {
      const safe = (p) => this.api(p).catch(() => null);
      const [metrics, fairness, drift, cc, models, watchlist, llm] = await Promise.all([
        safe("/api/v1/metrics"), safe("/api/v1/governance/fairness"), safe("/api/v1/governance/drift"),
        safe("/api/v1/governance/champion-challenger"), safe("/api/v1/models"), safe("/api/v1/portfolio/watchlist"), safe("/api/v1/llm/status")]);
      this.dash = { metrics, fairness, drift, cc, models: models ? models.models : [], watchlist };
      if (llm) this.llm = llm;
      if (this.dashTab === "dogrulama") this.loadValidation();
      this.$nextTick(() => {
        if (!metrics) return;
        const s = metrics.by_state_labels || {};
        this.chart("stateChart", { type: "doughnut", data: { labels: Object.keys(s), datasets: [{ data: Object.values(s) }] } });
        const d = metrics.pd_distribution || {};
        this.chart("pdChart", { type: "bar", data: { labels: Object.keys(d), datasets: [{ label: "Karar sayısı", data: Object.values(d), backgroundColor: "#2f6690" }] } });
      });
    },
    async seedDemo() {
      this.busy = true; this.seedResult = "Demo senaryoları yükleniyor…";
      try { const r = await this.api("/api/v1/demo/seed", { method: "POST" }); this.seedResult = r.results.map((x) => x.scenario + " → " + this.lbl("state", x.state)).join(" · "); await this.loadDashboard(); } catch (e) { this.seedResult = e.message; } finally { this.busy = false; }
    },
    async promote(id) {
      this.promoteError = "";
      try {
        const r = await this.api("/api/v1/models/" + id + "/promote", { method: "POST" });
        this.notify("Onay kaydedildi: " + r.approvals.length + "/2"); await this.loadDashboard();
      } catch (e) { this.promoteError = "Terfi reddedildi: " + e.message; }
    },
  };
}
