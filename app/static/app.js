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

document.addEventListener("alpine:init", () => { window.Alpine.data("platform", platform); });

function platform() {
  return {
    user: null, token: null, view: "portal", theme: "light", llm: null, error: "", toast: "", busy: false,
    loginForm: { username: "", password: "" },
    demoUsers: [["basvuran", "Başvuran"], ["uzman", "Uzman"], ["kidemli", "Kıdemli uzman"], ["komite", "Komite"], ["modelyon", "Model yöneticisi"], ["admin", "Admin"]],
    demoTckn: { temiz: "68846908942", ince_dosya: "29551411288", gri: "24377158546", gecikmeli: "21134086364" },
    applications: [], current: null, wizard: { open: false, step: 1 }, form: EMPTY_FORM(), quote: null,
    upload: { code: "IDENTITY", over: false }, objection: "",
    queue: [], reviews: [], staff: null, decisionForm: { action: "ONAY", justification: "", amount: null, term_months: null },
    policyQ: "Borç servis oranı sınırı nedir?", policyA: null, dash: {}, seedResult: "", charts: {},
    staffTabs: [["ozet", "Özet"], ["belgeler", "Belgeler"], ["kkb", "KKB"], ["nakit", "Nakit akışı"], ["karar", "Karar + SHAP"], ["fiyat", "Fiyatlama"], ["halka", "Halka"], ["memo", "AI memorandum"], ["politika", "Politika sor"], ["karar_ver", "Karar ver"]],
    bureauKeys: [["bureau_score", "KKB notu"], ["bureau_hit", "KKB kaydı"], ["active_loans", "Aktif kredi"], ["bureau_utilisation", "Limit kullanımı"], ["delinquency_count_24m", "24 ay gecikme"], ["max_dpd_24m", "Azami gecikme günü"], ["inquiries_6m", "6 ay sorgu"], ["existing_debt_service", "Mevcut aylık taksit"], ["existing_dsr", "Mevcut DSR"], ["dsr", "DSR (talep)"], ["employment_months", "Çalışma süresi (ay)"]],

    // ------------------------------------------------------------ helpers
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
    async api(path, opts = {}) {
      const headers = Object.assign({}, opts.headers || {});
      if (this.token) headers.Authorization = "Bearer " + this.token;
      if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
      const res = await fetch(path, { method: opts.method || "GET", headers, body: opts.body });
      if (res.status === 401 && this.user) { this.logout(); throw new Error("Oturum süresi doldu"); }
      const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
      if (!res.ok) {
        const detail = data && data.detail;
        throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : (detail || res.statusText));
      }
      return data;
    },
    async download(path, name) {
      try {
        const res = await fetch(path, { headers: { Authorization: "Bearer " + this.token } });
        if (!res.ok) throw new Error("dosya alınamadı");
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement("a"); a.href = url; a.download = name; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      } catch (e) { this.notify(e.message); }
    },

    // ------------------------------------------------------------ session
    async init() {
      try { this.theme = localStorage.getItem("anil2-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (e) { /* ignore */ }
      try {
        const saved = sessionStorage.getItem("anil2-session");
        if (saved) { const s = JSON.parse(saved); this.token = s.token; this.user = s.user; await this.afterLogin(); }
      } catch (e) { this.logout(); }
    },
    toggleTheme() { this.theme = this.theme === "dark" ? "light" : "dark"; try { localStorage.setItem("anil2-theme", this.theme); } catch (e) { /* ignore */ } },
    async login() {
      this.error = "";
      try {
        const r = await this.api("/api/v1/auth/login", { method: "POST", json: this.loginForm });
        this.token = r.access_token; this.user = { full_name: r.full_name, role: r.role, role_label: r.role_label, username: r.username };
        sessionStorage.setItem("anil2-session", JSON.stringify({ token: this.token, user: this.user }));
        await this.afterLogin();
      } catch (e) { this.error = e.message; }
    },
    demoLogin(username) { this.loginForm = { username, password: "Demo123!" }; return this.login(); },
    logout() { this.user = null; this.token = null; this.current = null; this.staff = null; sessionStorage.removeItem("anil2-session"); },
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
        this.notify(r.complete ? "Belgeler tamamlandı, değerlendirme başladı." : "Belge yüklendi. Eksik: " + r.missing_documents.join(", "));
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
    async loadQueue() {
      this.queue = (await this.api("/api/v1/workbench/queue")).items;
      this.reviews = (await this.api("/api/v1/workbench/reviews")).reviews;
    },
    async openStaff(id) {
      const base = "/api/v1/applications/" + id;
      const detail = await this.api(base);
      const [docs, cash, net] = await Promise.all([this.api(base + "/documents"), this.api(base + "/cashflow"), this.api(base + "/network")]);
      let memo = null; try { memo = await this.api("/api/v1/agent/" + id + "/memo"); } catch (e) { memo = null; }
      let authority = null; try { authority = await this.api("/api/v1/workbench/" + id + "/authority"); } catch (e) { authority = null; }
      this.staff = { detail, decision: detail.decision, documents: docs.documents, cashflow: cash, network: net, memo, authority, tab: "ozet", replay: "" };
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
      if (this.charts[id]) this.charts[id].destroy();
      const el = document.getElementById(id); if (!el) return;
      this.charts[id] = new Chart(el, Object.assign({ options: { responsive: true, maintainAspectRatio: false } }, config));
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
        const t = document.createElementNS(ns, "text"); t.setAttribute("x", x + 16); t.setAttribute("y", y + 4); t.textContent = n.kind + ": " + n.label; svg.appendChild(t); });
      if (!nodes.length) { const t = document.createElementNS(ns, "text"); t.setAttribute("x", 200); t.setAttribute("y", 160); t.textContent = "Ortak tanımlayıcı bulunmadı"; svg.appendChild(t); }
    },
    async assign() { try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/assign", { method: "POST" }); this.notify("Başvuru üzerinize atandı."); await this.loadQueue(); } catch (e) { this.notify(e.message); } },
    async replay() {
      try { const r = await this.api("/api/v1/decisions/" + this.staff.decision.decision_id + "/replay", { method: "POST" }); this.staff.replay = r.identical ? "✓ aynı sonuç" : "✗ farklı sonuç"; } catch (e) { this.notify(e.message); }
    },
    async correctField(f) {
      const value = window.prompt ? window.prompt("Yeni değer", f.value) : null;
      if (!value) return;
      try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/fields/" + f.field_id, { method: "POST", json: { value, note: "çalışma masası" } }); await this.openStaff(this.staff.detail.application_id); this.staff.tab = "belgeler"; } catch (e) { this.notify(e.message); }
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
        this.notify(r.review.status === "ONAY_BEKLIYOR" ? "Dört göz onayı bekleniyor." : "Karar kaydedildi: " + r.state);
        await this.loadQueue(); await this.openStaff(this.staff.detail.application_id);
      } catch (e) { this.notify(e.message); }
    },
    async checkReview(r, approve) {
      try { const res = await this.api("/api/v1/workbench/reviews/" + r.review_id + "/check", { method: "POST", json: { approve, note: approve ? "Uygundur." : "Uygun değildir." } }); this.notify("Durum: " + res.state); await this.loadQueue(); } catch (e) { this.notify(e.message); }
    },
    async resolveObjection(upheld) {
      try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/objection/resolve", { method: "POST", json: { upheld, note: this.decisionForm.justification || "İnceleme tamamlandı, gerekçe dosyada." } }); await this.loadQueue(); await this.openStaff(this.staff.detail.application_id); } catch (e) { this.notify(e.message); }
    },
    async disburse() { try { await this.api("/api/v1/workbench/" + this.staff.detail.application_id + "/disburse", { method: "POST" }); await this.openStaff(this.staff.detail.application_id); this.notify("Kredi kullandırıldı."); } catch (e) { this.notify(e.message); } },

    // ------------------------------------------------------------ dashboard
    async loadDashboard() {
      const safe = (p) => this.api(p).catch(() => null);
      const [metrics, fairness, drift, cc, models, watchlist, llm] = await Promise.all([
        safe("/api/v1/metrics"), safe("/api/v1/governance/fairness"), safe("/api/v1/governance/drift"),
        safe("/api/v1/governance/champion-challenger"), safe("/api/v1/models"), safe("/api/v1/portfolio/watchlist"), safe("/api/v1/llm/status")]);
      this.dash = { metrics, fairness, drift, cc, models: models ? models.models : [], watchlist };
      if (llm) this.llm = llm;
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
      try { const r = await this.api("/api/v1/demo/seed", { method: "POST" }); this.seedResult = r.results.map((x) => x.scenario + "→" + x.state).join(" · "); await this.loadDashboard(); } catch (e) { this.seedResult = e.message; } finally { this.busy = false; }
    },
    async promote(id) { try { const r = await this.api("/api/v1/models/" + id + "/promote", { method: "POST" }); this.notify("Onay kaydedildi: " + r.approvals.length + "/2"); await this.loadDashboard(); } catch (e) { this.notify(e.message); } },
  };
}
