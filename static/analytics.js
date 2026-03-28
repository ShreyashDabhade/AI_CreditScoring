(function () {
    const analytics = window.__MASTERMIND_ANALYTICS__ || {};
    const plots = analytics.plots || {};

    function makeAlert(message, level) {
        const alert = document.createElement("div");
        alert.className = `alert alert-${level}`;
        alert.textContent = message;
        return alert;
    }

    function makePlotCard(entry) {
        const col = document.createElement("div");
        col.className = "col-md-6 mb-4";

        const card = document.createElement("div");
        card.className = "card";

        const img = document.createElement("img");
        img.src = entry.url;
        img.className = "card-img-top";
        img.alt = entry.title;
        img.loading = "lazy";
        img.onerror = function () {
            this.style.display = "none";
        };

        const body = document.createElement("div");
        body.className = "card-body";

        const title = document.createElement("h6");
        title.className = "card-title";
        title.textContent = entry.title;

        body.appendChild(title);
        card.appendChild(img);
        card.appendChild(body);
        col.appendChild(card);
        return col;
    }

    function renderValueCell(value) {
        const td = document.createElement("td");
        if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) {
            td.innerHTML = "&mdash;";
            return td;
        }
        if (typeof value === "boolean") {
            const badge = document.createElement("span");
            badge.className = `badge ${value ? "bg-success" : "bg-danger"}`;
            badge.textContent = value ? "PASS" : "FAIL";
            td.appendChild(badge);
            return td;
        }
        td.textContent = String(value);
        return td;
    }

    function makeTable(columns, rows) {
        const table = document.createElement("table");
        table.className = "table table-sm table-striped table-bordered";

        const thead = document.createElement("thead");
        const headRow = document.createElement("tr");
        columns.forEach((column) => {
            const th = document.createElement("th");
            th.textContent = column;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);

        const tbody = document.createElement("tbody");
        rows.forEach((row) => {
            const tr = document.createElement("tr");
            columns.forEach((column) => {
                tr.appendChild(renderValueCell(row[column]));
            });
            tbody.appendChild(tr);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        return table;
    }

    function makePlotGrid(entries) {
        const row = document.createElement("div");
        row.className = "row";
        entries.forEach((entry) => {
            row.appendChild(makePlotCard(entry));
        });
        return row;
    }

    function makeSplitSizeTable(splitSizes) {
        const rows = Object.entries(splitSizes || {}).map(([key, value]) => ({
            Metric: key,
            Value: value,
        }));
        return makeTable(["Metric", "Value"], rows);
    }

    function makeStatusBadge(label, level) {
        const badge = document.createElement("span");
        badge.className = `badge ${
            level === "success"
                ? "bg-success"
                : level === "danger"
                    ? "bg-danger"
                    : level === "warning"
                        ? "bg-warning text-dark"
                        : "bg-secondary"
        }`;
        badge.textContent = label;
        return badge;
    }

    function comparisonBadge(payload) {
        if (payload.status === "stable") {
            return makeStatusBadge("STABLE", "success");
        }
        if (payload.status === "watch") {
            return makeStatusBadge("WATCH", "warning");
        }
        if (payload.available === false) {
            return makeStatusBadge("UNAVAILABLE", "secondary");
        }
        return makeStatusBadge("EXPERIMENTAL", "warning");
    }

    function appendBadgeRow(container, badges) {
        const validBadges = badges.filter(Boolean);
        if (validBadges.length === 0) {
            return;
        }
        const row = document.createElement("div");
        row.className = "d-flex flex-wrap gap-2 mb-3";
        validBadges.forEach((badge) => row.appendChild(badge));
        container.appendChild(row);
    }

    function fairnessLevelFromFamily(family) {
        if (!family || family.family_pass === null || family.family_pass === undefined) {
            return "secondary";
        }
        return family.family_pass ? "success" : "warning";
    }

    function makeFairnessFamilyTable(families) {
        const rows = (families || []).map((family) => ({
            Family: family.label || family.family || "Unknown",
            "DI Min": family.di_min_text || "—",
            "EOD Gap": family.eod_gap_text || "—",
            "Brier Ratio": family.brier_ratio_text || "—",
            DI: family.di_pass,
            EOD: family.eod_pass,
            Brier: family.brier_pass,
        }));
        return makeTable(["Family", "DI Min", "EOD Gap", "Brier Ratio", "DI", "EOD", "Brier"], rows);
    }

    function makeComparisonCard(payload, emptyMessage) {
        const col = document.createElement("div");
        col.className = "col-lg-6 mb-4";

        const card = document.createElement("div");
        card.className = "card h-100";

        const body = document.createElement("div");
        body.className = "card-body";

        const titleRow = document.createElement("div");
        titleRow.className = "d-flex justify-content-between align-items-start gap-3 flex-wrap";

        const titleWrap = document.createElement("div");
        const title = document.createElement("h5");
        title.className = "card-title mb-1";
        title.textContent = payload.label || "Comparison";
        titleWrap.appendChild(title);

        if (payload.subtitle) {
            const subtitle = document.createElement("p");
            subtitle.className = "text-muted small mb-0";
            subtitle.textContent = payload.subtitle;
            titleWrap.appendChild(subtitle);
        }
        titleRow.appendChild(titleWrap);

        titleRow.appendChild(comparisonBadge(payload));
        body.appendChild(titleRow);

        if (payload.headline) {
            const headline = document.createElement("p");
            headline.className = "mt-3 mb-3";
            headline.textContent = payload.headline;
            body.appendChild(headline);
        }

        if (Array.isArray(payload.families) && payload.families.length > 0) {
            appendBadgeRow(
                body,
                payload.families.map((family) =>
                    makeStatusBadge(
                        `${family.label || family.family}: ${
                            family.family_pass === true ? "PASS" : family.family_pass === false ? "WATCH" : "N/A"
                        }`,
                        fairnessLevelFromFamily(family)
                    )
                )
            );
            body.appendChild(makeFairnessFamilyTable(payload.families));
        } else {
            const empty = document.createElement("div");
            empty.className = "alert alert-secondary mb-0";
            empty.textContent = payload.message || emptyMessage;
            body.appendChild(empty);
        }

        if (payload.source) {
            const source = document.createElement("p");
            source.className = "text-muted small mt-3 mb-0";
            source.textContent = `Source: ${payload.source}`;
            body.appendChild(source);
        }

        card.appendChild(body);
        col.appendChild(card);
        return col;
    }

    function makeFairnessComparisonView(comparison) {
        const wrap = document.createElement("div");

        const banner = document.createElement("div");
        banner.className = `alert ${comparison.mode === "live_toggle" ? "alert-success" : "alert-info"}`;
        banner.innerHTML = comparison.mode === "live_toggle"
            ? "<strong>Live comparison:</strong> A second deployed fairness-aware runtime was detected."
            : "<strong>Offline comparison:</strong> No second deployed fairness-aware runtime was detected, so this view compares the live audit against optional offline experiment artifacts only.";
        wrap.appendChild(banner);

        const explanation = document.createElement("p");
        explanation.className = "text-muted";
        explanation.textContent = comparison.tradeoff_explanation || "";
        wrap.appendChild(explanation);

        appendBadgeRow(wrap, [
            makeStatusBadge(comparison.mode_label || "Comparison", comparison.mode === "live_toggle" ? "success" : "warning"),
            makeStatusBadge("Transparency first", "secondary"),
        ]);

        const row = document.createElement("div");
        row.className = "row";
        row.appendChild(
            makeComparisonCard(
                comparison.standard || {},
                "Current deployed fairness audit summary is unavailable."
            )
        );
        row.appendChild(
            makeComparisonCard(
                comparison.experimental || {},
                "No fairness-optimized experimental metrics are available."
            )
        );
        wrap.appendChild(row);

        if (Array.isArray(comparison.experimental && comparison.experimental.tradeoff_notes)
            && comparison.experimental.tradeoff_notes.length > 0) {
            const tradeoffs = document.createElement("div");
            tradeoffs.className = "alert alert-secondary";
            tradeoffs.innerHTML = `<strong>Trade-offs:</strong><br>${comparison.experimental.tradeoff_notes.join("<br>")}`;
            wrap.appendChild(tradeoffs);
        }

        const artifacts = Array.isArray(comparison.artifact_files) ? comparison.artifact_files : [];
        if (artifacts.length > 0) {
            const footer = document.createElement("p");
            footer.className = "text-muted small mb-0";
            footer.textContent = `Artifacts used: ${artifacts.join(" | ")}`;
            wrap.appendChild(footer);
        }

        return wrap;
    }

    function renderQualityReport(report) {
        const wrap = document.createElement("div");
        const heading = document.createElement("h3");
        heading.className = "h5 mt-4";
        heading.textContent = "Data Quality Report";
        wrap.appendChild(heading);

        const rows = [
            { Metric: "Total rows", Value: report.total_rows },
            { Metric: "Total columns", Value: report.total_cols },
            { Metric: "Default rate (train)", Value: report.target_default_rate },
            { Metric: "Income cap (99th pct)", Value: report.income_cap_p99 },
        ];
        wrap.appendChild(makeTable(["Metric", "Value"], rows));

        const splitHeading = document.createElement("h4");
        splitHeading.className = "h6 mt-3";
        splitHeading.textContent = "Split sizes";
        wrap.appendChild(splitHeading);
        wrap.appendChild(makeSplitSizeTable(report.split_sizes));
        return wrap;
    }

    function familyHeadingText(family) {
        return `${family.charAt(0).toUpperCase()}${family.slice(1)} Group Audit`;
    }

    function makeFamilySection(family, data) {
        const section = document.createElement("section");
        section.className = "mt-4";

        const heading = document.createElement("h3");
        heading.className = "h5";
        heading.textContent = familyHeadingText(family);
        section.appendChild(heading);

        section.appendChild(makeTable(data.columns, data.rows));

        const evaluableRows = data.rows.filter((row) => row.evaluable === true);
        const summary = document.createElement("p");
        summary.className = "text-muted";
        if (evaluableRows.length > 0) {
            const lead = evaluableRows[0];
            summary.textContent = `${evaluableRows.length} evaluable cells | DI: ${lead.di_pass ? "PASS" : "FAIL"} | EOD: ${lead.eod_pass ? "PASS" : "FAIL"} | Brier: ${lead.brier_pass ? "PASS" : "FAIL"}`;
        } else {
            summary.textContent = "0 evaluable cells | DI: FAIL | EOD: FAIL | Brier: FAIL";
        }
        section.appendChild(summary);
        return section;
    }

    function initEDA() {
        const container = document.getElementById("eda-content");
        if (!container) {
            return;
        }

        const content = document.createElement("div");
        const edaPlots = plots.eda || [];

        if (edaPlots.length > 0) {
            content.appendChild(makePlotGrid(edaPlots));
        } else {
            content.appendChild(makeAlert("EDA plots not available. Run Module 1 (data_pipeline.py) first.", "secondary"));
        }

        if (analytics.qualityReport) {
            content.appendChild(renderQualityReport(analytics.qualityReport));
        } else {
            content.appendChild(
                makeAlert(
                    "Data quality report not available. Run Module 1 (data_pipeline.py) first.",
                    "secondary"
                )
            );
        }

        container.replaceChildren(content);
    }

    function initEval() {
        const container = document.getElementById("eval-content");
        if (!container) {
            return;
        }

        const content = document.createElement("div");
        const evalPlots = plots.eval || [];

        if (evalPlots.length > 0) {
            content.appendChild(makePlotGrid(evalPlots));
        } else {
            content.appendChild(
                makeAlert(
                    "Evaluation plots not available. Run Module 3 (train.py) and Module 4 (fairness_audit.py) first.",
                    "secondary"
                )
            );
        }

        if (analytics.championReport) {
            const heading = document.createElement("h3");
            heading.className = "h5 mt-4";
            heading.textContent = "Champion Confirmation Report";
            const pre = document.createElement("pre");
            pre.className = "bg-light p-3 rounded";
            pre.textContent = analytics.championReport;
            content.appendChild(heading);
            content.appendChild(pre);
        } else {
            content.appendChild(
                makeAlert(
                    "Champion report not available. Run Module 3 (train.py) first.",
                    "secondary"
                )
            );
        }

        container.replaceChildren(content);
    }

    function initSHAP() {
        const container = document.getElementById("shap-content");
        if (!container) {
            return;
        }

        const content = document.createElement("div");
        const shapPlots = plots.shap || [];

        if (shapPlots.length > 0) {
            content.appendChild(makePlotGrid(shapPlots));
        } else {
            content.appendChild(
                makeAlert(
                    "SHAP plots not available. Run Module 4 (fairness_audit.py) first.",
                    "secondary"
                )
            );
        }

        const description = document.createElement("div");
        description.className = "alert alert-secondary mt-3";
        description.innerHTML = "<strong>How to read SHAP plots:</strong> SHAP (SHapley Additive exPlanations) values show each feature's contribution to the model output. Positive SHAP values push the prediction toward default (higher risk). Negative values push toward non-default (lower risk). The global bar chart shows mean absolute impact across all test-set applicants. The beeswarm plot shows the distribution of SHAP values per feature. Local force plots show the top 10 drivers for a single representative applicant in each decision band.";
        content.appendChild(description);

        container.replaceChildren(content);
    }

    function initFairness() {
        const container = document.getElementById("fairness-content");
        if (!container) {
            return;
        }

        const content = document.createElement("div");
        const comparison = analytics.fairnessComparison || {};
        const fairnessPlots = plots.fairness || [];
        const summaryCard = fairnessPlots.find((entry) => entry.filename === "fairness_summary_card.png");
        const otherPlots = fairnessPlots.filter((entry) => entry.filename !== "fairness_summary_card.png");
        const auditView = document.createElement("div");
        const comparisonView = document.createElement("div");
        comparisonView.className = "d-none";

        const viewToggle = document.createElement("div");
        viewToggle.className = "btn-group mb-4";
        viewToggle.setAttribute("role", "group");
        viewToggle.setAttribute("aria-label", "Fairness views");

        const auditButton = document.createElement("button");
        auditButton.type = "button";
        auditButton.className = "btn btn-outline-primary active";
        auditButton.textContent = "Current Audit";

        const comparisonButton = document.createElement("button");
        comparisonButton.type = "button";
        comparisonButton.className = "btn btn-outline-primary";
        comparisonButton.textContent = comparison.mode === "live_toggle"
            ? "Live Comparison"
            : "Experimental Comparison";

        const fairnessIntro = document.createElement("p");
        fairnessIntro.className = "text-muted";
        fairnessIntro.textContent = "Review the current deployed proxy audit first, then switch to the guarded comparison view to see whether any fairness-oriented experimental artifacts are available.";
        content.appendChild(fairnessIntro);

        function activateFairnessView(nextView) {
            const comparisonActive = nextView === "comparison";
            auditButton.classList.toggle("active", !comparisonActive);
            comparisonButton.classList.toggle("active", comparisonActive);
            auditView.classList.toggle("d-none", comparisonActive);
            comparisonView.classList.toggle("d-none", !comparisonActive);
        }

        auditButton.addEventListener("click", () => activateFairnessView("audit"));
        comparisonButton.addEventListener("click", () => activateFairnessView("comparison"));
        viewToggle.appendChild(auditButton);
        viewToggle.appendChild(comparisonButton);
        content.appendChild(viewToggle);

        if (summaryCard) {
            const card = document.createElement("div");
            card.className = "card mb-4";
            const img = document.createElement("img");
            img.src = summaryCard.url;
            img.className = "card-img-top";
            img.alt = summaryCard.title;
            img.loading = "lazy";
            const body = document.createElement("div");
            body.className = "card-body";
            const title = document.createElement("h6");
            title.className = "card-title";
            title.textContent = summaryCard.title;
            body.appendChild(title);
            card.appendChild(img);
            card.appendChild(body);
            auditView.appendChild(card);
        }

        if (otherPlots.length > 0) {
            auditView.appendChild(makePlotGrid(otherPlots));
        }

        const fairnessTables = analytics.fairnessTables || {};
        let renderedTable = false;
        ["primary", "secondary", "tertiary"].forEach((family) => {
            if (fairnessTables[family]) {
                renderedTable = true;
                auditView.appendChild(makeFamilySection(family, fairnessTables[family]));
            }
        });

        if (!summaryCard && otherPlots.length === 0 && !renderedTable) {
            auditView.appendChild(
                makeAlert(
                    "Fairness audit data not available. Run Module 4 (fairness_audit.py) first.",
                    "secondary"
                )
            );
        }

        const disclaimer = document.createElement("div");
        disclaimer.className = "alert alert-warning mt-3";
        disclaimer.innerHTML = "<strong>Regulatory disclaimer:</strong> These metrics evaluate stability across proxy-defined subgroups only and must not be interpreted as proof of fairness across legally protected characteristics.";
        auditView.appendChild(disclaimer);

        comparisonView.appendChild(makeFairnessComparisonView(comparison));
        content.appendChild(auditView);
        content.appendChild(comparisonView);

        container.replaceChildren(content);
    }

    function initDrift() {
        const container = document.getElementById("drift-content");
        if (!container) {
            return;
        }

        const drift = analytics.drift || {};
        const content = document.createElement("div");

        const heading = document.createElement("h3");
        heading.className = "h5";
        heading.textContent = "Live Scoring Drift";
        content.appendChild(heading);

        const summary = document.createElement("p");
        summary.className = "text-muted";
        summary.textContent = "A lightweight model-health monitor using the most recent persisted score runs from SQLite.";
        content.appendChild(summary);

        appendBadgeRow(content, [
            makeStatusBadge(String(drift.status_label || "Watch").toUpperCase(), drift.status === "alert" ? "danger" : drift.status === "watch" ? "warning" : "success"),
            makeStatusBadge(`${drift.sample_size || 0}/${drift.lookback || 0} runs`, "secondary"),
            makeStatusBadge("SQLite-backed", "secondary"),
        ]);

        const rows = [
            { Metric: "Baseline mean PD", Value: drift.baseline_mean_text || "—" },
            { Metric: "Live mean PD", Value: drift.live_mean_text || "—" },
            { Metric: "Live std. dev.", Value: drift.live_std_text || "—" },
            { Metric: "Deviation %", Value: drift.deviation_pct_text || "—" },
            { Metric: "Sample size", Value: `${drift.sample_size || 0} / ${drift.lookback || 0}` },
        ];
        content.appendChild(makeTable(["Metric", "Value"], rows));

        const note = document.createElement("div");
        note.className = `alert mt-3 ${
            drift.status === "alert"
                ? "alert-danger"
                : drift.status === "watch"
                    ? "alert-warning"
                    : "alert-success"
        }`;
        note.textContent = drift.alert_message || "Drift signal unavailable.";
        content.appendChild(note);

        const thresholds = document.createElement("p");
        thresholds.className = "text-muted small mb-0";
        thresholds.textContent = `Watch threshold: ${drift.watch_threshold_pct || 0}% | Alert threshold: ${drift.alert_threshold_pct || 0}% | Minimum sample: ${drift.min_sample_size || 0}`;
        content.appendChild(thresholds);

        container.replaceChildren(content);
    }

    document.addEventListener("DOMContentLoaded", () => {
        const tabEls = document.querySelectorAll("[data-bs-toggle=\"tab\"]");
        const initialized = {
            "#eda-tab-pane": true,
        };

        tabEls.forEach((tabEl) => {
            tabEl.addEventListener("shown.bs.tab", (event) => {
                const target = event.target.getAttribute("data-bs-target");
                if (initialized[target]) {
                    return;
                }
                initialized[target] = true;

                if (target === "#eda-tab-pane") {
                    initEDA();
                }
                if (target === "#eval-tab-pane") {
                    initEval();
                }
                if (target === "#shap-tab-pane") {
                    initSHAP();
                }
                if (target === "#fairness-tab-pane") {
                    initFairness();
                }
                if (target === "#drift-tab-pane") {
                    initDrift();
                }
            });
        });

        initEDA();
    });
})();
