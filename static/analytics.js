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
        const fairnessPlots = plots.fairness || [];
        const summaryCard = fairnessPlots.find((entry) => entry.filename === "fairness_summary_card.png");
        const otherPlots = fairnessPlots.filter((entry) => entry.filename !== "fairness_summary_card.png");

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
            content.appendChild(card);
        }

        if (otherPlots.length > 0) {
            content.appendChild(makePlotGrid(otherPlots));
        }

        const fairnessTables = analytics.fairnessTables || {};
        let renderedTable = false;
        ["primary", "secondary", "tertiary"].forEach((family) => {
            if (fairnessTables[family]) {
                renderedTable = true;
                content.appendChild(makeFamilySection(family, fairnessTables[family]));
            }
        });

        if (!summaryCard && otherPlots.length === 0 && !renderedTable) {
            content.appendChild(
                makeAlert(
                    "Fairness audit data not available. Run Module 4 (fairness_audit.py) first.",
                    "secondary"
                )
            );
        }

        const disclaimer = document.createElement("div");
        disclaimer.className = "alert alert-warning mt-3";
        disclaimer.innerHTML = "<strong>Regulatory disclaimer:</strong> These metrics evaluate stability across proxy-defined subgroups only and must not be interpreted as proof of fairness across legally protected characteristics.";
        content.appendChild(disclaimer);

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
            });
        });

        initEDA();
    });
})();
