(function () {
    const page = document.body.dataset.page || "";
    if (page !== "analyst") {
        return;
    }
    const hubData = window.__ANALYST_HUB_DATA__ || {};
    const applicationMap = new Map(
        ((hubData.applications || []).map((application) => [String(application.id), application]))
    );
    const healthSnapshot = hubData.healthSnapshot || {};
    const AGGREGATE_SECTIONS = [
        ["bureau_agg", "brief-bureau"],
        ["previous_agg", "brief-previous"],
        ["installments_agg", "brief-installments"],
        ["pos_cash_agg", "brief-pos-cash"],
        ["credit_card_agg", "brief-credit-card"],
    ];

    const loadingState = document.getElementById("analyst-loading-state");
    if (!loadingState) {
        return;
    }
    const table = document.getElementById("analyst-applications-table");
    if (!table) {
        loadingState.classList.add("is-hidden");
        return;
    }

    const searchInput = document.getElementById("analyst-search");
    const sortSelect = document.getElementById("analyst-sort");
    const filterButtons = Array.from(document.querySelectorAll("[data-status-filter]"));
    const emptyState = document.getElementById("analyst-filter-empty-state");
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    const tbody = table.querySelector("tbody");
    const state = {
        status: "ALL",
        search: "",
        sort: "date_desc",
    };

    function normalizedValue(value) {
        return String(value || "").trim().toLowerCase();
    }

    function rowData(row) {
        return {
            id: row.dataset.applicationId || "",
            skId: row.dataset.skId || "",
            applicantName: row.dataset.applicantName || "",
            submittedAt: row.dataset.submittedAt || "",
            updatedAt: row.dataset.updatedAt || "",
            tier: row.dataset.tier || "",
            status: row.dataset.status || "",
            lastDecision: row.dataset.lastDecision || "",
            lastProbability: row.dataset.lastProbability || "",
            modelVersion: row.dataset.modelVersion || "",
        };
    }

    function parseProbability(value) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function formatProbability(value) {
        const parsed = parseProbability(value);
        if (parsed === null) {
            return "-";
        }
        return `${(parsed * 100).toFixed(1)}%`;
    }

    function decisionClass(decision) {
        if (decision === "APPROVE") {
            return "approve";
        }
        if (decision === "DECLINE") {
            return "decline";
        }
        return "review";
    }

    function statusClass(status) {
        if (status === "APPROVED") {
            return "approve";
        }
        if (status === "DECLINED") {
            return "decline";
        }
        if (status === "REVIEW" || status === "READY_FOR_REVIEW") {
            return "review";
        }
        return "neutral";
    }

    function formatMaybeCurrency(value) {
        const parsed = Number.parseFloat(value);
        if (!Number.isFinite(parsed)) {
            return "-";
        }
        return new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: "USD",
            maximumFractionDigits: 0,
        }).format(parsed);
    }

    function formatMaybeDecimal(value) {
        const parsed = Number.parseFloat(value);
        if (!Number.isFinite(parsed)) {
            return "-";
        }
        return parsed.toFixed(3);
    }

    function rowMatches(row) {
        const data = rowData(row);
        const searchBlob = [
            data.skId,
            data.applicantName,
            data.status,
            data.tier,
            data.lastDecision,
        ].join(" ").toLowerCase();

        const searchMatch = !state.search || searchBlob.includes(state.search);
        const statusMatch = state.status === "ALL" || data.status === state.status;
        return searchMatch && statusMatch;
    }

    function compareRows(leftRow, rightRow) {
        const left = rowData(leftRow);
        const right = rowData(rightRow);

        if (state.sort === "date_asc") {
            return left.submittedAt.localeCompare(right.submittedAt);
        }
        if (state.sort === "name_asc") {
            return left.applicantName.localeCompare(right.applicantName);
        }
        if (state.sort === "prob_desc") {
            return (parseProbability(right.lastProbability) ?? -1) - (parseProbability(left.lastProbability) ?? -1);
        }
        if (state.sort === "prob_asc") {
            return (parseProbability(left.lastProbability) ?? Number.MAX_SAFE_INTEGER) - (parseProbability(right.lastProbability) ?? Number.MAX_SAFE_INTEGER);
        }
        return right.submittedAt.localeCompare(left.submittedAt);
    }

    function updateDetailPanel(row) {
        const data = rowData(row);
        const applicant = document.getElementById("detail-applicant-name");
        if (!applicant) {
            return;
        }

        const status = document.getElementById("detail-status");
        const skId = document.getElementById("detail-sk-id");
        const date = document.getElementById("detail-date");
        const tier = document.getElementById("detail-tier");
        const modelVersion = document.getElementById("detail-model-version");
        const probability = document.getElementById("detail-probability");
        const decision = document.getElementById("detail-decision");
        const viewLink = document.getElementById("detail-view-link");
        const editLink = document.getElementById("detail-edit-link");
        const reportLink = document.getElementById("detail-report-link");
        const analyzeForm = document.getElementById("detail-analyze-form");

        applicant.textContent = data.applicantName;
        status.className = `analyst-status-badge ${statusClass(data.status)}`;
        status.textContent = data.status.replaceAll("_", " ");
        skId.textContent = data.skId || "-";
        date.textContent = data.submittedAt || "-";
        tier.textContent = data.tier || "-";
        modelVersion.textContent = data.modelVersion || "-";
        probability.textContent = formatProbability(data.lastProbability);
        decision.innerHTML = data.lastDecision
            ? `<span class="decision-badge ${decisionClass(data.lastDecision)}">${data.lastDecision}</span>`
            : '<span class="analyst-detail-placeholder">No saved decision</span>';

        viewLink.href = `/analyst/applications/${data.id}`;
        editLink.href = `/analyst/applications/${data.id}#applicant_name`;
        reportLink.href = `/analyst/applications/${data.id}/report`;
        analyzeForm.action = `/analyst/applications/${data.id}/analyze`;
    }

    function markSelectedRow(selectedRow) {
        rows.forEach((row) => row.classList.toggle("selected", row === selectedRow));
        updateDetailPanel(selectedRow);
    }

    function aggregateCoverageText(sectionPayload) {
        if (!sectionPayload || typeof sectionPayload !== "object") {
            return "Not active";
        }
        const values = Object.values(sectionPayload);
        const populated = values.filter((value) => {
            if (value === null || value === undefined || value === "") {
                return false;
            }
            if (typeof value === "number") {
                return value !== 0;
            }
            return true;
        }).length;
        return `${populated}/${values.length} populated`;
    }

    function openCreditBrief(applicationId) {
        const application = applicationMap.get(String(applicationId));
        if (!application) {
            return;
        }
        const payload = application.application_payload_obj || {};
        const appPayload = payload.application || {};
        const overlay = document.getElementById("credit-brief-overlay");
        const drawer = document.getElementById("credit-brief-drawer");
        if (!overlay || !drawer) {
            return;
        }

        document.getElementById("brief-applicant-name").textContent = application.applicant_name || "-";
        document.getElementById("brief-sk-id").textContent = application.sk_id_curr || "-";
        document.getElementById("brief-submitted-at").textContent = application.submitted_at || "-";
        document.getElementById("brief-updated-at").textContent = application.updated_at || "-";
        const briefStatus = document.getElementById("brief-status");
        briefStatus.className = `analyst-status-badge ${statusClass(application.current_status)}`;
        briefStatus.textContent = String(application.current_status || "-").replaceAll("_", " ");
        document.getElementById("brief-tier").textContent = application.tier_type || "-";

        document.getElementById("brief-income").textContent = formatMaybeCurrency(appPayload.AMT_INCOME_TOTAL_CAPPED);
        document.getElementById("brief-credit").textContent = formatMaybeCurrency(appPayload.AMT_CREDIT);
        document.getElementById("brief-annuity").textContent = formatMaybeCurrency(appPayload.AMT_ANNUITY);
        document.getElementById("brief-goods-price").textContent = formatMaybeCurrency(appPayload.AMT_GOODS_PRICE);

        document.getElementById("brief-ext-1").textContent = formatMaybeDecimal(appPayload.EXT_SOURCE_1);
        document.getElementById("brief-ext-2").textContent = formatMaybeDecimal(appPayload.EXT_SOURCE_2);
        document.getElementById("brief-ext-3").textContent = formatMaybeDecimal(appPayload.EXT_SOURCE_3);

        AGGREGATE_SECTIONS.forEach(([sectionName, elementId]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = application.tier_type === "FULL"
                    ? aggregateCoverageText(payload[sectionName])
                    : "Not active";
            }
        });

        const lastDecision = document.getElementById("brief-last-decision");
        if (application.last_decision) {
            lastDecision.innerHTML = `<span class="decision-badge ${decisionClass(application.last_decision)}">${application.last_decision}</span>`;
        } else {
            lastDecision.textContent = "No saved decision";
        }
        document.getElementById("brief-last-probability").textContent = formatProbability(application.last_probability);
        document.getElementById("brief-model-version").textContent = application.last_model_version || "-";

        const fairnessStatus = document.getElementById("brief-fairness-status");
        fairnessStatus.innerHTML = `<span class="decision-badge ${healthSnapshot.fairness_audit_passed ? "approve" : "decline"}">${healthSnapshot.fairness_audit_passed ? "Passed" : "Not passed"}</span>`;

        overlay.classList.remove("is-hidden");
        overlay.setAttribute("aria-hidden", "false");
        document.body.classList.add("drawer-open");
        drawer.focus();
    }

    function closeCreditBrief() {
        const overlay = document.getElementById("credit-brief-overlay");
        if (!overlay) {
            return;
        }
        overlay.classList.add("is-hidden");
        overlay.setAttribute("aria-hidden", "true");
        document.body.classList.remove("drawer-open");
    }

    function refreshRows() {
        const visibleRows = rows.filter((row) => rowMatches(row)).sort(compareRows);
        rows.forEach((row) => row.classList.add("is-hidden"));
        visibleRows.forEach((row) => {
            row.classList.remove("is-hidden");
            tbody.appendChild(row);
        });

        if (emptyState) {
            emptyState.classList.toggle("is-hidden", visibleRows.length > 0);
        }

        const activeSelected = visibleRows.find((row) => row.classList.contains("selected"));
        if (!activeSelected && visibleRows.length > 0) {
            markSelectedRow(visibleRows[0]);
        }
    }

    filterButtons.forEach((button) => {
        button.addEventListener("click", function () {
            state.status = button.dataset.statusFilter || "ALL";
            filterButtons.forEach((item) => item.classList.toggle("active", item === button));
            refreshRows();
        });
    });

    if (searchInput) {
        searchInput.addEventListener("input", function () {
            state.search = normalizedValue(searchInput.value);
            refreshRows();
        });
    }

    if (sortSelect) {
        sortSelect.addEventListener("change", function () {
            state.sort = sortSelect.value || "date_desc";
            refreshRows();
        });
    }

    rows.forEach((row) => {
        row.addEventListener("click", function (event) {
            const target = event.target;
            if (target instanceof HTMLElement && target.closest("a, button, form")) {
                if (target.closest("[data-row-select]")) {
                    markSelectedRow(row);
                }
                return;
            }
            markSelectedRow(row);
        });
    });

    document.querySelectorAll("[data-credit-brief-open]").forEach((link) => {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            openCreditBrief(link.getAttribute("data-credit-brief-open"));
        });
    });

    document.querySelectorAll("[data-credit-brief-close]").forEach((element) => {
        element.addEventListener("click", function () {
            closeCreditBrief();
        });
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            closeCreditBrief();
        }
    });

    loadingState.classList.add("is-hidden");
    refreshRows();
})();
