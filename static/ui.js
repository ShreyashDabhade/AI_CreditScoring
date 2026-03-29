(function () {
    const BASE_URL = window.__API_BASE_URL__ || window.location.origin || "";
    const config = window.__MASTERMIND_UI__;
    if (!config) {
        return;
    }

    const page = document.body.dataset.page || "";
    const initialHealth = window.__MASTERMIND_HEALTH_SNAPSHOT__ || null;
    const state = {
        tier: "REDUCED",
    };

    const FIELD_GROUPS = {
        application: [
            "AMT_INCOME_TOTAL_CAPPED",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "NAME_CONTRACT_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
        ],
        bureau_agg: [
            "BUREAU_LOAN_COUNT",
            "BUREAU_ACTIVE_COUNT",
            "BUREAU_CLOSED_COUNT",
            "BUREAU_AMT_CREDIT_SUM_SUM",
            "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
            "BUREAU_DEBT_TO_CREDIT_RATIO",
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",
            "BUREAU_CREDIT_DAY_OVERDUE_MAX",
            "BUREAU_DAYS_CREDIT_MAX",
            "BUREAU_CNT_CREDIT_PROLONG_SUM",
        ],
        previous_agg: [
            "PREV_APP_COUNT",
            "PREV_APPROVED_COUNT",
            "PREV_REFUSED_COUNT",
            "PREV_APPROVAL_RATE",
            "PREV_REFUSAL_RATE",
            "PREV_AMT_APPLICATION_MEAN",
            "PREV_AMT_CREDIT_MEAN",
            "PREV_AMT_GOODS_PRICE_MEAN",
            "PREV_APP_CREDIT_DIFF_MEAN",
            "PREV_DAYS_DECISION_MAX",
            "PREV_RATE_DOWN_PAYMENT_MEAN",
        ],
        installments_agg: [
            "INST_RECORD_COUNT",
            "INST_MISSED_RATE",
            "INST_DPD_MEAN",
            "INST_DPD_MAX",
            "INST_PAYMENT_RATIO_MEAN",
            "INST_PAYMENT_RATIO_MIN",
            "INST_LATE_COUNT",
        ],
        pos_cash_agg: [
            "POS_RECORD_COUNT",
            "POS_DPD_MEAN",
            "POS_DPD_MAX",
            "POS_DPD_DEF_MEAN",
            "POS_DPD_DEF_MAX",
            "POS_COMPLETED_RATE",
            "POS_ACTIVE_RATE",
            "POS_CNT_INSTALMENT_FUTURE_MEAN",
        ],
        credit_card_agg: [
            "CC_RECORD_COUNT",
            "CC_BALANCE_MEAN",
            "CC_LIMIT_MEAN",
            "CC_UTILIZATION_MEAN",
            "CC_PAYMENT_RATIO_MEAN",
            "CC_DPD_MEAN",
            "CC_DPD_MAX",
            "CC_DRAWINGS_ATM_SUM",
            "CC_DRAWINGS_CURRENT_SUM",
        ],
    };

    function cloneValue(value) {
        return JSON.parse(JSON.stringify(value || {}));
    }

    function resolveApiUrl(path) {
        if (!path) {
            return BASE_URL;
        }
        if (/^https?:\/\//i.test(path)) {
            return path;
        }
        return `${BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
    }

    function navScrollState() {
        const nav = document.querySelector("[data-site-nav]");
        if (!nav) {
            return;
        }
        nav.classList.toggle("scrolled", window.scrollY > 60);
    }

    function formatProbability(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return "0%";
        }
        return `${Math.round(numeric * 100)}%`;
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

    async function readJson(response) {
        const text = await response.text();
        if (!text) {
            return {};
        }
        try {
            return JSON.parse(text);
        } catch (error) {
            return {
                error_code: "invalid_response",
                message: "The API returned invalid JSON.",
            };
        }
    }

    function clearValidation(form) {
        form.querySelectorAll(".invalid").forEach((field) => {
            field.classList.remove("invalid");
            field.removeAttribute("aria-invalid");
        });
    }

    function markInvalid(control) {
        control.classList.add("invalid");
        control.setAttribute("aria-invalid", "true");
    }

    function setTier(nextTier, elements) {
        state.tier = nextTier;
        elements.tierButtons.forEach((button) => {
            const active = button.dataset.tier === nextTier;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
        });

        const isFull = nextTier === "FULL";
        elements.bureauFields.classList.toggle("is-visible", isFull);
        elements.bureauDivider.classList.toggle("is-visible", isFull);
        elements.tierDescription.textContent = isFull
            ? "33 application fields + all 5 aggregate sections - full pipeline coverage"
            : "33 fields - application section only";

        elements.resultSection.classList.add("is-hidden");
        elements.formStage.classList.remove("is-hidden");
        elements.loadingStage.classList.add("is-hidden");
        elements.feedback.textContent = "";
    }

    function valueFromControl(control) {
        const rawValue = control.value.trim();
        if (!rawValue) {
            return null;
        }

        if (control.dataset.type === "number") {
            const parsed = Number.parseFloat(rawValue);
            if (!Number.isFinite(parsed)) {
                return null;
            }
            if (control.dataset.negate === "true") {
                return -Math.abs(parsed);
            }
            return parsed;
        }

        return rawValue;
    }

    function buildPayload(form) {
        const payload = cloneValue((config.samplePayloads || {})[state.tier] || {});
        payload.tier = state.tier;
        const sections = state.tier === "FULL"
            ? ["application", "bureau_agg", "previous_agg", "installments_agg", "pos_cash_agg", "credit_card_agg"]
            : ["application"];
        const missing = [];

        if (state.tier === "FULL") {
            sections
                .filter((sectionName) => sectionName !== "application")
                .forEach((sectionName) => {
                    payload[sectionName] = {};
                    FIELD_GROUPS[sectionName].forEach((fieldName) => {
                        payload[sectionName][fieldName] = 0;
                    });
                });
        }

        sections.forEach((sectionName) => {
            payload[sectionName] = payload[sectionName] || {};
            FIELD_GROUPS[sectionName].forEach((fieldName) => {
                const control = form.querySelector(`[data-section="${sectionName}"][name="${fieldName}"]`);
                if (!control) {
                    return;
                }

                const value = valueFromControl(control);
                if (value === null || value === "") {
                    if (sectionName === "application") {
                        missing.push(control);
                    } else {
                        payload[sectionName][fieldName] = 0;
                    }
                    return;
                }
                payload[sectionName][fieldName] = value;
            });
        });

        return {
            missing,
            payload,
        };
    }

    function renderMetadata(elements, response) {
        const rows = [
            ["Coverage Tier", response.coverage_tier || "-"],
            ["Calibrated", response.calibrated ? "Yes" : "No"],
            ["Fairness Audit", response.model_fairness_audit_passed ? "Passed" : "Not passed"],
            ["Model Version", response.model_version || "-", true],
        ];

        if (response.escalate) {
            rows.push(["Escalation Flag", "\u26a0 Referred for manual review"]);
        }

        elements.metadata.replaceChildren();
        rows.forEach((row) => {
            const item = document.createElement("div");
            item.className = "metadata-row";

            const label = document.createElement("span");
            label.className = "metadata-label";
            label.textContent = row[0];

            const value = document.createElement("span");
            value.className = row[2] ? "metadata-value monospace" : "metadata-value";
            value.textContent = row[1];

            item.append(label, value);
            elements.metadata.appendChild(item);
        });
    }

    function renderDrivers(elements, explanations) {
        elements.driverList.replaceChildren();
        (explanations || []).slice(0, 5).forEach((item) => {
            const row = document.createElement("div");
            row.className = "driver-row";

            const title = document.createElement("h3");
            title.textContent = item.feature || "Unknown feature";

            const reason = document.createElement("p");
            reason.textContent = item.reason || "No explanation was returned.";

            row.append(title, reason);
            elements.driverList.appendChild(row);
        });
    }

    function renderResult(elements, response) {
        elements.probability.textContent = formatProbability(response.probability_of_default);
        elements.decision.textContent = response.decision || "REVIEW";
        elements.decision.className = `decision-pill ${decisionClass(response.decision)}`;
        renderMetadata(elements, response);
        renderDrivers(elements, response.top_5_explanations || []);

        elements.formStage.classList.add("is-hidden");
        elements.loadingStage.classList.add("is-hidden");
        elements.resultSection.classList.remove("is-hidden");
        elements.feedback.textContent = "";
    }

    async function submitAnalysis(elements) {
        clearValidation(elements.form);
        const built = buildPayload(elements.form);
        if (built.missing.length > 0) {
            built.missing.forEach(markInvalid);
            elements.feedback.textContent = "Complete all visible fields before submitting the application.";
            return;
        }

        elements.feedback.textContent = "";
        elements.formStage.classList.add("is-hidden");
        elements.loadingStage.classList.remove("is-hidden");
        elements.resultSection.classList.add("is-hidden");

        try {
            const response = await fetch(resolveApiUrl(config.routes.score), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify(built.payload),
            });
            const body = await readJson(response);

            if (!response.ok) {
                throw new Error(body.message || "The scoring request failed.");
            }

            renderResult(elements, body);
        } catch (error) {
            elements.loadingStage.classList.add("is-hidden");
            elements.formStage.classList.remove("is-hidden");
            elements.feedback.textContent = error.message || "The scoring request failed.";
        }
    }

    function resetAnalysis(elements) {
        const currentTier = state.tier;
        elements.form.reset();
        clearValidation(elements.form);
        elements.feedback.textContent = "";
        elements.resultSection.classList.add("is-hidden");
        elements.loadingStage.classList.add("is-hidden");
        elements.formStage.classList.remove("is-hidden");
        setTier(currentTier, elements);
    }

    function initAnalyzePage() {
        const elements = {
            tierButtons: Array.from(document.querySelectorAll("[data-tier-toggle]")),
            tierDescription: document.getElementById("tier-description"),
            bureauFields: document.getElementById("bureau-fields"),
            bureauDivider: document.getElementById("bureau-divider"),
            form: document.getElementById("analysis-form"),
            formStage: document.getElementById("analysis-form-stage"),
            loadingStage: document.getElementById("analysis-loading-stage"),
            resultSection: document.getElementById("analysis-result"),
            scoreButton: document.getElementById("score-button"),
            scoreAnother: document.getElementById("score-another"),
            feedback: document.getElementById("analysis-feedback"),
            probability: document.getElementById("result-probability"),
            decision: document.getElementById("result-decision"),
            metadata: document.getElementById("result-metadata"),
            driverList: document.getElementById("driver-list"),
        };

        elements.tierButtons.forEach((button) => {
            button.addEventListener("click", function () {
                setTier(button.dataset.tier, elements);
            });
        });
        elements.scoreButton.addEventListener("click", function () {
            submitAnalysis(elements);
        });
        elements.scoreAnother.addEventListener("click", function () {
            resetAnalysis(elements);
        });

        elements.form.querySelectorAll("input, select").forEach((control) => {
            control.addEventListener("input", function () {
                control.classList.remove("invalid");
                control.removeAttribute("aria-invalid");
            });
            control.addEventListener("change", function () {
                control.classList.remove("invalid");
                control.removeAttribute("aria-invalid");
            });
        });

        setTier("REDUCED", elements);
    }

    function applyStatusHealth(health) {
        const system = document.getElementById("status-system");
        const version = document.getElementById("status-version");
        const fairness = document.getElementById("status-fairness");
        const tiers = document.getElementById("status-tiers");
        const error = document.getElementById("status-error");

        if (!system || !version || !fairness || !tiers || !error) {
            return;
        }

        if (!health) {
            error.classList.remove("is-hidden");
            system.innerHTML = '<span class="status-dot degraded"></span>Degraded';
            fairness.textContent = "Not passed";
            fairness.className = "status-value fairness-fail";
            version.textContent = "Unavailable";
            tiers.textContent = "Unavailable";
            return;
        }

        error.classList.add("is-hidden");
        system.innerHTML = health.status === "ok"
            ? '<span class="status-dot ok"></span>Operational'
            : '<span class="status-dot degraded"></span>Degraded';
        version.textContent = health.model_version || "Unavailable";
        fairness.textContent = health.fairness_audit_passed ? "Passed" : "Not passed";
        fairness.className = health.fairness_audit_passed
            ? "status-value fairness-pass"
            : "status-value fairness-fail";
        tiers.textContent = (health.coverage_tiers_available || []).join(", ") || "Unavailable";
    }

    async function initStatusPage() {
        applyStatusHealth(initialHealth);
        try {
            const response = await fetch(resolveApiUrl(config.routes.health), {
                headers: {
                    Accept: "application/json",
                },
            });
            if (!response.ok) {
                throw new Error("Health check failed.");
            }
            const body = await readJson(response);
            applyStatusHealth(body);
        } catch (error) {
            applyStatusHealth(null);
        }
    }

    window.addEventListener("scroll", navScrollState, { passive: true });
    navScrollState();

    if (page === "analyze") {
        initAnalyzePage();
    }

    if (page === "status") {
        initStatusPage();
    }
})();
