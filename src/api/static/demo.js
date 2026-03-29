(function () {
    const BASE_URL = window.__API_BASE_URL__ || window.location.origin || "";
    const config = window.__MASTERMIND_DEMO__;
    if (!config) {
        return;
    }

    const state = {
        tier: config.defaultTier || "REDUCED",
        payloads: structuredCloneSafe(config.samplePayloads || {}),
    };

    const elements = {
        tierButtons: Array.from(document.querySelectorAll(".tier-button")),
        loadReducedButton: document.getElementById("load-reduced"),
        loadFullButton: document.getElementById("load-full"),
        clearButton: document.getElementById("clear-form"),
        scoreButton: document.getElementById("score-button"),
        refreshHealthButton: document.getElementById("refresh-health"),
        sectionContainer: document.getElementById("section-container"),
        payloadPreview: document.getElementById("payload-preview"),
        resultSummary: document.getElementById("result-summary"),
        resultJson: document.getElementById("result-json"),
        requestSummary: document.getElementById("request-summary"),
        healthCard: document.getElementById("health-card"),
    };

    function structuredCloneSafe(value) {
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

    function getVisibleSections() {
        if (state.tier === "FULL") {
            return config.sections;
        }
        return config.sections.filter((section) => section.name === "application");
    }

    function getCurrentPayload() {
        const current = state.payloads[state.tier] || {};
        return structuredCloneSafe(current);
    }

    function setCurrentPayload(payload) {
        state.payloads[state.tier] = structuredCloneSafe(payload);
    }

    function setStatus(message, kind) {
        elements.requestSummary.textContent = message;
        elements.requestSummary.classList.remove("success", "error");
        if (kind) {
            elements.requestSummary.classList.add(kind);
        }
    }

    function formatJson(value) {
        return JSON.stringify(value, null, 2);
    }

    function toFieldValue(rawValue, kind) {
        if (rawValue === undefined || rawValue === null) {
            return "";
        }
        if (kind === "number") {
            return String(rawValue);
        }
        return rawValue;
    }

    function fromFieldValue(rawValue, kind) {
        if (rawValue === "") {
            return null;
        }
        if (kind === "number") {
            const parsed = Number(rawValue);
            return Number.isFinite(parsed) ? parsed : rawValue;
        }
        return rawValue;
    }

    function updateTierButtons() {
        elements.tierButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.tier === state.tier);
        });
    }

    function renderSections() {
        const payload = getCurrentPayload();
        const sections = getVisibleSections();
        elements.sectionContainer.innerHTML = "";

        sections.forEach((section) => {
            const sectionElement = document.createElement("section");
            sectionElement.className = "section-card";

            const title = document.createElement("h3");
            title.textContent = section.label;

            const copy = document.createElement("p");
            copy.textContent =
                section.name === "application"
                    ? "Base applicant attributes required by the public request contract."
                    : "Flattened aggregate features passed directly to the FULL builder artifact.";

            const grid = document.createElement("div");
            grid.className = "field-grid";

            const sectionPayload = payload[section.name] || {};

            section.fields.forEach((field) => {
                const fieldWrap = document.createElement("div");
                fieldWrap.className = "field";

                const label = document.createElement("label");
                label.htmlFor = `${section.name}-${field.name}`;
                label.textContent = field.name;

                let control;
                if (field.kind === "select") {
                    control = document.createElement("select");
                    const placeholder = document.createElement("option");
                    placeholder.value = "";
                    placeholder.textContent = "Select value";
                    control.appendChild(placeholder);
                    field.options.forEach((optionValue) => {
                        const option = document.createElement("option");
                        option.value = optionValue;
                        option.textContent = optionValue;
                        control.appendChild(option);
                    });
                } else {
                    control = document.createElement("input");
                    control.type = "number";
                    control.step = "any";
                    control.inputMode = "decimal";
                    control.placeholder = "Enter numeric value";
                }

                control.id = `${section.name}-${field.name}`;
                control.name = field.name;
                control.dataset.section = section.name;
                control.dataset.kind = field.kind;
                control.value = toFieldValue(sectionPayload[field.name], field.kind);
                control.addEventListener("input", handleFieldChange);
                control.addEventListener("change", handleFieldChange);

                fieldWrap.appendChild(label);
                fieldWrap.appendChild(control);
                grid.appendChild(fieldWrap);
            });

            sectionElement.appendChild(title);
            sectionElement.appendChild(copy);
            sectionElement.appendChild(grid);
            elements.sectionContainer.appendChild(sectionElement);
        });

        updatePreview();
    }

    function handleFieldChange(event) {
        const control = event.target;
        const payload = getCurrentPayload();
        const sectionName = control.dataset.section;
        const fieldKind = control.dataset.kind;
        payload[sectionName] = payload[sectionName] || {};
        payload[sectionName][control.name] = fromFieldValue(control.value, fieldKind);
        setCurrentPayload(payload);
        updatePreview();
    }

    function updatePreview() {
        elements.payloadPreview.textContent = formatJson(getCurrentPayload());
    }

    function renderResultSummary(responseBody, isError) {
        elements.resultSummary.className = `result-summary${isError ? " error" : ""}`;
        if (isError) {
            elements.resultSummary.textContent = `${responseBody.error_code}: ${responseBody.message}`;
            return;
        }

        const probability = Number(responseBody.probability_of_default || 0).toFixed(4);
        const fairnessLabel = responseBody.model_fairness_audit_passed ? "passed" : "failed";
        elements.resultSummary.innerHTML = [
            `<strong>${responseBody.decision}</strong> decision at PD ${probability}.`,
            `Tier ${responseBody.coverage_tier} scored with ${responseBody.model_version}.`,
            `Offline fairness audit ${fairnessLabel}; API returned ${responseBody.top_5_explanations.length} explanations.`,
            `<div class="result-chip-row">`,
            `<span class="result-chip">Escalate: ${responseBody.escalate}</span>`,
            `<span class="result-chip">Calibrated: ${responseBody.calibrated}</span>`,
            `<span class="result-chip">Fairness: ${responseBody.fairness_audit_version}</span>`,
            `</div>`,
        ].join(" ");
    }

    async function refreshHealth() {
        setStatus("Checking backend health...", null);
        try {
            const response = await fetch(resolveApiUrl(config.routes.health), {
                method: "GET",
                headers: { Accept: "application/json" },
            });
            const body = await response.json();
            const items = Array.from(elements.healthCard.querySelectorAll("dd"));
            items[0].textContent = body.status || "unknown";
            items[1].textContent = body.model_version || "unavailable";
            items[2].textContent = body.fairness_audit_passed ? "Passed" : "Not passed";
            items[3].textContent = (body.coverage_tiers_available || []).join(", ");
            setStatus("Backend health is ready. You can edit inputs and run the full scoring pipeline.", "success");
        } catch (error) {
            setStatus("Backend health check failed. The demo UI is up, but the API did not respond cleanly.", "error");
            const items = Array.from(elements.healthCard.querySelectorAll("dd"));
            items.forEach((item) => {
                item.textContent = "Unavailable";
            });
        }
    }

    async function submitScore() {
        const payload = getCurrentPayload();
        payload.tier = state.tier;
        setStatus(`Submitting ${state.tier} payload through validate -> transform -> score -> calibrate -> explain.`, null);
        elements.resultJson.textContent = "";
        try {
            const response = await fetch(resolveApiUrl(config.routes.score), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify(payload),
            });
            const body = await response.json();
            renderResultSummary(body, !response.ok);
            elements.resultJson.textContent = formatJson(body);
            if (response.ok) {
                setStatus(`${state.tier} scoring completed successfully through the backend pipeline.`, "success");
            } else {
                setStatus(`Backend rejected the request with ${body.error_code}.`, "error");
            }
        } catch (error) {
            elements.resultSummary.className = "result-summary error";
            elements.resultSummary.textContent = "The score request failed before the API could return JSON.";
            elements.resultJson.textContent = String(error);
            setStatus("Score request failed before a valid API response was returned.", "error");
        }
    }

    function resetCurrentTierToSample() {
        state.payloads[state.tier] = structuredCloneSafe(config.samplePayloads[state.tier] || {});
        renderSections();
        elements.resultSummary.className = "result-summary empty-state";
        elements.resultSummary.textContent = "Run a score request to populate this panel.";
        elements.resultJson.textContent = "";
        setStatus(`Loaded ${state.tier} sample payload into the form.`, null);
    }

    function clearCurrentTier() {
        const clearedPayload = {};
        getVisibleSections().forEach((section) => {
            clearedPayload[section.name] = {};
        });
        setCurrentPayload(clearedPayload);
        renderSections();
        elements.resultSummary.className = "result-summary empty-state";
        elements.resultSummary.textContent = "Run a score request to populate this panel.";
        elements.resultJson.textContent = "";
        setStatus(`Cleared the ${state.tier} form. Missing required fields will surface through the backend validator.`, null);
    }

    function setTier(nextTier) {
        state.tier = nextTier;
        updateTierButtons();
        if (!state.payloads[nextTier]) {
            state.payloads[nextTier] = structuredCloneSafe(config.samplePayloads[nextTier] || {});
        }
        renderSections();
        setStatus(`Switched to ${nextTier} coverage.`, null);
    }

    elements.tierButtons.forEach((button) => {
        button.addEventListener("click", () => setTier(button.dataset.tier));
    });
    elements.loadReducedButton.addEventListener("click", () => setTier("REDUCED"));
    elements.loadReducedButton.addEventListener("click", resetCurrentTierToSample);
    elements.loadFullButton.addEventListener("click", () => setTier("FULL"));
    elements.loadFullButton.addEventListener("click", resetCurrentTierToSample);
    elements.clearButton.addEventListener("click", clearCurrentTier);
    elements.scoreButton.addEventListener("click", submitScore);
    elements.refreshHealthButton.addEventListener("click", refreshHealth);

    updateTierButtons();
    renderSections();
    refreshHealth();
})();
