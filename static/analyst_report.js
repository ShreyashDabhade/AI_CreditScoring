(function () {
    const BASE_URL = window.__API_BASE_URL__ || window.location.origin || "";
    const simulatorConfig = window.__REPORT_SIMULATOR__;
    const root = document.getElementById("what-if-simulator-root");
    if (!simulatorConfig || !root) {
        return;
    }

    const inputs = Array.from(root.querySelectorAll("[data-sim-field]"));
    const sliders = Array.from(root.querySelectorAll("[data-sim-range]"));
    const runButton = document.getElementById("simulate-run-button");
    const resetButton = document.getElementById("simulate-reset-button");
    const loadingState = document.getElementById("simulate-loading-state");
    const errorState = document.getElementById("simulate-error-state");
    const resultCard = document.getElementById("simulate-result-card");
    const baselineValues = new Map(inputs.map((input) => [input.name, String(input.defaultValue || "").trim()]));
    const slidersByTarget = new Map(sliders.map((slider) => [slider.dataset.targetInput, slider]));
    const reportChatContext = window.__REPORT_CHAT_CONTEXT__ && typeof window.__REPORT_CHAT_CONTEXT__ === "object"
        ? window.__REPORT_CHAT_CONTEXT__
        : null;

    function resolveApiUrl(path) {
        if (!path) {
            return BASE_URL;
        }
        if (/^https?:\/\//i.test(path)) {
            return path;
        }
        return `${BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
    }

    function setHidden(element, hidden) {
        if (!element) {
            return;
        }
        element.classList.toggle("is-hidden", hidden);
    }

    function numericValue(value) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function formatValue(input, value) {
        const parsed = numericValue(value);
        if (parsed === null) {
            return "-";
        }
        if (input.dataset.valueKind === "amount") {
            return parsed.toLocaleString("en-US", { maximumFractionDigits: 0 });
        }
        const decimals = input.dataset.decimals === "" ? 2 : Number(input.dataset.decimals || 2);
        return parsed.toFixed(decimals);
    }

    function updateLiveValue(input) {
        const targetId = input.dataset.displayTarget;
        if (!targetId) {
            return;
        }
        const target = document.getElementById(targetId);
        if (!target) {
            return;
        }
        target.textContent = formatValue(input, input.value);
    }

    function syncSliderFromInput(input) {
        const slider = slidersByTarget.get(input.id);
        if (!slider) {
            return;
        }
        const parsed = numericValue(input.value);
        if (parsed === null) {
            return;
        }
        const min = numericValue(slider.min);
        const max = numericValue(slider.max);
        let nextValue = parsed;
        if (min !== null) {
            nextValue = Math.max(min, nextValue);
        }
        if (max !== null) {
            nextValue = Math.min(max, nextValue);
        }
        slider.value = String(nextValue);
    }

    function syncInputFromSlider(slider) {
        const targetInput = document.getElementById(slider.dataset.targetInput || "");
        if (!targetInput) {
            return;
        }
        targetInput.value = slider.value;
        updateLiveValue(targetInput);
    }

    function changedPayload() {
        const changes = {};
        inputs.forEach((input) => {
            const current = String(input.value || "").trim();
            const baseline = baselineValues.get(input.name) || "";
            const currentNumber = numericValue(current);
            const baselineNumber = numericValue(baseline);
            const numericChanged = currentNumber !== null
                && baselineNumber !== null
                && Math.abs(currentNumber - baselineNumber) > 1e-9;
            if (current && (numericChanged || (currentNumber === null && current !== baseline))) {
                changes[input.name] = current;
            }
        });
        return changes;
    }

    function setBusy(isBusy) {
        if (runButton) {
            runButton.disabled = isBusy;
        }
        if (resetButton) {
            resetButton.disabled = isBusy;
        }
        inputs.forEach((input) => {
            input.disabled = isBusy;
        });
        sliders.forEach((slider) => {
            slider.disabled = isBusy;
        });
        setHidden(loadingState, !isBusy);
    }

    function renderChanges(changes) {
        const list = document.getElementById("sim-change-list");
        if (!list) {
            return;
        }
        if (!Array.isArray(changes) || changes.length === 0) {
            list.innerHTML = '<div class="report-empty-state compact"><h3>No scenario delta</h3><p>The simulator did not detect any effective field changes.</p></div>';
            return;
        }
        list.innerHTML = changes.map((change) => `
            <article class="report-sim-change-card">
                <div class="report-sim-change-topline">
                    <strong>${change.label}</strong>
                </div>
                <div class="report-sim-change-values">
                    <span>${change.before}</span>
                    <span class="report-sim-change-arrow">&rarr;</span>
                    <span>${change.after}</span>
                </div>
                <p>${change.field}</p>
            </article>
        `).join("");
    }

    function setStateClass(element, statePrefix, stateValue) {
        if (!element) {
            return;
        }
        Array.from(element.classList)
            .filter((className) => className.startsWith(statePrefix))
            .forEach((className) => element.classList.remove(className));
        element.classList.add(`${statePrefix}${stateValue}`);
    }

    function renderResult(result) {
        document.getElementById("sim-original-probability").textContent = result.original_probability_text || simulatorConfig.originalProbabilityText || "-";
        document.getElementById("sim-original-decision").textContent = result.original_decision_label || simulatorConfig.originalDecisionLabel || "Pending";
        document.getElementById("sim-probability").textContent = result.simulated_probability_text || "-";
        document.getElementById("sim-decision").textContent = result.simulated_decision_label || "Pending";
        document.getElementById("sim-delta").textContent = result.delta_text || "-";
        document.getElementById("sim-risk-movement").textContent = result.risk_movement_label || "Risk movement unavailable";
        document.getElementById("sim-change-count").textContent = `${result.change_count || 0} change${result.change_count === 1 ? "" : "s"} applied`;

        const deltaCard = document.getElementById("sim-delta-card");
        if (deltaCard) {
            deltaCard.classList.remove("delta-up", "delta-down", "delta-flat");
            deltaCard.classList.add(`delta-${result.delta_direction || "flat"}`);
        }

        const decisionCard = document.getElementById("sim-decision-card");
        if (decisionCard) {
            setStateClass(decisionCard, "state-", result.delta_direction || "flat");
        }

        const decisionBadge = document.getElementById("sim-decision-delta-badge");
        if (decisionBadge) {
            decisionBadge.textContent = result.decision_delta_label || "Decision unchanged";
            decisionBadge.classList.remove("approve", "review", "decline");
            decisionBadge.classList.add(result.simulated_decision_badge_class || "review");
        }

        renderChanges(result.changed_features || []);
        if (reportChatContext) {
            reportChatContext.simulator_result = {
                original_probability_text: result.original_probability_text || simulatorConfig.originalProbabilityText || "-",
                original_decision_label: result.original_decision_label || simulatorConfig.originalDecisionLabel || "Pending",
                simulated_probability_text: result.simulated_probability_text || "-",
                simulated_decision_label: result.simulated_decision_label || "Pending",
                simulated_decision_badge_class: result.simulated_decision_badge_class || "review",
                delta_text: result.delta_text || "-",
                delta_direction: result.delta_direction || "flat",
                risk_movement_label: result.risk_movement_label || "Risk movement unavailable",
                decision_delta_label: result.decision_delta_label || "Decision unchanged",
                change_count: result.change_count || 0,
                changed_features: Array.isArray(result.changed_features) ? result.changed_features : [],
            };
        }
        setHidden(resultCard, false);
    }

    async function runSimulation() {
        const changes = changedPayload();
        setHidden(errorState, true);
        errorState.textContent = "";

        if (Object.keys(changes).length === 0) {
            errorState.textContent = "Adjust at least one simulator field before running a scenario.";
            setHidden(errorState, false);
            setHidden(resultCard, true);
            return;
        }

        setBusy(true);
        try {
            const response = await fetch(resolveApiUrl(simulatorConfig.endpoint), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body: JSON.stringify({ changes }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.message || "Simulation failed.");
            }
            renderResult(payload);
        } catch (error) {
            errorState.textContent = error instanceof Error ? error.message : "Simulation failed.";
            setHidden(errorState, false);
            setHidden(resultCard, true);
        } finally {
            setBusy(false);
        }
    }

    function resetSimulation() {
        inputs.forEach((input) => {
            input.value = input.defaultValue;
            syncSliderFromInput(input);
            updateLiveValue(input);
        });
        setHidden(errorState, true);
        errorState.textContent = "";
        setHidden(resultCard, true);
        if (reportChatContext) {
            reportChatContext.simulator_result = null;
        }
    }

    inputs.forEach((input) => {
        updateLiveValue(input);
        input.addEventListener("input", () => {
            syncSliderFromInput(input);
            updateLiveValue(input);
        });
    });

    sliders.forEach((slider) => {
        slider.addEventListener("input", () => {
            syncInputFromSlider(slider);
        });
    });

    if (runButton) {
        runButton.addEventListener("click", runSimulation);
    }
    if (resetButton) {
        resetButton.addEventListener("click", resetSimulation);
    }
})();
