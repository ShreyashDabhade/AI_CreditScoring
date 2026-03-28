(function () {
    const simulatorConfig = window.__REPORT_SIMULATOR__;
    const root = document.getElementById("what-if-simulator-root");
    if (!simulatorConfig || !root) {
        return;
    }

    const inputs = Array.from(root.querySelectorAll("[data-sim-field]"));
    const runButton = document.getElementById("simulate-run-button");
    const resetButton = document.getElementById("simulate-reset-button");
    const loadingState = document.getElementById("simulate-loading-state");
    const errorState = document.getElementById("simulate-error-state");
    const resultCard = document.getElementById("simulate-result-card");
    const baselineValues = new Map(inputs.map((input) => [input.name, String(input.defaultValue || "").trim()]));

    function setHidden(element, hidden) {
        if (!element) {
            return;
        }
        element.classList.toggle("is-hidden", hidden);
    }

    function changedPayload() {
        const changes = {};
        inputs.forEach((input) => {
            const current = String(input.value || "").trim();
            const baseline = baselineValues.get(input.name) || "";
            if (current && current !== baseline) {
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

    function renderResult(result) {
        document.getElementById("sim-original-probability").textContent = result.original_probability_text || simulatorConfig.originalProbabilityText || "-";
        document.getElementById("sim-original-decision").textContent = result.original_decision_label || simulatorConfig.originalDecisionLabel || "Pending";
        document.getElementById("sim-probability").textContent = result.simulated_probability_text || "-";
        document.getElementById("sim-decision").textContent = result.simulated_decision_label || "Pending";
        document.getElementById("sim-delta").textContent = result.delta_text || "-";
        document.getElementById("sim-change-count").textContent = `${result.change_count || 0} change${result.change_count === 1 ? "" : "s"} applied`;

        const deltaCard = document.getElementById("sim-delta-card");
        if (deltaCard) {
            deltaCard.classList.remove("delta-up", "delta-down", "delta-flat");
            deltaCard.classList.add(`delta-${result.delta_direction || "flat"}`);
        }

        renderChanges(result.changed_features || []);
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
            const response = await fetch(simulatorConfig.endpoint, {
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
        });
        setHidden(errorState, true);
        errorState.textContent = "";
        setHidden(resultCard, true);
    }

    if (runButton) {
        runButton.addEventListener("click", runSimulation);
    }
    if (resetButton) {
        resetButton.addEventListener("click", resetSimulation);
    }
})();
