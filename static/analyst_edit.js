(function () {
    const page = document.body.dataset.page || "";
    if (page !== "analyst") {
        return;
    }

    const form = document.getElementById("analyst-edit-form");
    if (!form) {
        return;
    }

    const FULL_OPTIONAL_SECTIONS = [
        "bureau_agg",
        "previous_agg",
        "installments_agg",
        "pos_cash_agg",
        "credit_card_agg",
    ];
    const state = {
        tier: document.getElementById("analyst-tier-input")?.value || "REDUCED",
    };

    function $(id) {
        return document.getElementById(id);
    }

    function allFields() {
        return Array.from(document.querySelectorAll("[data-portal-field]"));
    }

    function sectionFields(sectionName) {
        return allFields().filter((field) => field.dataset.section === sectionName);
    }

    function valueFromControl(control) {
        const rawValue = String(control.value || "").trim();
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

    function displayedValue(control) {
        return String(control.value || "").trim();
    }

    function markValidity(control, valid) {
        control.classList.toggle("invalid", !valid);
        control.setAttribute("aria-invalid", valid ? "false" : "true");
    }

    function setTier(nextTier) {
        state.tier = nextTier;
        $("analyst-tier-input").value = nextTier;
        document.querySelectorAll("[data-analyst-tier-toggle]").forEach((button) => {
            const active = button.dataset.tier === nextTier;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
        });
        $("analyst-full-panel").classList.toggle("is-hidden", nextTier !== "FULL");
        $("analyst-tier-hint").textContent = `${nextTier} currently selected`;
        $("analyst-tier-description").textContent = nextTier === "FULL"
            ? "FULL preserves the full payload contract and keeps omitted aggregate fields as explicit 0 values."
            : "REDUCED preserves the application-only payload contract with required borrower fields only.";
        refreshState();
    }

    function fieldChanged(control) {
        return displayedValue(control) !== String(control.dataset.originalValue || "");
    }

    function buildPayload() {
        const payload = { application: {} };
        const missing = [];
        const changes = [];
        const aggregateDefaults = [];

        sectionFields("application").forEach((field) => {
            const value = valueFromControl(field);
            const valid = value !== null;
            markValidity(field, valid);
            if (!valid) {
                missing.push(field);
                return;
            }
            payload.application[field.name] = value;
            if (fieldChanged(field)) {
                changes.push(field.dataset.fieldPath || field.name);
            }
        });

        if (state.tier === "FULL") {
            FULL_OPTIONAL_SECTIONS.forEach((sectionName) => {
                payload[sectionName] = {};
                sectionFields(sectionName).forEach((field) => {
                    const value = valueFromControl(field);
                    if (value === null) {
                        payload[sectionName][field.name] = 0;
                        aggregateDefaults.push(field.dataset.fieldPath || `${sectionName}.${field.name}`);
                    } else {
                        payload[sectionName][field.name] = value;
                    }
                    if (fieldChanged(field)) {
                        changes.push(field.dataset.fieldPath || field.name);
                    }
                    markValidity(field, true);
                });
            });
        }

        return {
            payload,
            missing,
            changes,
            aggregateDefaults,
        };
    }

    function refreshState() {
        const applicantName = $("analyst_applicant_name");
        const status = $("analyst_current_status");
        const metaFields = [applicantName, $("analyst_sk_id_curr"), status].filter(Boolean);
        let modifiedCount = 0;

        metaFields.forEach((field) => {
            const changed = fieldChanged(field);
            field.classList.toggle("modified", changed);
            if (changed) {
                modifiedCount += 1;
            }
        });

        const built = buildPayload();
        allFields().forEach((field) => {
            const changed = fieldChanged(field);
            field.classList.toggle("modified", changed);
            if (changed) {
                modifiedCount += 1;
            }
        });

        const requiredReady = String(applicantName.value || "").trim().length > 0 && built.missing.length === 0;
        $("analyst-required-status").textContent = requiredReady ? "Ready" : "Incomplete";
        $("analyst-meta-completion").textContent = modifiedCount > 0 ? `${modifiedCount} changes detected` : "No edits detected yet";
        $("analyst-change-pill").textContent = modifiedCount > 0 ? `${modifiedCount} field changes pending save` : "No field changes yet";
        $("analyst-modified-count").textContent = String(modifiedCount);

        const applicationFields = sectionFields("application");
        const completeCount = applicationFields.filter((field) => valueFromControl(field) !== null).length;
        $("analyst-application-completion").textContent = `${completeCount}/${applicationFields.length} required fields complete`;

        if (state.tier === "FULL") {
            let totalDefaults = 0;
            FULL_OPTIONAL_SECTIONS.forEach((sectionName) => {
                const fields = sectionFields(sectionName);
                const filled = fields.filter((field) => valueFromControl(field) !== null).length;
                const defaults = fields.length - filled;
                totalDefaults += defaults;
                const completion = $(`analyst-completion-${sectionName}`);
                if (completion) {
                    completion.textContent = `${filled}/${fields.length} filled • ${defaults} default to 0`;
                }
            });
            $("analyst-full-summary").textContent = `${totalDefaults} aggregate fields currently default to 0`;
            $("analyst-defaults-status").textContent = totalDefaults > 0 ? `${totalDefaults} defaults active` : "All aggregate fields supplied";
        } else {
            $("analyst-full-summary").textContent = "FULL aggregates hidden in REDUCED mode";
            $("analyst-defaults-status").textContent = "Not active";
        }

        const changeList = $("analyst-change-list");
        if (built.changes.length > 0 || metaFields.some(fieldChanged)) {
            const entries = [];
            metaFields.forEach((field) => {
                if (fieldChanged(field)) {
                    entries.push(field.name);
                }
            });
            built.changes.forEach((entry) => entries.push(entry));
            changeList.innerHTML = `<ul class="analyst-inline-change-list">${entries.slice(0, 12).map((entry) => `<li>${entry}</li>`).join("")}</ul>`;
        } else {
            changeList.innerHTML = '<p class="field-hint">No field changes have been made in this session yet.</p>';
        }
    }

    document.querySelectorAll("[data-analyst-tier-toggle]").forEach((button) => {
        button.addEventListener("click", function () {
            setTier(button.dataset.tier || "REDUCED");
        });
    });

    [$("analyst_applicant_name"), $("analyst_sk_id_curr"), $("analyst_current_status")]
        .concat(allFields())
        .forEach((control) => {
            if (!control) {
                return;
            }
            const handler = function () {
                if (control.id === "analyst_applicant_name") {
                    markValidity(control, String(control.value || "").trim().length > 0);
                }
                refreshState();
            };
            control.addEventListener("input", handler);
            control.addEventListener("change", handler);
        });

    form.addEventListener("submit", function (event) {
        const feedback = $("analyst-edit-feedback");
        feedback.textContent = "";
        const applicantName = $("analyst_applicant_name");
        const nameValid = String(applicantName.value || "").trim().length > 0;
        markValidity(applicantName, nameValid);

        const built = buildPayload();
        if (!nameValid || built.missing.length > 0) {
            event.preventDefault();
            feedback.textContent = "Complete all required borrower and application fields before saving.";
            return;
        }

        $("analyst_application_payload_json").value = JSON.stringify(built.payload);
        feedback.textContent = built.changes.length > 0 || fieldChanged(applicantName) || fieldChanged($("analyst_sk_id_curr")) || fieldChanged($("analyst_current_status"))
            ? "Updates are ready to save. A field-level change summary will be recorded."
            : "No field changes detected. Saving will preserve the current record.";
    });

    setTier(state.tier);
})();
