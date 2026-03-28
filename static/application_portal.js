(function () {
    const config = window.__MASTERMIND_UI__;
    const page = document.body.dataset.page || "";
    if (!config || page !== "applications") {
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
        tier: "REDUCED",
    };

    function $(id) {
        return document.getElementById(id);
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

    function allPortalFields() {
        return Array.from(document.querySelectorAll("[data-portal-field]"));
    }

    function sectionFields(sectionName) {
        return allPortalFields().filter((field) => field.dataset.section === sectionName);
    }

    function visibleRequiredFields() {
        return allPortalFields().filter((field) => {
            if (field.dataset.required !== "true") {
                return false;
            }
            if (state.tier === "REDUCED" && field.dataset.section !== "application") {
                return false;
            }
            return true;
        });
    }

    function clone(value) {
        return JSON.parse(JSON.stringify(value || {}));
    }

    function setTier(nextTier) {
        state.tier = nextTier;
        document.querySelectorAll("[data-portal-tier-toggle]").forEach((button) => {
            const active = button.dataset.tier === nextTier;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
        });

        $("portal-tier-input").value = nextTier;
        $("portal-full-panel").classList.toggle("is-hidden", nextTier !== "FULL");
        $("portal-tier-description").textContent = nextTier === "FULL"
            ? "Application fields plus five optional aggregate families. Blank aggregate values are saved as explicit zeros."
            : "Application-only intake. Best for fast submission when aggregate credit history is unavailable.";
        $("portal-tier-hint").textContent = nextTier === "FULL"
            ? "FULL selected. Aggregate drawers are now available."
            : "REDUCED selected by default";

        refreshPortalState();
    }

    function markFieldValidity(control, valid) {
        control.classList.toggle("invalid", !valid);
        if (valid) {
            control.removeAttribute("aria-invalid");
            return;
        }
        control.setAttribute("aria-invalid", "true");
    }

    function buildPayload() {
        const payload = clone((config.samplePayloads || {})[state.tier] || {});
        const missing = [];
        const aggregateDefaults = [];

        payload.application = payload.application || {};
        sectionFields("application").forEach((field) => {
            const value = valueFromControl(field);
            const valid = value !== null && value !== "";
            markFieldValidity(field, valid);
            if (!valid) {
                missing.push(field);
                return;
            }
            payload.application[field.name] = value;
        });

        if (state.tier === "FULL") {
            FULL_OPTIONAL_SECTIONS.forEach((sectionName) => {
                payload[sectionName] = {};
                sectionFields(sectionName).forEach((field) => {
                    const value = valueFromControl(field);
                    if (value === null || value === "") {
                        payload[sectionName][field.name] = 0;
                        aggregateDefaults.push(`${sectionName}.${field.name}`);
                        markFieldValidity(field, true);
                        return;
                    }
                    payload[sectionName][field.name] = value;
                    markFieldValidity(field, true);
                });
            });
        }

        return {
            payload,
            missing,
            aggregateDefaults,
        };
    }

    function updateCompletion() {
        const applicantName = $("applicant_name");
        const metaComplete = String(applicantName.value || "").trim().length > 0;
        $("portal-intake-completion").textContent = metaComplete
            ? "Applicant identity captured"
            : "Applicant name still required";
        $("portal-meta-status").textContent = metaComplete ? "Ready" : "Incomplete";

        const applicationFields = sectionFields("application");
        const applicationFilled = applicationFields.filter((field) => valueFromControl(field) !== null).length;
        $("portal-application-completion").textContent = `${applicationFilled}/${applicationFields.length} required fields complete`;
        $("portal-application-status").textContent = applicationFilled === applicationFields.length ? "Ready" : "Incomplete";

        if (state.tier === "FULL") {
            let totalDefaults = 0;
            FULL_OPTIONAL_SECTIONS.forEach((sectionName) => {
                const fields = sectionFields(sectionName);
                const filled = fields.filter((field) => valueFromControl(field) !== null).length;
                const defaults = fields.length - filled;
                totalDefaults += defaults;
                const completion = $(`completion-${sectionName}`);
                if (completion) {
                    completion.textContent = `${filled}/${fields.length} filled • ${defaults} default to 0`;
                }
            });
            $("portal-full-summary").textContent = `${totalDefaults} aggregate fields currently default to 0`;
            $("portal-defaults-status").textContent = totalDefaults > 0
                ? `${totalDefaults} defaults active`
                : "All aggregate fields supplied";
        } else {
            FULL_OPTIONAL_SECTIONS.forEach((sectionName) => {
                const completion = $(`completion-${sectionName}`);
                if (completion) {
                    completion.textContent = `${sectionFields(sectionName).length} optional fields`;
                }
            });
            $("portal-full-summary").textContent = "Optional sections hidden";
            $("portal-defaults-status").textContent = "Not active";
        }

        const formReady = metaComplete && applicationFilled === applicationFields.length;
        $("portal-readiness-pill").textContent = formReady
            ? "Ready to save"
            : "Complete required items";
    }

    function refreshPortalState() {
        updateCompletion();
    }

    function initGenerateSkId() {
        const button = $("generate-sk-id");
        const input = $("sk_id_curr");
        if (!button || !input) {
            return;
        }
        button.addEventListener("click", function () {
            input.value = String(Date.now()).slice(-9);
            refreshPortalState();
        });
    }

    function initFormSubmit() {
        const form = $("application-portal-form");
        const feedback = $("portal-feedback");
        const hiddenPayload = $("application_payload_json");
        if (!form || !feedback || !hiddenPayload) {
            return;
        }

        form.addEventListener("submit", function (event) {
            feedback.textContent = "";
            const applicantName = $("applicant_name");
            const applicantReady = String(applicantName.value || "").trim().length > 0;
            markFieldValidity(applicantName, applicantReady);

            const built = buildPayload();
            if (!applicantReady || built.missing.length > 0) {
                event.preventDefault();
                feedback.textContent = "Complete all required applicant and application fields before saving.";
                $("portal-readiness-pill").textContent = "Validation blocked submission";
                return;
            }

            hiddenPayload.value = JSON.stringify(built.payload);
            feedback.textContent = state.tier === "FULL" && built.aggregateDefaults.length > 0
                ? `${built.aggregateDefaults.length} omitted FULL fields will be stored as explicit 0 values.`
                : "Application is ready and will be saved to the review queue.";
        });
    }

    function initRealtimeValidation() {
        const controls = [$("applicant_name"), $("sk_id_curr")].concat(allPortalFields());
        controls.forEach((control) => {
            if (!control) {
                return;
            }
            const handler = function () {
                if (control.id === "applicant_name") {
                    markFieldValidity(control, String(control.value || "").trim().length > 0);
                } else if (control.hasAttribute("data-portal-field") && control.dataset.required === "true") {
                    markFieldValidity(control, valueFromControl(control) !== null);
                } else {
                    markFieldValidity(control, true);
                }
                refreshPortalState();
            };
            control.addEventListener("input", handler);
            control.addEventListener("change", handler);
        });
    }

    function initTierToggle() {
        document.querySelectorAll("[data-portal-tier-toggle]").forEach((button) => {
            button.addEventListener("click", function () {
                setTier(button.dataset.tier || "REDUCED");
            });
        });
    }

    initTierToggle();
    initGenerateSkId();
    initRealtimeValidation();
    initFormSubmit();
    setTier("REDUCED");
})();
