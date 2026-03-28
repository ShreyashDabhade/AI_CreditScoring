(function () {
    const config = window.__REPORT_COPILOT__;
    const reportContext = window.__REPORT_CHAT_CONTEXT__;
    const root = document.getElementById("report-copilot-root");

    if (!config || !root || !reportContext || typeof reportContext !== "object") {
        return;
    }

    const toggleButton = document.getElementById("report-copilot-toggle");
    const body = document.getElementById("report-copilot-body");
    const status = document.getElementById("report-copilot-status");
    const messages = document.getElementById("report-copilot-messages");
    const typing = document.getElementById("report-copilot-typing");
    const errorState = document.getElementById("report-copilot-error");
    const form = document.getElementById("report-copilot-form");
    const input = document.getElementById("report-copilot-input");
    const sendButton = document.getElementById("report-copilot-send");
    const promptButtons = Array.from(root.querySelectorAll("[data-copilot-prompt]"));

    if (!body || !status || !messages || !typing || !form || !input || !sendButton) {
        return;
    }

    const conversationHistory = [];
    let isSending = false;

    function setHidden(element, hidden) {
        if (!element) {
            return;
        }
        element.classList.toggle("is-hidden", hidden);
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    function renderMarkdown(text) {
        let html = escapeHtml(text || "");
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/\n(\d+)\. /g, "</p><p>$1. ");
        html = html.replace(/\n- /g, "</p><p>- ");
        html = html.replace(/\n\n/g, "</p><p>");
        html = html.replace(/\n/g, "<br>");
        return "<p>" + html + "</p>";
    }

    function appendMessage(role, content) {
        const message = document.createElement("div");
        message.className = `report-copilot-message report-copilot-message-${role}`;

        const avatar = document.createElement("div");
        avatar.className = "report-copilot-avatar";
        avatar.textContent = role === "bot" ? "AI" : "You";

        const bubble = document.createElement("div");
        bubble.className = "report-copilot-bubble";
        bubble.innerHTML = renderMarkdown(content);

        message.appendChild(avatar);
        message.appendChild(bubble);
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }

    function buildRequestContext() {
        const latest = reportContext.latest_assessment || {};
        const topDrivers = Array.isArray(reportContext.top_drivers) ? reportContext.top_drivers : [];

        return {
            score_data: {
                probability_of_default: latest.calibrated_probability_text || latest.calibrated_probability || null,
                decision: latest.decision_label || latest.decision || null,
                model_version: latest.model_version || null,
            },
            shap_values: topDrivers.map((item) => ({
                feature: item.feature || "Unknown",
                reason: item.reason || "",
            })),
            page_context: {
                current_page: window.location.pathname,
                page_title: document.title,
                report_scope: reportContext.scope || "analyst_application_report",
                application_id: reportContext.application_id || null,
                has_simulator_scenario: !!reportContext.simulator_result,
            },
            report_context: reportContext,
        };
    }

    function setStatus(label, statusClass) {
        status.className = "report-copilot-statusline";
        if (statusClass) {
            status.classList.add(statusClass);
        }
        status.innerHTML = `<span class="report-copilot-status-dot"></span>${label}`;
    }

    function setBusy(busy) {
        isSending = busy;
        input.disabled = busy;
        sendButton.disabled = busy;
        promptButtons.forEach((button) => {
            button.disabled = busy;
        });
        setHidden(typing, !busy);
    }

    async function refreshAvailability() {
        try {
            const response = await fetch(config.healthEndpoint, {
                headers: { Accept: "application/json" },
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.message || "Copilot health check failed.");
            }
            if (payload.gemini_available) {
                setStatus("Grounded copilot ready with Gemini support", "is-gemini");
            } else {
                setStatus("Grounded copilot ready in fallback mode", "is-fallback");
            }
        } catch (error) {
            console.error("[report-copilot] availability check failed", error);
            setStatus("Copilot availability check failed. Grounded fallback may still respond.", "is-fallback");
        }
    }

    async function sendMessage(messageText) {
        const text = String(messageText || "").trim();
        if (!text || isSending) {
            return;
        }

        setHidden(errorState, true);
        errorState.textContent = "";
        appendMessage("user", text);
        conversationHistory.push({ role: "user", content: text });
        input.value = "";
        setBusy(true);

        try {
            const response = await fetch(config.endpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify({
                    message: text,
                    context: buildRequestContext(),
                    history: conversationHistory.slice(-8),
                }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.message || "Copilot request failed.");
            }

            const reply = payload.response || "I could not generate a grounded response for this report.";
            appendMessage("bot", reply);
            conversationHistory.push({ role: "bot", content: reply });

            if (payload.source === "gemini") {
                setStatus("Grounded copilot answered with Gemini", "is-gemini");
            } else {
                setStatus("Grounded copilot answered in fallback mode", "is-fallback");
            }
        } catch (error) {
            console.error("[report-copilot] send failed", error);
            errorState.textContent = error instanceof Error
                ? error.message
                : "The copilot could not answer right now.";
            setHidden(errorState, false);
        } finally {
            setBusy(false);
            input.focus();
        }
    }

    if (toggleButton) {
        toggleButton.addEventListener("click", () => {
            const collapsed = root.classList.toggle("is-collapsed");
            toggleButton.setAttribute("aria-expanded", collapsed ? "false" : "true");
            toggleButton.textContent = collapsed ? "Show" : "Hide";
        });
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        sendMessage(input.value);
    });

    promptButtons.forEach((button) => {
        button.addEventListener("click", () => {
            sendMessage(button.textContent || "");
        });
    });

    refreshAvailability();
})();
