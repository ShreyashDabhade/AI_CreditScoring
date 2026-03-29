/**
 * MasterMind AI — Credit Advisor Chatbot (v2)
 *
 * - Auto-detects score context from the current page
 * - Users can paste JSON data directly
 * - Sends page location so the bot knows where the user is
 * - Maintains conversation history for multi-turn chat
 */
(function () {
    'use strict';

    var BASE_URL = window.__API_BASE_URL__ || window.location.origin || '';

    var widget   = document.getElementById('chat-widget');
    var toggle   = document.getElementById('chat-toggle');
    var closeBtn = document.getElementById('chat-close');
    var form     = document.getElementById('chat-form');
    var input    = document.getElementById('chat-input');
    var messages = document.getElementById('chat-messages');
    var typing   = document.getElementById('chat-typing');
    var status   = document.getElementById('chat-status');

    if (!widget || !toggle || !form || !input || !messages) return;

    var conversationHistory = [];
    var isOpen = false;
    var isSending = false;

    function _resolveApiUrl(path) {
        if (!path) return BASE_URL;
        if (/^https?:\/\//i.test(path)) return path;
        return BASE_URL + (path.charAt(0) === '/' ? path : '/' + path);
    }

    function _getReportContext() {
        if (window.__REPORT_CHAT_CONTEXT__ && typeof window.__REPORT_CHAT_CONTEXT__ === 'object') {
            return window.__REPORT_CHAT_CONTEXT__;
        }
        return null;
    }

    // ── Auto-detect score context from the current page ─────────
    function _getPageContext() {
        var ctx = {
            current_page: window.location.pathname,
            page_title: document.title
        };

        // Try to get health snapshot
        var health = window.__MASTERMIND_HEALTH_SNAPSHOT__ || {};
        if (health.model_version) {
            ctx.model_version = health.model_version;
            ctx.fairness_audit_passed = health.fairness_audit_passed;
        }

        var reportContext = _getReportContext();
        if (reportContext) {
            ctx.report_scope = reportContext.scope || 'analyst_application_report';
            ctx.application_id = reportContext.application_id || null;
            ctx.has_simulator_scenario = !!reportContext.simulator_result;
        }

        return ctx;
    }

    function _getScoreData() {
        var reportContext = _getReportContext();
        if (reportContext && reportContext.latest_assessment) {
            var latest = reportContext.latest_assessment;
            return {
                probability_of_default: latest.calibrated_probability_text || latest.calibrated_probability || null,
                decision: latest.decision_label || latest.decision || null,
                model_version: latest.model_version || null,
            };
        }

        // Method 1: Look for report data in the simulator config
        var simConfig = window.__REPORT_SIMULATOR__;
        if (simConfig) {
            var scoreData = {
                probability_of_default: simConfig.originalProbabilityText,
                decision: simConfig.originalDecisionLabel,
            };
            return scoreData;
        }

        // Method 2: Extract from visible DOM elements on report pages
        var probEl = document.querySelector('.report-hero-probability');
        var decisionEl = document.querySelector('.decision-badge');
        if (probEl || decisionEl) {
            return {
                probability_of_default: probEl ? probEl.textContent.trim() : null,
                decision: decisionEl ? decisionEl.textContent.trim() : null,
            };
        }

        // Method 3: Look for result-panel on analyze page
        var resultPanel = document.querySelector('.result-panel');
        if (resultPanel) {
            var pd = resultPanel.querySelector('[data-result-pd]');
            var dec = resultPanel.querySelector('[data-result-decision]');
            return {
                probability_of_default: pd ? pd.textContent.trim() : null,
                decision: dec ? dec.textContent.trim() : null,
            };
        }

        return null;
    }

    function _getShapValues() {
        var reportContext = _getReportContext();
        if (reportContext && Array.isArray(reportContext.top_drivers) && reportContext.top_drivers.length > 0) {
            return reportContext.top_drivers.map(function (item) {
                return {
                    feature: item.feature || 'Unknown',
                    reason: item.reason || ''
                };
            });
        }

        var shapEls = document.querySelectorAll('.report-driver-card');
        if (shapEls.length === 0) return null;

        var values = [];
        shapEls.forEach(function (el) {
            var h3 = el.querySelector('h3');
            var p = el.querySelector('.report-driver-topline p');
            if (h3) {
                values.push({
                    feature: h3.textContent.trim(),
                    reason: p ? p.textContent.trim() : ''
                });
            }
        });
        return values.length > 0 ? values : null;
    }

    // ── Toggle ──────────────────────────────────────────────────
    function _open() {
        isOpen = true;
        widget.setAttribute('data-state', 'open');
        input.focus();
    }
    function _close() {
        isOpen = false;
        widget.setAttribute('data-state', 'collapsed');
    }

    toggle.addEventListener('click', function () { isOpen ? _close() : _open(); });
    closeBtn.addEventListener('click', _close);

    // ── Render ──────────────────────────────────────────────────
    function _escapeHtml(t) {
        var d = document.createElement('div');
        d.appendChild(document.createTextNode(t));
        return d.innerHTML;
    }

    function _renderMarkdown(text) {
        var html = _escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\n• /g, '</p><p>• ');
        html = html.replace(/\n- /g, '</p><p>• ');
        html = html.replace(/\n(\d+)\. /g, '</p><p>$1. ');
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        return '<p>' + html + '</p>';
    }

    function _appendMessage(role, content) {
        var isBot = role === 'bot';
        var el = document.createElement('div');
        el.className = 'chat-message chat-message-' + role;

        var avatar = document.createElement('div');
        avatar.className = 'chat-message-avatar';
        avatar.innerHTML = isBot
            ? '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>';

        var body = document.createElement('div');
        body.className = 'chat-message-content';
        body.innerHTML = _renderMarkdown(content);

        el.appendChild(avatar);
        el.appendChild(body);

        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        messages.appendChild(el);
        requestAnimationFrame(function () {
            el.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        });
        messages.scrollTop = messages.scrollHeight;
    }

    function _showTyping() { typing.style.display = 'flex'; messages.scrollTop = messages.scrollHeight; }
    function _hideTyping() { typing.style.display = 'none'; }

    // ── Send ────────────────────────────────────────────────────
    async function _sendMessage(text) {
        if (isSending || !text.trim()) return;
        isSending = true;

        _appendMessage('user', text);
        conversationHistory.push({ role: 'user', content: text });
        input.value = '';
        input.disabled = true;
        _showTyping();

        try {
            var response = await fetch(_resolveApiUrl('/api/chat'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    context: {
                        score_data: _getScoreData(),
                        shap_values: _getShapValues(),
                        page_context: _getPageContext(),
                        report_context: _getReportContext(),
                    },
                    history: conversationHistory.slice(-8),
                }),
            });

            var data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Request failed');

            var botText = data.response || 'Sorry, I could not process that.';
            _appendMessage('bot', botText);
            conversationHistory.push({ role: 'bot', content: botText });

            if (data.source === 'gemini') {
                status.innerHTML = '<span class="chat-status-dot chat-status-gemini"></span>Powered by Gemini AI';
            } else {
                status.innerHTML = '<span class="chat-status-dot"></span>Smart Advisor (set GEMINI_API_KEY for full AI)';
            }
        } catch (err) {
            _appendMessage('bot', 'Connection error. Make sure the server is running and try again.');
            console.error('[chatbot]', err);
        } finally {
            _hideTyping();
            input.disabled = false;
            input.focus();
            isSending = false;
        }
    }

    form.addEventListener('submit', function (e) { e.preventDefault(); _sendMessage(input.value); });

    // Ctrl+Shift+/ to toggle
    document.addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.shiftKey && e.key === '/') { e.preventDefault(); isOpen ? _close() : _open(); }
    });
})();
