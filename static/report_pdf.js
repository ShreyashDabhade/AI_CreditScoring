/**
 * MasterMind AI — PDF Credit Assessment Report Generator
 *
 * Uses jsPDF to generate a branded, professional PDF report
 * containing applicant details, credit score, SHAP risk factors,
 * and fairness audit status.
 */

(function () {
    'use strict';

    // ── Constants ───────────────────────────────────────────────────
    var BRAND_DARK    = '#0f1729';
    var BRAND_PRIMARY = '#7F77DD';
    var BRAND_GREEN   = '#1D9E75';
    var BRAND_RED     = '#D85A30';
    var BRAND_AMBER   = '#E5A100';
    var BRAND_GRAY    = '#888780';
    var TEXT_LIGHT    = '#f0f0f0';

    // ── Helpers ─────────────────────────────────────────────────────

    function _hexToRgb(hex) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return [r, g, b];
    }

    function _decisionColor(decision) {
        var upper = (decision || '').toUpperCase();
        if (upper === 'APPROVE') return BRAND_GREEN;
        if (upper === 'DECLINE') return BRAND_RED;
        return BRAND_AMBER;
    }

    function _nowFormatted() {
        var d = new Date();
        var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear() + ', ' +
               String(d.getHours()).padStart(2,'0') + ':' + String(d.getMinutes()).padStart(2,'0');
    }

    // ── Data extraction from page ───────────────────────────────────

    function _extractReportData() {
        var data = {};

        // Applicant info
        var nameEl = document.querySelector('[data-applicant-name]');
        data.applicantName = nameEl ? nameEl.textContent.trim() : 'Applicant';

        var idEl = document.querySelector('[data-application-id]');
        data.applicationId = idEl ? idEl.textContent.trim() : 'N/A';

        // Score data
        var pdEl = document.querySelector('[data-probability]');
        data.probabilityPct = pdEl ? pdEl.textContent.trim() : 'N/A';
        data.probability = pdEl ? parseFloat(pdEl.getAttribute('data-probability')) : null;

        var decisionEl = document.querySelector('[data-decision]');
        data.decision = decisionEl ? decisionEl.getAttribute('data-decision') || decisionEl.textContent.trim() : 'N/A';

        // Model info
        var versionEl = document.querySelector('[data-model-version]');
        data.modelVersion = versionEl ? versionEl.textContent.trim() : 'N/A';

        var tierEl = document.querySelector('[data-coverage-tier]');
        data.coverageTier = tierEl ? tierEl.textContent.trim() : 'N/A';

        var fairnessEl = document.querySelector('[data-fairness-passed]');
        data.fairnessAuditPassed = fairnessEl ? fairnessEl.getAttribute('data-fairness-passed') === 'true' : null;

        // SHAP explanations
        data.shapFactors = [];
        var shapEls = document.querySelectorAll('[data-shap-feature]');
        shapEls.forEach(function (el) {
            data.shapFactors.push({
                feature: el.getAttribute('data-shap-feature') || '',
                reason: el.getAttribute('data-shap-reason') || el.textContent.trim(),
            });
        });

        // Application snapshot
        data.snapshot = [];
        var snapshotEls = document.querySelectorAll('[data-snapshot-label]');
        snapshotEls.forEach(function (el) {
            data.snapshot.push({
                label: el.getAttribute('data-snapshot-label'),
                value: el.getAttribute('data-snapshot-value') || el.textContent.trim(),
            });
        });

        return data;
    }

    // ── PDF generation ──────────────────────────────────────────────

    function _generatePDF() {
        if (typeof window.jspdf === 'undefined') {
            alert('PDF library is still loading. Please try again in a moment.');
            return;
        }

        var data = _extractReportData();
        var jsPDF = window.jspdf.jsPDF;
        var doc = new jsPDF('p', 'mm', 'a4');
        var pageW = doc.internal.pageSize.getWidth();
        var pageH = doc.internal.pageSize.getHeight();
        var margin = 20;
        var contentW = pageW - 2 * margin;
        var y = 0;

        // ── Header band ─────────────────────────────────────────────
        var rgb = _hexToRgb(BRAND_DARK);
        doc.setFillColor(rgb[0], rgb[1], rgb[2]);
        doc.rect(0, 0, pageW, 45, 'F');

        // Brand accent bar
        rgb = _hexToRgb(BRAND_PRIMARY);
        doc.setFillColor(rgb[0], rgb[1], rgb[2]);
        doc.rect(0, 45, pageW, 2, 'F');

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(22);
        doc.setTextColor(255, 255, 255);
        doc.text('MasterMind AI', margin, 20);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(11);
        doc.setTextColor(200, 200, 200);
        doc.text('Credit Assessment Report', margin, 28);
        doc.text('Generated: ' + _nowFormatted(), margin, 36);

        doc.setFontSize(9);
        doc.text('CONFIDENTIAL', pageW - margin - 30, 36);

        y = 55;

        // ── Applicant section ───────────────────────────────────────
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 41);
        doc.text('Applicant Details', margin, y);
        y += 8;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.setTextColor(60, 60, 60);

        var applicantRows = [
            ['Applicant Name', data.applicantName],
            ['Application ID', '#' + data.applicationId],
            ['Coverage Tier', data.coverageTier],
            ['Model Version', data.modelVersion],
        ];

        applicantRows.forEach(function (row) {
            doc.setFont('helvetica', 'bold');
            doc.text(row[0] + ':', margin, y);
            doc.setFont('helvetica', 'normal');
            doc.text(String(row[1]), margin + 45, y);
            y += 6;
        });

        y += 5;

        // ── Score section ───────────────────────────────────────────
        // Decision badge
        var badgeColor = _decisionColor(data.decision);
        rgb = _hexToRgb(badgeColor);
        doc.setFillColor(rgb[0], rgb[1], rgb[2]);
        doc.roundedRect(margin, y, contentW, 28, 3, 3, 'F');

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(18);
        doc.setTextColor(255, 255, 255);
        doc.text('Decision: ' + (data.decision || 'N/A').toUpperCase(), margin + 10, y + 12);

        doc.setFontSize(13);
        doc.text('Probability of Default: ' + data.probabilityPct, margin + 10, y + 22);

        y += 36;

        // ── Score gauge ─────────────────────────────────────────────
        if (data.probability !== null && !isNaN(data.probability)) {
            var gaugeY = y;
            var gaugeH = 8;
            var pct = Math.min(Math.max(data.probability, 0), 1);

            // Background bar
            doc.setFillColor(230, 230, 230);
            doc.roundedRect(margin, gaugeY, contentW, gaugeH, 2, 2, 'F');

            // Green zone (0-15%)
            var greenW = contentW * 0.15;
            rgb = _hexToRgb(BRAND_GREEN);
            doc.setFillColor(rgb[0], rgb[1], rgb[2]);
            doc.roundedRect(margin, gaugeY, greenW, gaugeH, 2, 2, 'F');

            // Amber zone (15-35%)
            var amberW = contentW * 0.20;
            rgb = _hexToRgb(BRAND_AMBER);
            doc.setFillColor(rgb[0], rgb[1], rgb[2]);
            doc.rect(margin + greenW, gaugeY, amberW, gaugeH, 'F');

            // Red zone (35-100%)
            var redW = contentW * 0.65;
            rgb = _hexToRgb(BRAND_RED);
            doc.setFillColor(rgb[0], rgb[1], rgb[2]);
            doc.roundedRect(margin + greenW + amberW, gaugeY, redW, gaugeH, 2, 2, 'F');

            // Pointer marker
            var pointerX = margin + (pct * contentW);
            doc.setFillColor(15, 23, 41);
            doc.triangle(pointerX - 3, gaugeY - 2, pointerX + 3, gaugeY - 2, pointerX, gaugeY + 1, 'F');

            // Labels
            doc.setFontSize(7);
            doc.setTextColor(100, 100, 100);
            doc.text('APPROVE', margin + 2, gaugeY + gaugeH + 5);
            doc.text('REVIEW', margin + greenW + 5, gaugeY + gaugeH + 5);
            doc.text('DECLINE', margin + greenW + amberW + 5, gaugeY + gaugeH + 5);

            y = gaugeY + gaugeH + 12;
        }

        // ── SHAP Risk Factors ───────────────────────────────────────
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 41);
        doc.text('Top Risk Factors (SHAP Explanations)', margin, y);
        y += 8;

        if (data.shapFactors.length > 0) {
            data.shapFactors.forEach(function (factor, i) {
                if (y > pageH - 30) {
                    doc.addPage();
                    y = 20;
                }

                // Rank badge
                rgb = _hexToRgb(BRAND_PRIMARY);
                doc.setFillColor(rgb[0], rgb[1], rgb[2]);
                doc.circle(margin + 4, y - 1, 3, 'F');
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(8);
                doc.setTextColor(255, 255, 255);
                doc.text(String(i + 1), margin + 3, y + 0.5);

                // Feature name
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(10);
                doc.setTextColor(15, 23, 41);
                var featureLabel = factor.feature.replace(/_/g, ' ');
                doc.text(featureLabel, margin + 12, y);
                y += 5;

                // Reason
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(9);
                doc.setTextColor(100, 100, 100);
                var reasonLines = doc.splitTextToSize(factor.reason, contentW - 15);
                doc.text(reasonLines, margin + 12, y);
                y += reasonLines.length * 4.5 + 4;
            });
        } else {
            doc.setFont('helvetica', 'italic');
            doc.setFontSize(9);
            doc.setTextColor(150, 150, 150);
            doc.text('No SHAP explanations available for this assessment.', margin, y);
            y += 8;
        }

        y += 5;

        // ── Application Snapshot ────────────────────────────────────
        if (data.snapshot.length > 0) {
            if (y > pageH - 60) {
                doc.addPage();
                y = 20;
            }

            doc.setFont('helvetica', 'bold');
            doc.setFontSize(13);
            doc.setTextColor(15, 23, 41);
            doc.text('Application Summary', margin, y);
            y += 8;

            data.snapshot.forEach(function (row) {
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(9);
                doc.setTextColor(100, 100, 100);
                doc.text(row.label + ':', margin, y);
                doc.setTextColor(15, 23, 41);
                doc.text(String(row.value), margin + 50, y);
                y += 5;
            });

            y += 5;
        }

        // ── Fairness & Compliance ───────────────────────────────────
        if (y > pageH - 40) {
            doc.addPage();
            y = 20;
        }

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 41);
        doc.text('Fairness & Compliance', margin, y);
        y += 8;

        var fairnessText = data.fairnessAuditPassed === true
            ? 'PASSED — Model has been audited for proxy fairness across demographic subgroups.'
            : data.fairnessAuditPassed === false
            ? 'NOT PASSED — The model fairness audit did not pass all criteria.'
            : 'Status unavailable for this assessment.';

        var fairnessColor = data.fairnessAuditPassed === true ? BRAND_GREEN : BRAND_RED;
        rgb = _hexToRgb(fairnessColor);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        doc.setTextColor(rgb[0], rgb[1], rgb[2]);
        doc.text(fairnessText, margin, y);
        y += 10;

        doc.setTextColor(130, 130, 130);
        doc.setFontSize(8);
        var disclaimerLines = doc.splitTextToSize(
            'This report is generated by the MasterMind AI Credit Scoring System. ' +
            'Scores are model-generated probabilities and should be used alongside human judgment ' +
            'in credit decisions. Protected attributes (gender, ethnicity, religion) are excluded ' +
            'from scoring. This system complies with RBI Fair Practices Code and EU AI Act mandates ' +
            'for algorithmic transparency.',
            contentW
        );
        doc.text(disclaimerLines, margin, y);

        // ── Footer band ─────────────────────────────────────────────
        rgb = _hexToRgb(BRAND_DARK);
        doc.setFillColor(rgb[0], rgb[1], rgb[2]);
        doc.rect(0, pageH - 12, pageW, 12, 'F');

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7);
        doc.setTextColor(180, 180, 180);
        doc.text('MasterMind AI — Explainable Credit Scoring for Financial Inclusion', margin, pageH - 5);
        doc.text('Page 1', pageW - margin - 10, pageH - 5);

        // ── Save ────────────────────────────────────────────────────
        var filename = 'MasterMind_Credit_Report_' + (data.applicationId || 'unknown') + '.pdf';
        doc.save(filename);
    }

    // ── Bind to download button ─────────────────────────────────────

    document.addEventListener('DOMContentLoaded', function () {
        var btn = document.getElementById('download-pdf-btn');
        if (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                _generatePDF();
            });
        }
    });

    // Expose globally for inline triggering
    window.MasterMindPDF = { generate: _generatePDF };

})();
