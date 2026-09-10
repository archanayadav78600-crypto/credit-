// script.js
// Small helper script for the Fraud Watch app.
// 1. Draws a simple bar chart (Normal vs Fraud) on the dashboard page
//    using plain <canvas> — no external chart library needed.
// 2. Keeps the device-trust-score slider label in sync (inline oninput
//    is also set in detect.html as a fallback).

document.addEventListener("DOMContentLoaded", function () {
    drawClassChart();
});

function drawClassChart() {
    const canvas = document.getElementById("classChart");
    if (!canvas) return; // Not on the dashboard page

    const ctx = canvas.getContext("2d");
    const normal = parseInt(canvas.dataset.normal, 10) || 0;
    const fraud = parseInt(canvas.dataset.fraud, 10) || 0;
    const maxValue = Math.max(normal, fraud);

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const chartHeight = height - padding * 2;
    const barWidth = 90;
    const gap = 60;

    ctx.clearRect(0, 0, width, height);

    // Axis line
    ctx.strokeStyle = "#D7DCE5";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    const bars = [
        { label: "Normal", value: normal, color: "#146C43" },
        { label: "Fraud", value: fraud, color: "#B3261E" },
    ];

    const startX = (width - (barWidth * 2 + gap)) / 2;

    bars.forEach((bar, index) => {
        const barHeight = maxValue > 0 ? (bar.value / maxValue) * chartHeight : 0;
        const x = startX + index * (barWidth + gap);
        const y = height - padding - barHeight;

        ctx.fillStyle = bar.color;
        ctx.fillRect(x, y, barWidth, barHeight);

        // Value label above the bar
        ctx.fillStyle = "#101826";
        ctx.font = "600 14px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(bar.value.toLocaleString(), x + barWidth / 2, y - 10);

        // Category label below the axis
        ctx.fillStyle = "#3d4a5f";
        ctx.font = "500 13px 'IBM Plex Sans', sans-serif";
        ctx.fillText(bar.label, x + barWidth / 2, height - padding + 20);
    });
}
