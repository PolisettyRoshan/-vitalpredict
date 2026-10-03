
const form = document.getElementById("predictionForm");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.textContent = "Running Random Forest...";

    const payload = {
        pregnancies: document.getElementById("pregnancies").value,
        glucose: document.getElementById("glucose").value,
        blood_pressure: document.getElementById("blood_pressure").value,
        skin_thickness: document.getElementById("skin_thickness").value,
        insulin: document.getElementById("insulin").value,
        bmi: document.getElementById("bmi").value,
        pedigree: document.getElementById("pedigree").value,
        age: document.getElementById("age").value
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!data.success) throw new Error(data.error);

        showResult(data.result);
        loadHistory();
    } catch (err) {
        alert("Prediction error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Predict Diabetes Risk";
    }
});

function showResult(r) {
    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("result").classList.remove("hidden");

    const pill = document.getElementById("riskPill");
    pill.textContent = r.risk;
    pill.className = "risk-pill" + (r.risk === "Medium Risk" ? " medium" : "");

    document.getElementById("probability").textContent = r.probability;
    document.getElementById("meterBar").style.width = r.probability + "%";
    document.getElementById("resultTime").textContent = "Generated: " + r.timestamp;

    const container = document.getElementById("features");
    container.innerHTML = "";

    r.top_features.forEach(item => {
        container.innerHTML += `
            <div class="feature">
                <div class="feature-head">
                    <span>${item.feature}</span><span>${item.importance}%</span>
                </div>
                <div class="feature-bar"><div style="width:${Math.min(item.importance * 4, 100)}%"></div></div>
            </div>`;
    });
}

async function loadHistory() {
    const response = await fetch("/history");
    const rows = await response.json();
    const container = document.getElementById("history");

    if (!rows.length) {
        container.innerHTML = '<p class="muted">No predictions yet.</p>';
        return;
    }

    let html = `
        <div class="history-row header">
            <div>Time</div><div>Risk</div><div>Probability</div><div>Status</div>
        </div>`;

    rows.forEach(r => {
        html += `
            <div class="history-row">
                <div>${r.timestamp}</div>
                <div>${r.risk}</div>
                <div>${r.probability}%</div>
                <div>Random Forest</div>
            </div>`;
    });
    container.innerHTML = html;
}

loadHistory();
