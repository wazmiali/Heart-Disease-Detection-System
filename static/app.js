const body = document.body;
const themeSelect = document.getElementById("themePreference");
const accentSelect = document.getElementById("accentStyle");
const compactMode = document.getElementById("compactMode");
const settingsPanel = document.getElementById("settingsPanel");
const settingsToggle = document.getElementById("settingsToggle");
const themeCycleBtn = document.getElementById("themeCycleBtn");

function applyTheme(value) {
    if (!value) {
        return;
    }
    body.dataset.theme = value;
    if (themeSelect) {
        themeSelect.value = value;
    }
}

function applyAccent(value) {
    if (!value) {
        return;
    }
    body.dataset.accent = value;
}

if (themeSelect) {
    themeSelect.addEventListener("change", (event) => applyTheme(event.target.value));
}

if (accentSelect) {
    accentSelect.addEventListener("change", (event) => applyAccent(event.target.value));
}

if (compactMode) {
    compactMode.addEventListener("change", () => {
        body.classList.toggle("compact-mode", compactMode.checked);
    });
}

if (settingsToggle && settingsPanel) {
    settingsToggle.addEventListener("click", () => {
        settingsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}

if (themeCycleBtn) {
    const themeOrder = ["system", "light", "dark"];
    themeCycleBtn.addEventListener("click", () => {
        const currentTheme = body.dataset.theme || "system";
        const currentIndex = themeOrder.indexOf(currentTheme);
        const nextTheme = themeOrder[(currentIndex + 1) % themeOrder.length];
        applyTheme(nextTheme);
    });
}

applyTheme(body.dataset.theme || "system");
applyAccent(body.dataset.accent || "teal");
