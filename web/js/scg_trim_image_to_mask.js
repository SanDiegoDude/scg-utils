import { app } from "../../../scripts/app.js";

// SCG Trim Image to Mask - show custom_color only for "custom" background,
// and offer an EyeDropper to sample colors from previews / the desktop.

const COLLAPSED = () => [0, -4];

function hideWidget(w) {
    if (!w || w.type === "scg-hidden") return;
    w._scgOrig = { type: w.type, computeSize: w.computeSize };
    w.type = "scg-hidden";
    w.computeSize = COLLAPSED;
    w.hidden = true;
}

function showWidget(w) {
    if (!w || !w._scgOrig) return;
    w.type = w._scgOrig.type;
    w.computeSize = w._scgOrig.computeSize;
    delete w._scgOrig;
    w.hidden = false;
}

function setVis(node, name, visible) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return;
    if (visible) showWidget(w);
    else hideWidget(w);
}

function normalizeHex(value, fallback = "#808080") {
    if (typeof value !== "string") return fallback;
    let h = value.trim();
    if (!h.startsWith("#")) h = `#${h}`;
    if (/^#[0-9a-fA-F]{3}$/.test(h)) {
        h = `#${h[1]}${h[1]}${h[2]}${h[2]}${h[3]}${h[3]}`;
    }
    if (/^#[0-9a-fA-F]{8}$/.test(h)) {
        h = h.slice(0, 7);
    }
    return /^#[0-9a-fA-F]{6}$/.test(h) ? h.toLowerCase() : fallback;
}

function setCustomColor(node, hex) {
    const colorW = node.widgets?.find((w) => w.name === "custom_color");
    if (!colorW) return;
    const value = normalizeHex(hex);
    colorW.value = value;
    if (typeof colorW.callback === "function") {
        colorW.callback(value);
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function updateVisibility(node) {
    const bg = node.widgets?.find((w) => w.name === "background")?.value;
    const isCustom = bg === "custom";
    setVis(node, "custom_color", isCustom);
    setVis(node, "eyedropper", isCustom);
    const sz = node.computeSize?.();
    if (sz) {
        node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function wrapWidgetCallback(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = function () {
        const r = orig ? orig.apply(this, arguments) : undefined;
        updateVisibility(node);
        return r;
    };
}

function ensureEyedropperButton(node) {
    if (node.widgets?.some((w) => w.name === "eyedropper")) return;

    const hasEyeDropper = typeof window.EyeDropper === "function";
    const label = hasEyeDropper ? "Eyedropper (sample screen)" : "Eyedropper (unsupported)";

    node.addWidget("button", label, null, async () => {
        if (!hasEyeDropper) {
            window.alert(
                "EyeDropper API is not available in this browser. Use Chrome/Edge, or pick a color with the custom_color widget."
            );
            return;
        }
        try {
            const result = await new window.EyeDropper().open();
            if (result?.sRGBHex) {
                setCustomColor(node, result.sRGBHex);
            }
        } catch (err) {
            // User cancel is normal; ignore AbortError-style cancels.
            if (err && (err.name === "AbortError" || err.name === "NotAllowedError")) {
                return;
            }
            console.warn("[SCG Trim] Eyedropper failed:", err);
        }
    });

    const btn = node.widgets[node.widgets.length - 1];
    btn.name = "eyedropper";
    btn.serialize = false;
}

app.registerExtension({
    name: "SCG.TrimImageToMask",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SCGTrimImageToMask") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            ensureEyedropperButton(this);
            wrapWidgetCallback(this, "background");
            updateVisibility(this);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            setTimeout(() => {
                ensureEyedropperButton(this);
                updateVisibility(this);
            }, 0);
            return r;
        };
    },
});
