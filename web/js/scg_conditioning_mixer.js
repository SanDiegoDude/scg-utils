import { app } from "../../../scripts/app.js";

// SCG Conditioning Mixer - progressive disclosure of widgets.
//
// Widgets are hidden (not removed) so their values + any input-conversions stay
// intact and keep serializing. Input sockets are left in place on purpose:
// removing/re-adding sockets risks dropping links when a graph is loaded before
// its connections are restored.

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

function inputConnected(node, name) {
    const i = node.inputs?.find((x) => x.name === name);
    return !!(i && i.link != null);
}

function widgetValue(node, name) {
    return node.widgets?.find((x) => x.name === name)?.value;
}

function updateVisibility(node) {
    const aOn = inputConnected(node, "conditioning_a");
    const bOn = inputConnected(node, "conditioning_b");
    const anyCondOn = aOn || bOn;
    const includeText = !!widgetValue(node, "include_text_only");
    // With no conditioning inputs the node is a pure text encoder, so the text
    // controls are always relevant (regardless of include_text_only).
    const textActive = !anyCondOn || includeText;

    const aTaper = widgetValue(node, "a_taper") && widgetValue(node, "a_taper") !== "off";
    const bTaper = widgetValue(node, "b_taper") && widgetValue(node, "b_taper") !== "off";
    const tTaper = widgetValue(node, "text_taper") && widgetValue(node, "text_taper") !== "off";

    // A group (revealed once A is connected)
    setVis(node, "a_strength", aOn);
    setVis(node, "a_start", aOn);
    setVis(node, "a_end", aOn);
    setVis(node, "a_taper", aOn);
    setVis(node, "a_taper_target", aOn && aTaper);

    // B group (revealed once B is connected)
    setVis(node, "b_strength", bOn);
    setVis(node, "b_start", bOn);
    setVis(node, "b_end", bOn);
    setVis(node, "b_taper", bOn);
    setVis(node, "b_taper_target", bOn && bTaper);

    // Merge controls only matter when there are two inputs to merge
    setVis(node, "merge_style", aOn && bOn);
    setVis(node, "merge_strength", aOn && bOn);

    // taper_steps only matters when something is actually tapering
    const anyTaper = (aOn && aTaper) || (bOn && bTaper) || (textActive && tTaper);
    setVis(node, "taper_steps", anyTaper);

    // include_text_only only matters when there is conditioning to merge text into
    setVis(node, "include_text_only", anyCondOn);

    // Text group (text_prompt + text_custom_template stay visible always so they
    // never get disconnected when toggled off)
    setVis(node, "text_template", textActive);
    setVis(node, "text_strength", textActive);
    setVis(node, "text_start", textActive);
    setVis(node, "text_end", textActive);
    setVis(node, "text_taper", textActive);
    setVis(node, "text_taper_target", textActive && tTaper);
    // Text merge controls only matter when merging text WITH conditioning inputs
    setVis(node, "text_merge_style", includeText && anyCondOn);
    setVis(node, "text_merge_strength", includeText && anyCondOn);

    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas(true, true);
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

app.registerExtension({
    name: "SCG.ConditioningMixer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SCGConditioningTrajectory") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            for (const name of ["include_text_only", "a_taper", "b_taper", "text_taper"]) {
                wrapWidgetCallback(this, name);
            }
            // Move the multiline (DOM) widgets to the end so no canvas widget is
            // rendered after them - otherwise trailing widgets drift outside the
            // node body and won't collapse when hidden.
            if (this.widgets) {
                for (const name of ["text_prompt", "text_custom_template"]) {
                    const i = this.widgets.findIndex((w) => w.name === name);
                    if (i !== -1) this.widgets.push(this.widgets.splice(i, 1)[0]);
                }
            }
            updateVisibility(this);
            return r;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
            updateVisibility(this);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            // Defer so restored links/inputs are in place before we read them.
            setTimeout(() => updateVisibility(this), 0);
            return r;
        };
    },
});
