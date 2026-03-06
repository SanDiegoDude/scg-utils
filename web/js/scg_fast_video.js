import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// SCG Fast Video from Audio - Video preview with autoplay
app.registerExtension({
    name: "SCG.FastVideoFromAudio",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SCGFastVideoFromAudio") {
            return;
        }
        
        // Store original onNodeCreated
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            // Create video preview container with fixed max height
            const videoContainer = document.createElement("div");
            videoContainer.style.cssText = "width: 100%; padding: 5px 0; max-height: 200px; overflow: hidden;";
            
            const videoElement = document.createElement("video");
            videoElement.controls = true;
            videoElement.style.width = "100%";
            videoElement.style.maxHeight = "200px";
            videoElement.style.objectFit = "contain";
            videoElement.style.backgroundColor = "#000";
            videoElement.playsInline = true;
            
            videoContainer.appendChild(videoElement);
            
            // Add the video preview widget with fixed size
            const videoPreviewWidget = this.addDOMWidget("video_preview", "video", videoContainer, {
                serialize: false,
                hideOnZoom: false,
            });
            
            // Fixed height for the video widget to prevent it from expanding
            videoPreviewWidget.computeSize = function(width) {
                return [width, 210]; // Fixed height
            };
            
            // Store references
            this.videoElement = videoElement;
            this.videoPreviewWidget = videoPreviewWidget;
            
            // Reorder widgets to put video preview at the top
            const widgets = this.widgets;
            const reorderedWidgets = [];
            
            // Find video preview widget
            const videoPreview = widgets.find(w => w.name === "video_preview");
            
            // Add video preview first
            if (videoPreview) reorderedWidgets.push(videoPreview);
            
            // Add remaining widgets in their original order
            for (const w of widgets) {
                if (!reorderedWidgets.includes(w)) {
                    reorderedWidgets.push(w);
                }
            }
            
            this.widgets = reorderedWidgets;
        };
        
        // Handle execution results - show video preview
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }
            
            if (message && message.video && message.video.length > 0) {
                const videoInfo = message.video[0];
                
                // Build URL to fetch the video
                const params = new URLSearchParams({
                    filename: videoInfo.filename,
                    subfolder: videoInfo.subfolder || "",
                    type: videoInfo.type || "output"
                });
                
                const videoUrl = api.apiURL(`/view?${params.toString()}`);
                
                if (this.videoElement) {
                    this.videoElement.src = videoUrl;
                    this.videoElement.load();
                    
                    // Autoplay if enabled
                    if (videoInfo.autoplay) {
                        this.videoElement.play().catch(err => {
                            console.log("[SCG Fast Video] Autoplay blocked by browser:", err);
                        });
                    }
                }
                
                console.log(`[SCG Fast Video] Video preview loaded: ${videoInfo.filename}`);
            }
        };
    }
});
