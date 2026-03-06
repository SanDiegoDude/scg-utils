import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// SCG Load Audio Plus - Audio upload with preview and trimming
app.registerExtension({
    name: "SCG.LoadAudioPlus",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SCGLoadAudioPlus") {
            return;
        }
        
        // Store original onNodeCreated
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            // Find the audio combo widget
            const audioWidget = this.widgets?.find(w => w.name === "audio");
            if (!audioWidget) return;
            
            // Create audio preview element
            const audioContainer = document.createElement("div");
            audioContainer.style.cssText = "width: 100%; padding: 5px 0;";
            
            const audioElement = document.createElement("audio");
            audioElement.controls = true;
            audioElement.style.width = "100%";
            audioElement.style.height = "32px";
            
            audioContainer.appendChild(audioElement);
            
            // Add the audio preview widget
            const audioPreviewWidget = this.addDOMWidget("audio_preview", "audio", audioContainer, {
                serialize: false,
                hideOnZoom: false,
            });
            
            audioPreviewWidget.computeSize = function(width) {
                return [width, 42];
            };
            
            // Store references
            this.audioElement = audioElement;
            this.audioPreviewWidget = audioPreviewWidget;
            
            // Function to update audio preview from selected file
            const updateAudioPreview = (filename) => {
                if (!filename || filename === "no_audio_files_found" || filename === "audio_not_available") {
                    audioElement.src = "";
                    return;
                }
                
                // Build URL to fetch audio from input folder
                const params = new URLSearchParams({
                    filename: filename,
                    subfolder: "",
                    type: "input"
                });
                
                const audioUrl = api.apiURL(`/view?${params.toString()}`);
                audioElement.src = audioUrl;
                audioElement.load();
            };
            
            // Set up callback for when audio selection changes
            const origCallback = audioWidget.callback;
            audioWidget.callback = function(value) {
                if (origCallback) {
                    origCallback.apply(this, arguments);
                }
                updateAudioPreview(value);
            };
            
            // Initial preview load
            if (audioWidget.value) {
                updateAudioPreview(audioWidget.value);
            }
            
            // Find trim widgets and add validation to prevent empty values
            const trimStartWidget = this.widgets?.find(w => w.name === "trim_start_time");
            const trimLengthWidget = this.widgets?.find(w => w.name === "trim_sample_length");
            
            // Helper to ensure float widgets always have valid numeric values
            const ensureNumericValue = (widget, defaultVal = 0.0) => {
                if (!widget) return;
                
                // Store original callback
                const origWidgetCallback = widget.callback;
                
                widget.callback = function(value) {
                    // If value is empty string or invalid, reset to default
                    if (value === "" || value === null || value === undefined || isNaN(parseFloat(value))) {
                        widget.value = defaultVal;
                        value = defaultVal;
                    }
                    if (origWidgetCallback) {
                        origWidgetCallback.call(this, value);
                    }
                };
                
                // Also ensure initial value is valid
                if (widget.value === "" || widget.value === null || widget.value === undefined || isNaN(parseFloat(widget.value))) {
                    widget.value = defaultVal;
                }
            };
            
            ensureNumericValue(trimStartWidget, 0.0);
            ensureNumericValue(trimLengthWidget, 0.0);
            
            // Add upload button
            const uploadWidget = this.addWidget("button", "choose_file_upload", "Choose file to upload", () => {
                // Create hidden file input
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "audio/*,video/*";
                fileInput.style.display = "none";
                
                fileInput.onchange = async () => {
                    if (fileInput.files.length === 0) return;
                    
                    const file = fileInput.files[0];
                    
                    try {
                        // Upload file to ComfyUI input folder
                        const formData = new FormData();
                        formData.append("image", file);  // ComfyUI uses "image" for all uploads
                        formData.append("subfolder", "");
                        formData.append("type", "input");
                        
                        const response = await api.fetchApi("/upload/image", {
                            method: "POST",
                            body: formData,
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            const filename = data.name;
                            
                            // Update the combo widget with new file
                            // Add to options if not already present
                            if (!audioWidget.options.values.includes(filename)) {
                                audioWidget.options.values.push(filename);
                                audioWidget.options.values.sort();
                            }
                            
                            // Select the uploaded file
                            audioWidget.value = filename;
                            audioWidget.callback?.(filename);
                            
                            console.log(`[SCG Load Audio Plus] Uploaded: ${filename}`);
                        } else {
                            console.error("[SCG Load Audio Plus] Upload failed:", response.statusText);
                            alert("Upload failed: " + response.statusText);
                        }
                    } catch (error) {
                        console.error("[SCG Load Audio Plus] Upload error:", error);
                        alert("Upload error: " + error.message);
                    }
                    
                    // Clean up
                    document.body.removeChild(fileInput);
                };
                
                document.body.appendChild(fileInput);
                fileInput.click();
            });
            
            // Reorder widgets: audio preview at top, then selector, trim controls, upload button at bottom
            const widgets = this.widgets;
            const reorderedWidgets = [];
            
            // Find widgets by name
            const audioPreview = widgets.find(w => w.name === "audio_preview");
            const audioSelectorWidget = widgets.find(w => w.name === "audio");
            const trimStart = widgets.find(w => w.name === "trim_start_time");
            const trimLength = widgets.find(w => w.name === "trim_sample_length");
            const uploadBtn = widgets.find(w => w.name === "choose_file_upload");
            
            // Add in desired order: preview, selector, trim controls, upload button
            if (audioPreview) reorderedWidgets.push(audioPreview);
            if (audioSelectorWidget) reorderedWidgets.push(audioSelectorWidget);
            if (trimStart) reorderedWidgets.push(trimStart);
            if (trimLength) reorderedWidgets.push(trimLength);
            if (uploadBtn) reorderedWidgets.push(uploadBtn);
            
            // Add any remaining widgets
            for (const w of widgets) {
                if (!reorderedWidgets.includes(w)) {
                    reorderedWidgets.push(w);
                }
            }
            
            this.widgets = reorderedWidgets;
        };
        
        // Handle execution results (update preview with trimmed audio if returned)
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }
            
            // If execution returned audio preview data, update to show trimmed version
            if (message && message.audio && message.audio.length > 0) {
                const audioInfo = message.audio[0];
                
                const params = new URLSearchParams({
                    filename: audioInfo.filename,
                    subfolder: audioInfo.subfolder || "",
                    type: audioInfo.type || "temp"
                });
                
                const audioUrl = api.apiURL(`/view?${params.toString()}`);
                
                if (this.audioElement) {
                    this.audioElement.src = audioUrl;
                    this.audioElement.load();
                }
            }
        };
    }
});
