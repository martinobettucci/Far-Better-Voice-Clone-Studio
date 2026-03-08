(function () {
    const resolveSourceUrl = (value) => {
        if (!value) return "";
        if (typeof value === "string") return value;
        if (typeof value === "object") {
            return value.path || value.url || value.name || "";
        }
        return "";
    };

    const setGradioValue = (elemId, value) => {
        const host = document.getElementById(elemId);
        if (!host) return;
        const input = host.querySelector("textarea, input");
        if (!input) return;
        input.value = value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
    };

    const buildController = () => {
        const canvas = document.getElementById("singing-waveform-canvas");
        const status = document.getElementById("singing-waveform-status");
        const clearButton = document.getElementById("singing-waveform-clear");
        if (!canvas || !status || !clearButton) {
            return null;
        }

        const ctx = canvas.getContext("2d");
        const state = {
            sourceUrl: "",
            audioBuffer: null,
            duration: 0,
            selection: null,
            dragging: false,
        };

        const resizeCanvas = () => {
            const ratio = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const width = Math.max(320, Math.round((rect.width || 960) * ratio));
            const height = Math.max(120, Math.round((rect.height || 180) * ratio));
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }
        };

        const clearSelection = () => {
            state.selection = null;
            setGradioValue("singing-selection-start", "");
            setGradioValue("singing-selection-end", "");
            setGradioValue("singing-selection-enabled", "false");
        };

        const updateStatus = (text) => {
            status.textContent = text;
        };

        const fractionFromEvent = (event) => {
            const rect = canvas.getBoundingClientRect();
            if (!rect.width) return 0;
            return Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        };

        const draw = () => {
            resizeCanvas();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = "#10131a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            if (!state.audioBuffer) {
                ctx.strokeStyle = "rgba(255,255,255,0.1)";
                ctx.beginPath();
                ctx.moveTo(0, canvas.height / 2);
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.stroke();
                return;
            }

            const samples = state.audioBuffer.getChannelData(0);
            const width = canvas.width;
            const height = canvas.height;
            const mid = height / 2;
            const step = Math.max(1, Math.floor(samples.length / width));

            ctx.strokeStyle = "rgba(255,255,255,0.1)";
            ctx.beginPath();
            ctx.moveTo(0, mid);
            ctx.lineTo(width, mid);
            ctx.stroke();

            ctx.strokeStyle = "#6bc7ff";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let x = 0; x < width; x += 1) {
                let min = 1.0;
                let max = -1.0;
                const start = x * step;
                const end = Math.min(samples.length, start + step);
                for (let i = start; i < end; i += 1) {
                    const value = samples[i];
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
                ctx.moveTo(x, mid + (min * mid * 0.9));
                ctx.lineTo(x, mid + (max * mid * 0.9));
            }
            ctx.stroke();

            if (state.selection) {
                const startX = Math.min(state.selection.start, state.selection.end) * width;
                const endX = Math.max(state.selection.start, state.selection.end) * width;
                ctx.fillStyle = "rgba(255, 202, 87, 0.18)";
                ctx.fillRect(startX, 0, endX - startX, height);
                ctx.strokeStyle = "rgba(255, 202, 87, 0.9)";
                ctx.lineWidth = 2;
                ctx.strokeRect(startX, 1, Math.max(1, endX - startX), height - 2);
            }
        };

        const syncSelection = () => {
            if (!state.audioBuffer || !state.selection) {
                updateStatus("No segment selected. Processing will use the full track.");
                clearSelection();
                draw();
                return;
            }

            const startFrac = Math.min(state.selection.start, state.selection.end);
            const endFrac = Math.max(state.selection.start, state.selection.end);
            const start = startFrac * state.duration;
            const end = endFrac * state.duration;
            if ((end - start) < 0.03) {
                updateStatus("Selection too small. Processing will use the full track.");
                clearSelection();
                draw();
                return;
            }

            setGradioValue("singing-selection-start", start.toFixed(4));
            setGradioValue("singing-selection-end", end.toFixed(4));
            setGradioValue("singing-selection-enabled", "true");
            updateStatus(`Selected segment: ${start.toFixed(2)}s -> ${end.toFixed(2)}s`);
            draw();
        };

        const loadAudio = async (sourceUrl) => {
            if (!sourceUrl) {
                state.sourceUrl = "";
                state.audioBuffer = null;
                state.duration = 0;
                clearSelection();
                updateStatus("Upload or record a vocal track, then drag on the waveform to select the segment to process.");
                draw();
                return;
            }

            if (state.sourceUrl === sourceUrl) {
                draw();
                return;
            }

            try {
                updateStatus("Loading waveform...");
                const response = await fetch(sourceUrl);
                const arrayBuffer = await response.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
                state.sourceUrl = sourceUrl;
                state.audioBuffer = decoded;
                state.duration = decoded.duration || 0;
                clearSelection();
                updateStatus("No segment selected. Processing will use the full track.");
                draw();
                if (audioContext && audioContext.close) {
                    audioContext.close().catch(() => {});
                }
            } catch (_error) {
                state.sourceUrl = sourceUrl;
                state.audioBuffer = null;
                state.duration = 0;
                clearSelection();
                updateStatus("Could not render waveform preview. Processing still works on the full track.");
                draw();
            }
        };

        const controller = {
            canvas,
            refresh(sourceValue, force = false) {
                const directSource = resolveSourceUrl(sourceValue);
                const fallbackAudioEl = document.querySelector("#singing-source-audio audio");
                const fallbackSource = fallbackAudioEl
                    ? (fallbackAudioEl.currentSrc || fallbackAudioEl.src || "")
                    : "";
                const sourceUrl = directSource || fallbackSource;
                if (force && !sourceUrl) {
                    loadAudio("");
                    return;
                }
                if (force || sourceUrl !== state.sourceUrl) {
                    loadAudio(sourceUrl);
                    return;
                }
                draw();
            },
            clearSelection() {
                clearSelection();
                updateStatus("Selection cleared. Processing will use the full track.");
                draw();
            },
        };

        canvas.addEventListener("mousedown", (event) => {
            if (!state.audioBuffer) return;
            state.dragging = true;
            const start = fractionFromEvent(event);
            state.selection = { start, end: start };
            draw();
        });

        window.addEventListener("mousemove", (event) => {
            if (!state.dragging || !state.audioBuffer || !state.selection) return;
            state.selection.end = fractionFromEvent(event);
            draw();
        });

        window.addEventListener("mouseup", () => {
            if (!state.dragging) return;
            state.dragging = false;
            syncSelection();
        });

        clearButton.addEventListener("click", () => controller.clearSelection());
        window.addEventListener("resize", () => draw());
        draw();

        return controller;
    };

    window.ensureSingingWaveformStudio = function ensureSingingWaveformStudio() {
        const canvas = document.getElementById("singing-waveform-canvas");
        if (!canvas) {
            return null;
        }
        const existing = window.__singingWaveformStudio;
        if (existing && existing.canvas === canvas) {
            return existing;
        }
        const controller = buildController();
        window.__singingWaveformStudio = controller;
        if (!window.__singingWaveformStudioInterval) {
            window.__singingWaveformStudioInterval = window.setInterval(() => {
                window.__singingWaveformStudio?.refresh(null, false);
            }, 1200);
        }
        return controller;
    };

    window.refreshSingingWaveformStudio = function refreshSingingWaveformStudio(sourceValue, force = true) {
        const controller = window.ensureSingingWaveformStudio?.();
        if (!controller) {
            return "";
        }
        window.setTimeout(() => controller.refresh(sourceValue, force), 150);
        return "";
    };
})();
