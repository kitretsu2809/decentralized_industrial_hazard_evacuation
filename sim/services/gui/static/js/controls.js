class ControlPanel {
    constructor(websocketManager) {
        this.ws = websocketManager; // Not strictly used for POSTs, but good for reference
        this.selectedThreatType = 'FIRE';
        this.intensity = 0.8;
        this.spreadRate = 0.1;
        this.selectedNodeId = null;
        this.selectedFloor = null;
    }

    init() {
        // Threat injection controls
        const typeSelect = document.getElementById('threat-type');
        if (typeSelect) {
            typeSelect.addEventListener('change', (e) => {
                this.selectedThreatType = e.target.value;
            });
        }

        const intensitySlider = document.getElementById('threat-intensity');
        const intensityVal = document.getElementById('intensity-val');
        if (intensitySlider && intensityVal) {
            intensitySlider.addEventListener('input', (e) => {
                this.intensity = parseFloat(e.target.value);
                intensityVal.innerText = this.intensity.toFixed(1);
            });
        }

        const spreadSlider = document.getElementById('threat-spread');
        const spreadVal = document.getElementById('spread-val');
        if (spreadSlider && spreadVal) {
            spreadSlider.addEventListener('input', (e) => {
                this.spreadRate = parseFloat(e.target.value);
                spreadVal.innerText = this.spreadRate.toFixed(2);
            });
        }

        const injectBtn = document.getElementById('btn-inject');
        if (injectBtn) {
            injectBtn.addEventListener('click', () => this.injectDisaster());
        }

        // Simulation controls
        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) btnPlay.addEventListener('click', () => this.sendControl('play'));
        
        const btnPause = document.getElementById('btn-pause');
        if (btnPause) btnPause.addEventListener('click', () => this.sendControl('pause'));
        
        const btnStep = document.getElementById('btn-step');
        if (btnStep) btnStep.addEventListener('click', () => this.sendControl('step'));
        
        const btnReset = document.getElementById('btn-reset');
        if (btnReset) btnReset.addEventListener('click', () => this.sendControl('reset'));

        const speedSelect = document.getElementById('sim-speed');
        if (speedSelect) {
            speedSelect.addEventListener('change', (e) => {
                this.sendControl('speed', parseFloat(e.target.value));
            });
        }
    }

    onNodeSelected(nodeId, floor) {
        this.selectedNodeId = nodeId;
        this.selectedFloor = floor;
        
        const infoDiv = document.getElementById('selected-node-info');
        const idSpan = document.getElementById('selected-node-id');
        
        if (nodeId) {
            idSpan.innerText = `${nodeId} (Floor ${floor})`;
            infoDiv.classList.remove('hidden');
        } else {
            infoDiv.classList.add('hidden');
        }
    }

    async injectDisaster() {
        if (!this.selectedNodeId) return;

        const payload = {
            threat_type: this.selectedThreatType,
            target_node: this.selectedNodeId,
            floor: this.selectedFloor,
            intensity: this.intensity,
            spread_rate: this.spreadRate,
            metadata: {}
        };

        try {
            const res = await fetch('/api/inject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            console.log("Injection result:", data);
            
            // Give visual feedback
            const btn = document.getElementById('btn-inject');
            const originalText = btn.innerText;
            btn.innerText = "Injected!";
            btn.style.background = "var(--status-safe)";
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.style.background = "var(--threat-fire)";
            }, 2000);
            
        } catch (err) {
            console.error("Failed to inject disaster:", err);
        }
    }

    async sendControl(command, value = null) {
        const payload = {
            command: command,
            value: value
        };

        try {
            const res = await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            console.log(`Control ${command} sent:`, await res.json());
        } catch (err) {
            console.error(`Failed to send control ${command}:`, err);
        }
    }
}
