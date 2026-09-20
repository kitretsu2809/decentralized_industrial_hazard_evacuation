class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.onStateUpdate = null;
        this.isConnected = false;
        this.reconnectInterval = 2000;
        this.pollInterval = null;
    }

    connect() {
        console.log(`Connecting to WebSocket at ${this.url}...`);
        try {
            this.ws = new WebSocket(this.url);
        } catch (e) {
            console.warn("WebSocket construct error, activating HTTP fallback:", e);
            this.startPollingFallback();
            return;
        }

        this.ws.onopen = () => {
            console.log("WebSocket connected!");
            this.isConnected = true;
            this.stopPollingFallback();
            this.updateConnectionStatus();
        };

        this.ws.onmessage = (event) => {
            try {
                const state = JSON.parse(event.data);
                if (this.onStateUpdate) {
                    this.onStateUpdate(state);
                }
            } catch (err) {
                console.error("Failed to parse WebSocket message:", err);
            }
        };

        this.ws.onclose = () => {
            console.log("WebSocket disconnected. Falling back to HTTP polling and reconnecting...");
            this.isConnected = false;
            this.startPollingFallback();
            this.updateConnectionStatus();
            setTimeout(() => this.connect(), this.reconnectInterval);
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            this.startPollingFallback();
            try { this.ws.close(); } catch(e) {}
        };
    }

    startPollingFallback() {
        if (this.pollInterval) return;
        console.log("Active HTTP polling fallback started (/api/state)...");
        this.pollInterval = setInterval(async () => {
            if (this.isConnected) return;
            try {
                const res = await fetch('/api/state');
                if (res.ok) {
                    const state = await res.json();
                    if (this.onStateUpdate) {
                        this.onStateUpdate(state);
                    }
                    this.updateConnectionStatus(true);
                }
            } catch (e) {
                // ignore transient poll errors
            }
        }, 120);
    }

    stopPollingFallback() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
            console.log("HTTP polling fallback stopped, WebSocket active.");
        }
    }

    disconnect() {
        this.stopPollingFallback();
        if (this.ws) {
            this.ws.close();
        }
    }

    updateConnectionStatus(pollingActive = false) {
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            if (this.isConnected || pollingActive) {
                indicator.classList.add('connected');
                indicator.title = this.isConnected ? "WebSocket Connected (Real-Time)" : "HTTP Polling Active (Fallback)";
            } else {
                indicator.classList.remove('connected');
                indicator.title = "Disconnected";
            }
        }
    }
}

