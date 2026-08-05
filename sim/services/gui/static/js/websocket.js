class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.onStateUpdate = null;
        this.isConnected = false;
        this.reconnectInterval = 2000;
    }

    connect() {
        console.log(`Connecting to WebSocket at ${this.url}...`);
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log("WebSocket connected!");
            this.isConnected = true;
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
            console.log("WebSocket disconnected. Reconnecting...");
            this.isConnected = false;
            this.updateConnectionStatus();
            setTimeout(() => this.connect(), this.reconnectInterval);
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            this.ws.close(); // trigger reconnect
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }

    updateConnectionStatus() {
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            if (this.isConnected) {
                indicator.classList.add('connected');
            } else {
                indicator.classList.remove('connected');
            }
        }
    }
}
