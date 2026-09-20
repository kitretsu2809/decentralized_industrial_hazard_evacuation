class FloorManager {
    constructor(renderer) {
        this.renderer = renderer;
        this.floors = [1, 2, 3];
        this.currentFloor = 1;
        this.floorHazards = {};
        this.floorNames = {
            1: "L1: Process Units & Tank Farm",
            2: "L2: Catwalks & SCADA",
            3: "L3: Flare Stack Platform"
        };
    }

    init() {
        this.renderTabs();
    }

    renderTabs() {
        const container = document.getElementById('floor-tabs');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.floors.forEach(floor => {
            const btn = document.createElement('button');
            btn.className = `floor-btn ${floor === this.currentFloor ? 'active' : ''}`;
            const label = this.floorNames[floor] || `Floor ${floor}`;
            btn.innerHTML = `${label} <span class="hazard-dot" id="hazard-dot-${floor}"></span>`;
            btn.onclick = () => this.switchFloor(floor);
            container.appendChild(btn);
        });
        
        // "All" button for cross-section
        const allBtn = document.createElement('button');
        allBtn.className = `floor-btn ${this.currentFloor === 'ALL' ? 'active' : ''}`;
        allBtn.innerHTML = `All Units (3D Elevation) <span class="hazard-dot" id="hazard-dot-all"></span>`;
        allBtn.onclick = () => this.showCrossSection();
        container.appendChild(allBtn);
    }

    switchFloor(floor) {
        this.currentFloor = floor;
        this.renderer.setFloor(floor);
        this.updateTabStyles();
    }
    
    showCrossSection() {
        this.currentFloor = 'ALL';
        this.renderer.setFloor('ALL');
        this.updateTabStyles();
    }

    updateTabStyles() {
        const btns = document.querySelectorAll('.floor-btn');
        btns.forEach((btn, index) => {
            if (index < this.floors.length) {
                if (this.floors[index] === this.currentFloor) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            } else {
                if (this.currentFloor === 'ALL') {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            }
        });
    }

    updateFloorIndicators(state) {
        if (!state || !state.nodes) return;

        this.floorHazards = { 1: 0, 2: 0, 3: 0, 'all': 0 };

        Object.values(state.nodes).forEach(node => {
            const f = node.floor;
            const h = node.hazard_score || 0;
            if (this.floorHazards[f] !== undefined) {
                this.floorHazards[f] = Math.max(this.floorHazards[f], h);
            }
            this.floorHazards['all'] = Math.max(this.floorHazards['all'], h);
        });

        const getStatusColor = (h) => {
            if (h <= 0.01) return 'var(--status-safe)';
            if (h <= 0.3) return 'var(--status-caution)';
            return 'var(--status-danger)';
        };

        this.floors.forEach(floor => {
            const dot = document.getElementById(`hazard-dot-${floor}`);
            if (dot) {
                dot.style.background = getStatusColor(this.floorHazards[floor]);
            }
        });

        const allDot = document.getElementById('hazard-dot-all');
        if (allDot) {
            allDot.style.background = getStatusColor(this.floorHazards['all']);
        }
    }
}
