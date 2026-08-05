class FloorManager {
    constructor(renderer) {
        this.renderer = renderer;
        this.floors = [1, 2, 3]; // Default, will update from state/building
        this.currentFloor = 1;
        this.floorHazards = {};  // floor -> max hazard
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
            btn.innerHTML = `Floor ${floor} <span class="hazard-dot" id="hazard-dot-${floor}"></span>`;
            btn.onclick = () => this.switchFloor(floor);
            container.appendChild(btn);
        });
        
        // "All" button for cross-section
        const allBtn = document.createElement('button');
        allBtn.className = `floor-btn ${this.currentFloor === 'ALL' ? 'active' : ''}`;
        allBtn.innerText = 'All (Cross)';
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
                // The 'All' button
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
        
        // Reset max hazards
        this.floors.forEach(f => { this.floorHazards[f] = 0; });
        
        // Calculate new max hazards
        for (const nodeData of Object.values(state.nodes)) {
            const f = nodeData.floor;
            if (!this.floors.includes(f)) {
                this.floors.push(f);
                this.floors.sort();
                this.renderTabs(); // Re-render if new floor discovered
            }
            if (nodeData.hazard_score > (this.floorHazards[f] || 0)) {
                this.floorHazards[f] = nodeData.hazard_score;
            }
        }
        
        // Update dots
        this.floors.forEach(floor => {
            const dot = document.getElementById(`hazard-dot-${floor}`);
            if (dot) {
                const h = this.floorHazards[floor];
                if (h <= 0.3) {
                    dot.style.background = 'var(--status-safe)';
                    dot.style.boxShadow = 'none';
                    dot.style.animation = 'none';
                } else if (h <= 0.7) {
                    dot.style.background = 'var(--status-caution)';
                    dot.style.boxShadow = '0 0 5px var(--status-caution)';
                    dot.style.animation = 'none';
                } else {
                    dot.style.background = 'var(--status-danger)';
                    dot.style.boxShadow = '0 0 10px var(--status-danger)';
                    dot.style.animation = 'pulse 1s infinite';
                }
            }
        });
    }
}
