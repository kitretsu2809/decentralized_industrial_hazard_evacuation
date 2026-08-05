class BuildingRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.currentFloor = 1;
        this.buildingData = null;  // Setup initially
        this.state = null;         // Updated via WS
        
        // Viewport config
        this.scale = 3.0;
        this.offset = { x: 50, y: 50 };
        
        // Interaction
        this.hoveredNode = null;
        this.selectedNode = null;
        this.onNodeSelected = null; // Callback for UI
        
        this.animationFrame = null;
        
        // Bind events
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));
        
        // Handle resize
        window.addEventListener('resize', this.resize.bind(this));
        
        // Last update time for interpolation/animation
        this.lastTime = performance.now();
    }

    init(buildingData) {
        this.buildingData = buildingData;
        
        // Collect floor numbers if buildingData has them, else rely on floorManager
        if (buildingData && buildingData.floors) {
            let keys = Object.keys(buildingData.floors).map(k => parseInt(k));
            if (keys.length > 0) this.currentFloor = Math.min(...keys);
        }
        
        this.resize();
    }

    setFloor(floor) {
        this.currentFloor = floor;
        this.selectedNode = null; // Deselect on floor switch
    }

    updateState(state) {
        this.state = state;
    }

    resize() {
        const parent = this.canvas.parentElement;
        if (parent) {
            this.canvas.width = parent.clientWidth;
            this.canvas.height = parent.clientHeight;
        }
    }

    worldToScreen(x, y) {
        return {
            x: x * this.scale + this.offset.x + this.canvas.width / 2,
            y: y * this.scale + this.offset.y + this.canvas.height / 2
        };
    }

    screenToWorld(x, y) {
        return {
            x: (x - this.canvas.width / 2 - this.offset.x) / this.scale,
            y: (y - this.canvas.height / 2 - this.offset.y) / this.scale
        };
    }

    getHazardColor(h) {
        // h is 0.0 to 1.0
        if (h <= 0.01) return '#10b981'; // Green
        if (h <= 0.3) return '#eab308'; // Yellow
        if (h <= 0.6) return '#f97316'; // Orange
        return '#ef4444'; // Red
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        
        const worldPos = this.screenToWorld(mx, my);
        
        this.hoveredNode = null;
        
        if (!this.state || !this.state.nodes) return;
        
        for (const [nodeId, nodeData] of Object.entries(this.state.nodes)) {
            if (nodeData.floor !== this.currentFloor) continue;
            
            const dx = nodeData.position[0] - worldPos.x;
            const dy = nodeData.position[1] - worldPos.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            // Check radius (base 12 / scale)
            if (dist < 15 / this.scale) {
                this.hoveredNode = nodeId;
                break;
            }
        }
    }

    handleClick(e) {
        if (this.hoveredNode) {
            this.selectedNode = this.hoveredNode;
            if (this.onNodeSelected) {
                this.onNodeSelected(this.selectedNode, this.currentFloor);
            }
        } else {
            this.selectedNode = null;
            if (this.onNodeSelected) {
                this.onNodeSelected(null, null);
            }
        }
    }

    startAnimation() {
        const loop = (time) => {
            const dt = (time - this.lastTime) / 1000.0;
            this.lastTime = time;
            
            this.render(time);
            this.animationFrame = requestAnimationFrame(loop);
        };
        this.animationFrame = requestAnimationFrame(loop);
    }

    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    drawArrow(ctx, fromX, fromY, toX, toY, color) {
        const headlen = 8;
        const dx = toX - fromX;
        const dy = toY - fromY;
        const angle = Math.atan2(dy, dx);
        
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 6), toY - headlen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 6), toY - headlen * Math.sin(angle + Math.PI / 6));
        ctx.lineTo(toX, toY);
        ctx.fillStyle = color;
        ctx.fill();
    }

    render(time) {
        // Clear background (handled by CSS/transparent, but we draw a rect if needed)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.state) return;

        // Draw edges
        if (this.buildingData && this.buildingData.floors && this.buildingData.floors[this.currentFloor]) {
            const floorData = this.buildingData.floors[this.currentFloor];
            if (floorData.edges) {
                for (const edge of floorData.edges) {
                    const src = this.state.nodes[edge.source];
                    const tgt = this.state.nodes[edge.target];
                    if (!src || !tgt) continue;

                    const p1 = this.worldToScreen(src.position[0], src.position[1]);
                    const p2 = this.worldToScreen(tgt.position[0], tgt.position[1]);

                    this.ctx.beginPath();
                    this.ctx.moveTo(p1.x, p1.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    
                    let eState = src.edge_states ? src.edge_states[edge.id] : 'OPEN';
                    if (eState === 'SEVERED') {
                        this.ctx.strokeStyle = '#ef4444'; // Red
                        this.ctx.setLineDash([5, 5]);
                    } else if (eState === 'LOCKED') {
                        this.ctx.strokeStyle = '#8b5cf6'; // Purple
                        this.ctx.setLineDash([]);
                    } else {
                        this.ctx.strokeStyle = 'rgba(255,255,255,0.2)'; // White/open
                        this.ctx.setLineDash([]);
                    }
                    
                    this.ctx.lineWidth = 2;
                    this.ctx.stroke();
                    this.ctx.setLineDash([]);
                }
            }
        }

        // Draw nodes
        for (const [nodeId, nodeData] of Object.entries(this.state.nodes)) {
            if (nodeData.floor !== this.currentFloor) continue;

            const pos = this.worldToScreen(nodeData.position[0], nodeData.position[1]);
            const isHovered = (nodeId === this.hoveredNode);
            const isSelected = (nodeId === this.selectedNode);
            const hazard = nodeData.hazard_score || 0;
            const color = this.getHazardColor(hazard);
            
            // Base size 12px, grow with crowd
            const baseRadius = 12;
            const capacityRatio = nodeData.capacity > 0 ? (nodeData.crowd_count / nodeData.capacity) : 0;
            const radius = baseRadius + (capacityRatio * 10);

            // Hazard pulse effect
            if (hazard > 0.3) {
                const pulseScale = 1 + Math.sin(time / 200) * 0.2 * hazard;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius * pulseScale + 5, 0, Math.PI * 2);
                this.ctx.fillStyle = color;
                this.ctx.globalAlpha = 0.2 * hazard;
                this.ctx.fill();
                this.ctx.globalAlpha = 1.0;
            }

            this.ctx.beginPath();
            
            if (nodeData.node_type === 'EXIT') {
                // Diamond
                this.ctx.moveTo(pos.x, pos.y - radius);
                this.ctx.lineTo(pos.x + radius, pos.y);
                this.ctx.lineTo(pos.x, pos.y + radius);
                this.ctx.lineTo(pos.x - radius, pos.y);
                this.ctx.closePath();
                this.ctx.fillStyle = '#3b82f6'; // Always blue
            } else if (nodeData.node_type === 'STAIRWELL') {
                // Square
                this.ctx.rect(pos.x - radius, pos.y - radius, radius*2, radius*2);
                this.ctx.fillStyle = color;
            } else if (nodeData.node_type === 'ELEVATOR') {
                // Rounded rect
                this.ctx.roundRect(pos.x - radius, pos.y - radius*1.5, radius*2, radius*3, 4);
                this.ctx.fillStyle = color;
            } else {
                // Circle
                this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                this.ctx.fillStyle = color;
            }
            
            this.ctx.fill();
            
            // Intersections get a ring
            if (nodeData.node_type === 'INTERSECTION') {
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius + 2, 0, Math.PI * 2);
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            }

            // Selection / Hover highlights
            if (isSelected) {
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.lineWidth = 3;
                this.ctx.stroke();
            } else if (isHovered) {
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
                this.ctx.strokeStyle = 'rgba(255,255,255,0.5)';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
                
                // Draw tooltip
                this.ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
                this.ctx.fillRect(pos.x + 15, pos.y - 30, 120, 50);
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = '12px Inter';
                this.ctx.fillText(`ID: ${nodeId}`, pos.x + 20, pos.y - 15);
                this.ctx.fillText(`Pop: ${nodeData.crowd_count}/${nodeData.capacity}`, pos.x + 20, pos.y + 2);
                this.ctx.fillText(`Haz: ${(hazard).toFixed(2)}`, pos.x + 20, pos.y + 17);
            }
            
            // Labels for special nodes
            if (['EXIT', 'STAIRWELL', 'ELEVATOR'].includes(nodeData.node_type)) {
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = 'bold 10px Inter';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                let txt = '';
                if (nodeData.node_type === 'EXIT') txt = 'E';
                if (nodeData.node_type === 'STAIRWELL') txt = '↕';
                if (nodeData.node_type === 'ELEVATOR') txt = 'EL';
                this.ctx.fillText(txt, pos.x, pos.y);
            }
        }
        
        // Draw evacuees
        if (this.state.evacuees) {
            for (const ev of this.state.evacuees) {
                if (ev.floor !== this.currentFloor) continue;
                
                const pos = this.worldToScreen(ev.position[0], ev.position[1]);
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, 2, 0, Math.PI * 2);
                
                if (ev.status === 'casualty') {
                    this.ctx.fillStyle = '#ef4444'; // Red
                } else if (ev.status === 'evacuated') {
                    this.ctx.fillStyle = '#3b82f6'; // Blue
                } else {
                    this.ctx.fillStyle = '#10b981'; // Green
                }
                this.ctx.fill();
            }
        }

        // Draw threat icons
        if (this.state.active_threats) {
            for (const threat of this.state.active_threats) {
                if (threat.floor !== this.currentFloor) continue;
                
                const node = this.state.nodes[threat.node_id];
                if (!node) continue;
                
                const pos = this.worldToScreen(node.position[0], node.position[1]);
                
                this.ctx.font = '20px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                let icon = '⚠️';
                if (threat.threat_type === 'FIRE') icon = '🔥';
                if (threat.threat_type === 'COLLAPSE') icon = '🏗️';
                if (threat.threat_type === 'GAS') icon = '💨';
                if (threat.threat_type === 'WATER') icon = '💧';
                if (threat.threat_type === 'ANIMAL') icon = '🦁';
                if (threat.threat_type === 'WEAPON') icon = '⚔️';
                
                // Offset icon slightly up
                this.ctx.fillText(icon, pos.x, pos.y - 20);
            }
        }
    }
}
