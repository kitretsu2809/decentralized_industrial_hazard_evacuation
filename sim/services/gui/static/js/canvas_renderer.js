class BuildingRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.currentFloor = 1;
        this.buildingData = null;  // Setup initially
        this.state = null;         // Updated via WS
        
        // Viewport config - centered on Industrial Facility
        this.scale = 3.2;
        this.offset = { x: -410, y: -180 };
        this.hasUserPanned = false;
        this.hasCentered = false;
        
        // Interaction
        this.hoveredNode = null;
        this.selectedNode = null;
        this.onNodeSelected = null; // Callback for UI
        
        // Panning state
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.dragOffsetStart = { x: 0, y: 0 };
        
        this.animationFrame = null;
        
        // Bind events
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this), {passive: false});
        
        // Handle resize
        window.addEventListener('resize', this.resize.bind(this));
        
        // Last update time for interpolation/animation
        this.lastTime = performance.now();
    }

    init(buildingData) {
        this.buildingData = buildingData || {};
        
        // Reconstruct floors structure from flat nodes/edges if missing
        if (this.buildingData && !this.buildingData.floors && this.buildingData.nodes && this.buildingData.edges) {
            this.buildingData.floors = {};
            const nodeFloorMap = {};
            for (const n of this.buildingData.nodes) {
                nodeFloorMap[n.id] = n.floor;
                if (!this.buildingData.floors[n.floor]) {
                    this.buildingData.floors[n.floor] = { level: n.floor, nodes: [], edges: [] };
                }
                this.buildingData.floors[n.floor].nodes.push(n.id);
            }
            for (const e of this.buildingData.edges) {
                const fSrc = nodeFloorMap[e.source];
                const fTgt = nodeFloorMap[e.target];
                if (fSrc && fSrc === fTgt && this.buildingData.floors[fSrc]) {
                    this.buildingData.floors[fSrc].edges.push(e);
                }
            }
        }

        // Collect floor numbers if buildingData has them, else rely on floorManager
        if (this.buildingData && this.buildingData.floors) {
            let keys = Object.keys(this.buildingData.floors).map(k => parseInt(k)).filter(k => !isNaN(k));
            if (keys.length > 0) this.currentFloor = Math.min(...keys);
        }
        
        this.resize();
    }

    initFromState(state) {
        if (!state || !state.nodes) return;
        const nodes = [];
        for (const [id, n] of Object.entries(state.nodes)) {
            nodes.push({
                id: id,
                floor: n.floor,
                position: n.position,
                type: n.node_type || "ROOM"
            });
        }
        this.init({ nodes: nodes, edges: [] });
    }

    setFloor(floor) {
        this.currentFloor = floor;
        this.selectedNode = null; // Deselect on floor switch
        if (!this.hasUserPanned) {
            this.autoFit(false);
        }
    }

    updateState(state) {
        this.state = state;
        if (!this.hasCentered && state && state.nodes && Object.keys(state.nodes).length > 0) {
            this.hasCentered = true;
            this.resize();
            this.autoFit(true);
        }
    }

    autoFit(force = false) {
        if (!this.state || !this.state.nodes) return;
        if (this.hasUserPanned && !force) return;

        const currentNodes = Object.values(this.state.nodes).filter(n => this.currentFloor === 'ALL' || n.floor === this.currentFloor);
        if (currentNodes.length === 0) return;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const n of currentNodes) {
            const x = n.position[0];
            const y = n.position[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        const spanX = Math.max(30, maxX - minX);
        const spanY = Math.max(30, maxY - minY);
        const midX = (minX + maxX) / 2;
        const midY = (minY + maxY) / 2;

        const padX = 110;
        const padY = 80;
        const availW = Math.max(200, this.canvas.width - padX * 2);
        const availH = Math.max(200, this.canvas.height - padY * 2);

        this.scale = Math.min(availW / spanX, availH / spanY);
        this.scale = Math.max(1.8, Math.min(this.scale, 4.0));

        this.offset.x = -midX * this.scale;
        this.offset.y = -midY * this.scale;
        if (force) this.hasUserPanned = false;
    }

    resize() {
        const parent = this.canvas.parentElement;
        if (parent) {
            this.canvas.width = parent.clientWidth;
            this.canvas.height = parent.clientHeight;
            if (!this.hasUserPanned) {
                this.autoFit(false);
            }
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
        
        if (this.isDragging) {
            this.offset.x = this.dragOffsetStart.x + (mx - this.dragStart.x);
            this.offset.y = this.dragOffsetStart.y + (my - this.dragStart.y);
            return;
        }
        
        const worldPos = this.screenToWorld(mx, my);
        
        this.hoveredNode = null;
        
        if (!this.state || !this.state.nodes) return;
        
        for (const [nodeId, nodeData] of Object.entries(this.state.nodes)) {
            if (this.currentFloor !== 'ALL' && nodeData.floor !== this.currentFloor) continue;
            
            const dx = nodeData.position[0] - worldPos.x;
            const dy = nodeData.position[1] - worldPos.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            // Generous hit check radius for industrial units (26 / scale)
            if (dist < 26 / this.scale) {
                this.hoveredNode = nodeId;
                break;
            }
        }
    }

    handleMouseDown(e) {
        // Only start dragging if we didn't click on a node
        if (!this.hoveredNode) {
            this.isDragging = true;
            this.hasUserPanned = true;
            const rect = this.canvas.getBoundingClientRect();
            this.dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            this.dragOffsetStart = { x: this.offset.x, y: this.offset.y };
            this.canvas.style.cursor = 'grabbing';
        }
    }

    handleMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = 'default';
    }

    handleWheel(e) {
        e.preventDefault();
        this.hasUserPanned = true;
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Get world pos before zoom
        const worldPos = this.screenToWorld(mx, my);

        // Zoom in or out
        const zoomFactor = 1.1;
        if (e.deltaY < 0) {
            this.scale *= zoomFactor;
        } else {
            this.scale /= zoomFactor;
        }

        // Clamp scale
        this.scale = Math.max(0.5, Math.min(this.scale, 20.0));

        // Adjust offset so the mouse point stays at the same screen pos
        this.offset.x = mx - this.canvas.width / 2 - (worldPos.x * this.scale);
        this.offset.y = my - this.canvas.height / 2 - (worldPos.y * this.scale);
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
        // Clear background
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 1. Draw Industrial Facility Blueprint Background & Zones
        this.drawIndustrialBlueprint(time);

        if (!this.state) return;

        // 2. Draw Pipes, Catwalks & Directional Signage
        this.drawIndustrialEdges(time);

        // 3. Draw Realistic Animated Hazard Plumes (Gas, Fire, Chemical, Explosion)
        this.drawHazardPlumes(time);

        // 4. Draw Industrial Infrastructure Nodes (Reactors, Tanks, Control Room, Muster Points)
        this.drawIndustrialNodes(time);

        // 5. Draw Industrial Workers (PPE Hardhats & Motion Vectors)
        this.drawIndustrialWorkers(time);

        // 6. Draw Threat Badges & Industrial Telemetry HUD
        this.drawIndustrialHUD(time);
    }

    drawIndustrialBlueprint(time) {
        const ctx = this.ctx;
        
        // Blueprint background fill
        ctx.fillStyle = '#070b14';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Subtle structural grid lines
        ctx.save();
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.04)';
        ctx.lineWidth = 1;
        const gridSize = 40 * (this.scale / 3.0);
        const startX = (this.offset.x + this.canvas.width / 2) % gridSize;
        const startY = (this.offset.y + this.canvas.height / 2) % gridSize;

        for (let x = startX; x < this.canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.canvas.height);
            ctx.stroke();
        }
        for (let y = startY; y < this.canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.canvas.width, y);
            ctx.stroke();
        }

        // Draw Industrial Sector Boundaries on Floor 1
        if (this.currentFloor === 1) {
            this.drawSectorBox(25, 20, 60, 75, 'HAZMAT SECTOR A // TANK FARM', 'rgba(234, 179, 8, 0.15)', '#eab308');
            this.drawSectorBox(95, 22, 80, 72, 'UNIT 4 // CATALYTIC REACTOR YARD', 'rgba(14, 165, 233, 0.15)', '#38bdf8');
            this.drawSectorBox(180, 22, 45, 72, 'SCADA CONTROL & LOGISTICS', 'rgba(99, 102, 241, 0.15)', '#818cf8');
            this.drawSectorBox(105, 0, 50, 16, 'ISO 7010 MUSTER AREA ALPHA', 'rgba(16, 185, 129, 0.22)', '#10b981');
            this.drawSectorBox(105, 100, 50, 18, 'ISO 7010 MUSTER AREA BRAVO', 'rgba(16, 185, 129, 0.22)', '#10b981');
        } else if (this.currentFloor === 2) {
            this.drawSectorBox(90, 25, 125, 50, 'PROCESSING MEZZANINE // CATWALK GALLERY', 'rgba(56, 189, 248, 0.15)', '#38bdf8');
            this.drawSectorBox(205, 45, 30, 20, 'EMERGENCY ESCAPE CHUTE', 'rgba(16, 185, 129, 0.22)', '#10b981');
        } else if (this.currentFloor === 3) {
            this.drawSectorBox(95, 22, 80, 45, 'FLARE HEADER & SCRUBBER PLATFORM', 'rgba(249, 115, 22, 0.15)', '#f97316');
            this.drawSectorBox(180, 42, 35, 25, 'AERIAL RESCUE HELIPAD', 'rgba(16, 185, 129, 0.22)', '#10b981');
        }
        ctx.restore();
    }

    drawSectorBox(wx, wy, ww, wh, label, bgFill, strokeColor) {
        const ctx = this.ctx;
        const p1 = this.worldToScreen(wx, wy);
        const p2 = this.worldToScreen(wx + ww, wy + wh);
        const w = p2.x - p1.x;
        const h = p2.y - p1.y;

        // Sector fill & border
        ctx.fillStyle = bgFill;
        ctx.fillRect(p1.x, p1.y, w, h);

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(p1.x, p1.y, w, h);
        ctx.setLineDash([]);

        // Label banner
        ctx.fillStyle = strokeColor;
        ctx.font = 'bold 9px "JetBrains Mono", monospace';
        ctx.fillText(label, p1.x + 8, p1.y + 14);
    }

    drawIndustrialEdges(time) {
        let edgesToDraw = [];
        if (this.buildingData && this.buildingData.floors && this.buildingData.floors[this.currentFloor]) {
            edgesToDraw = this.buildingData.floors[this.currentFloor].edges || [];
        } else if (this.buildingData && this.buildingData.edges) {
            edgesToDraw = this.buildingData.edges.filter(edge => {
                const src = this.state.nodes[edge.source];
                const tgt = this.state.nodes[edge.target];
                return src && tgt && (this.currentFloor === 'ALL' || (src.floor === this.currentFloor && tgt.floor === this.currentFloor));
            });
        }

        for (const edge of edgesToDraw) {
            const src = this.state.nodes[edge.source];
            const tgt = this.state.nodes[edge.target];
            if (!src || !tgt) continue;

            const p1 = this.worldToScreen(src.position[0], src.position[1]);
            const p2 = this.worldToScreen(tgt.position[0], tgt.position[1]);

            let eState = (src.edge_states && src.edge_states[edge.id]) || (tgt.edge_states && tgt.edge_states[edge.id]) || 'OPEN';
            let isRedirect = (src.sign_directions && src.sign_directions[edge.id] === 'REDIRECT') || 
                             (tgt.sign_directions && tgt.sign_directions[edge.id] === 'REDIRECT');

            const ctx = this.ctx;

            // 1. Base Pipe / Catwalk Channel
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            
            if (eState === 'SEVERED') {
                ctx.strokeStyle = '#ef4444'; // Red ruptured line
                ctx.setLineDash([6, 6]);
                ctx.lineWidth = 3.5;
                ctx.stroke();
                ctx.setLineDash([]);
                
                // Rupture symbol at midpoint
                const mx = (p1.x + p2.x) / 2;
                const my = (p1.y + p2.y) / 2;
                ctx.fillStyle = '#ef4444';
                ctx.font = 'bold 12px Inter';
                ctx.fillText('⚡ SEVERED', mx - 25, my);
                continue;
            } else if (eState === 'LOCKED') {
                ctx.strokeStyle = '#f97316'; // Closed blast isolation valve
                ctx.setLineDash([8, 4]);
                ctx.lineWidth = 4.0;
                ctx.stroke();
                ctx.setLineDash([]);
            } else if (isRedirect) {
                // Active ATEX dynamic redirect route - neon emerald green
                ctx.strokeStyle = '#00e676';
                ctx.lineWidth = 4.0;
                ctx.shadowColor = '#00e676';
                ctx.shadowBlur = 10;
                ctx.stroke();
                ctx.shadowBlur = 0;
            } else {
                // Standard industrial pipe rack / corridor
                ctx.strokeStyle = 'rgba(71, 85, 105, 0.55)';
                ctx.lineWidth = Math.max(2.5, (edge.width || 2.0) * 1.5);
                ctx.stroke();
            }

            // 2. Animated Moving Chevrons & ATEX Signage for REDIRECT
            if (isRedirect && eState === 'OPEN') {
                let fromP = p1;
                let toP = p2;
                if ((tgt.hazard_score || 0) > (src.hazard_score || 0)) {
                    fromP = p2;
                    toP = p1;
                }

                const dx = toP.x - fromP.x;
                const dy = toP.y - fromP.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist > 18) {
                    const ux = dx / dist;
                    const uy = dy / dist;
                    const angle = Math.atan2(dy, dx);

                    // Moving neon chevrons
                    const numChevrons = Math.max(2, Math.floor(dist / 30));
                    const animOffset = ((time / 450) % 1.0);

                    for (let c = 0; c < numChevrons; c++) {
                        const progress = (c / numChevrons + animOffset) % 1.0;
                        const cx = fromP.x + ux * dist * progress;
                        const cy = fromP.y + uy * dist * progress;

                        const chLen = 8;
                        ctx.beginPath();
                        ctx.moveTo(cx - chLen * Math.cos(angle - Math.PI / 5), cy - chLen * Math.sin(angle - Math.PI / 5));
                        ctx.lineTo(cx, cy);
                        ctx.lineTo(cx - chLen * Math.cos(angle + Math.PI / 5), cy - chLen * Math.sin(angle + Math.PI / 5));
                        ctx.strokeStyle = (this.state && this.state.metrics && this.state.metrics.policy_mode === 'ai') ? '#38bdf8' : '#00ff88';
                        ctx.lineWidth = 3.5;
                        ctx.stroke();
                    }

                    // ATEX Explosion-Proof Digital LED Sign at midpoint
                    const midX = (fromP.x + toP.x) / 2;
                    const midY = (fromP.y + toP.y) / 2;
                    const isAI = this.state && this.state.metrics && this.state.metrics.policy_mode === 'ai';
                    const badgeW = isAI ? 116 : 84;
                    const badgeH = 20;

                    ctx.fillStyle = '#0f172a';
                    ctx.strokeStyle = isAI ? '#38bdf8' : '#10b981';
                    ctx.lineWidth = 1.8;
                    ctx.beginPath();
                    ctx.roundRect(midX - badgeW/2, midY - badgeH/2, badgeW, badgeH, 4);
                    ctx.fill();
                    ctx.stroke();

                    ctx.fillStyle = isAI ? '#38bdf8' : '#00ff88';
                    ctx.font = 'bold 9px "JetBrains Mono", monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(isAI ? '◄ AI MUSTER (GAT)' : '◄ MUSTER PATH', midX, midY + 1);
                }
            }
        }
    }

    drawHazardPlumes(time) {
        if (!this.state || !this.state.nodes) return;
        const ctx = this.ctx;

        // Render localized hazard plumes per node
        for (const [nodeId, nodeData] of Object.entries(this.state.nodes)) {
            if (nodeData.floor !== this.currentFloor) continue;
            const haz = nodeData.hazard_score || 0;
            if (haz <= 0.05) continue;

            const pos = this.worldToScreen(nodeData.position[0], nodeData.position[1]);

            // Determine if there is an active threat type registered for this node
            let threatType = 'FIRE';
            if (this.state.active_threats) {
                const threat = this.state.active_threats.find(t => t.node_id === nodeId);
                if (threat) threatType = threat.threat_type;
            }

            ctx.save();
            if (threatType === 'GAS') {
                // TOXIC GAS PLUME: Swirling chartreuse-lime chemical vapor cloud with real-time wind drift
                let wx = 0.707, wy = 0.707, wspd = 4.2;
                if (this.state && this.state.metrics) {
                    if (this.state.metrics.wind_x !== undefined) wx = this.state.metrics.wind_x;
                    if (this.state.metrics.wind_y !== undefined) wy = this.state.metrics.wind_y;
                    if (this.state.metrics.wind_speed !== undefined) wspd = this.state.metrics.wind_speed;
                }
                const windScale = (wspd / 4.2) * 22.0;
                const windDx = (wx * windScale) * haz;
                const windDy = (-wy * windScale) * haz; // negative because screen Y is downwards
                const baseRadius = 25 + (haz * 65);
                
                // Swirling outer vapor cloud
                const grad = ctx.createRadialGradient(pos.x + windDx, pos.y + windDy, 5, pos.x + windDx, pos.y + windDy, baseRadius);
                grad.addColorStop(0, `rgba(163, 230, 53, ${0.45 * haz})`);
                grad.addColorStop(0.5, `rgba(234, 179, 8, ${0.28 * haz})`);
                grad.addColorStop(1, 'rgba(163, 230, 53, 0)');
                
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(pos.x + windDx, pos.y + windDy, baseRadius, 0, Math.PI * 2);
                ctx.fill();

                // Swirling dynamic puffs
                for (let p = 0; p < 4; p++) {
                    const angle = (time / 800) + (p * Math.PI / 2);
                    const px = pos.x + windDx + Math.cos(angle) * (baseRadius * 0.4);
                    const py = pos.y + windDy + Math.sin(angle) * (baseRadius * 0.35);
                    ctx.beginPath();
                    ctx.arc(px, py, baseRadius * 0.35, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(190, 242, 100, ${0.20 * haz})`;
                    ctx.fill();
                }

                // Gas Warning PPM indicator
                ctx.fillStyle = '#facc15';
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.fillText(`Cl2: ${(haz * 450).toFixed(0)} PPM`, pos.x + windDx - 25, pos.y + windDy - baseRadius + 8);

            } else if (threatType === 'CHEMICAL_SPILL') {
                // CORROSIVE CHEMICAL SPILL: Fluid pool expanding on floor
                const poolRadius = 20 + (haz * 40);
                const grad = ctx.createRadialGradient(pos.x, pos.y, 4, pos.x, pos.y, poolRadius);
                grad.addColorStop(0, 'rgba(250, 204, 21, 0.7)');
                grad.addColorStop(0.7, 'rgba(234, 88, 12, 0.4)');
                grad.addColorStop(1, 'rgba(234, 88, 12, 0)');
                
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.ellipse(pos.x, pos.y, poolRadius, poolRadius * 0.7, 0, 0, Math.PI * 2);
                ctx.fill();

            } else if (threatType === 'EXPLOSION' || threatType === 'COLLAPSE') {
                // BLAST SHOCKWAVE: Expanding blast ring & structural debris
                const pulse = ((time / 300) % 1.0);
                const blastRadius = (20 + (haz * 50)) * pulse;
                ctx.strokeStyle = `rgba(239, 68, 68, ${1.0 - pulse})`;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, blastRadius, 0, Math.PI * 2);
                ctx.stroke();

            } else {
                // FLASH FIRE: Thermal combustion flame & heat radiation glow
                const flameRadius = 22 + (haz * 55);
                const pulseScale = 1 + Math.sin(time / 180) * 0.15;
                const grad = ctx.createRadialGradient(pos.x, pos.y, 2, pos.x, pos.y, flameRadius * pulseScale);
                grad.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
                grad.addColorStop(0.2, 'rgba(253, 224, 71, 0.75)');
                grad.addColorStop(0.55, 'rgba(249, 115, 22, 0.55)');
                grad.addColorStop(0.85, 'rgba(239, 68, 68, 0.35)');
                grad.addColorStop(1, 'rgba(239, 68, 68, 0)');
                
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, flameRadius * pulseScale, 0, Math.PI * 2);
                ctx.fill();

                // Thermal radiation label
                ctx.fillStyle = '#f87171';
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.fillText(`${(haz * 65).toFixed(0)} kW/m²`, pos.x - 20, pos.y - flameRadius + 6);
            }
            ctx.restore();
        }
    }

    drawIndustrialNodes(time) {
        const ctx = this.ctx;
        for (const [nodeId, nodeData] of Object.entries(this.state.nodes)) {
            if (nodeData.floor !== this.currentFloor) continue;

            const pos = this.worldToScreen(nodeData.position[0], nodeData.position[1]);
            const isHovered = (nodeId === this.hoveredNode);
            const isSelected = (nodeId === this.selectedNode);
            const hazard = nodeData.hazard_score || 0;
            const pop = nodeData.crowd_count || 0;
            const cap = nodeData.capacity || 10;

            ctx.save();

            // 1. Draw Custom Industrial Vector Equipment
            if (nodeId.includes('reactor')) {
                // CATALYTIC REACTOR: Cylindrical pressure vessel with dome head and hazard diamond
                const rw = 22;
                const rh = 30;
                ctx.fillStyle = '#1e293b';
                ctx.strokeStyle = isSelected ? '#ffffff' : (hazard > 0.3 ? '#ef4444' : '#38bdf8');
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.roundRect(pos.x - rw/2, pos.y - rh/2, rw, rh, 8);
                ctx.fill();
                ctx.stroke();

                // Reactor Dome Lines
                ctx.beginPath();
                ctx.arc(pos.x, pos.y - rh/4, rw/2 - 2, Math.PI, 0);
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
                ctx.stroke();

                // NFPA 704 Diamond
                this.drawNFPADiamond(pos.x, pos.y + 4, 7);

                // Label
                ctx.fillStyle = '#e2e8f0';
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(nodeId.toUpperCase(), pos.x, pos.y + rh/2 + 12);

            } else if (nodeId.includes('tank_farm')) {
                // STORAGE TANK: Double-containment concentric steel tank
                const tr = 18;
                ctx.fillStyle = '#0f172a';
                ctx.strokeStyle = isSelected ? '#ffffff' : (hazard > 0.3 ? '#ef4444' : '#eab308');
                ctx.lineWidth = 2.2;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, tr, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Containment dike outer ring
                ctx.setLineDash([4, 3]);
                ctx.strokeStyle = 'rgba(234, 179, 8, 0.6)';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, tr + 5, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);

                // Tank content icon
                ctx.fillStyle = '#facc15';
                ctx.font = 'bold 10px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🛢️', pos.x, pos.y);

                ctx.fillStyle = '#e2e8f0';
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.fillText(nodeId.toUpperCase(), pos.x, pos.y + tr + 12);

            } else if (nodeId === 'control_room') {
                // SCADA CONTROL CENTER: Command console with illuminated screens
                const cw = 28;
                const ch = 20;
                ctx.fillStyle = '#0f172a';
                ctx.strokeStyle = isSelected ? '#ffffff' : '#818cf8';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.roundRect(pos.x - cw/2, pos.y - ch/2, cw, ch, 4);
                ctx.fill();
                ctx.stroke();

                // Dual Monitors
                ctx.fillStyle = '#38bdf8';
                ctx.fillRect(pos.x - cw/2 + 4, pos.y - ch/2 + 4, 8, 6);
                ctx.fillRect(pos.x + 2, pos.y - ch/2 + 4, 8, 6);

                ctx.fillStyle = '#c7d2fe';
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('SCADA', pos.x, pos.y + ch/2 + 10);

            } else if (nodeData.node_type === 'EXIT' || nodeId.includes('muster')) {
                // ISO 7010 EMERGENCY MUSTER ASSEMBLY POINT
                const mw = 32;
                const mh = 22;
                ctx.fillStyle = '#059669'; // ISO Green
                ctx.strokeStyle = isSelected ? '#ffffff' : '#34d399';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.roundRect(pos.x - mw/2, pos.y - mh/2, mw, mh, 4);
                ctx.fill();
                ctx.stroke();

                // ISO Running man icon & exit arrow
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🏃 ➔', pos.x, pos.y);

                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.fillStyle = '#34d399';
                ctx.fillText('MUSTER POINT', pos.x, pos.y + mh/2 + 10);

            } else if (nodeData.node_type === 'STAIRWELL') {
                // INDUSTRIAL STEEL EGRESS STAIRWELL: Steel grating pattern
                const sw = 20;
                ctx.fillStyle = '#1e293b';
                ctx.strokeStyle = isSelected ? '#ffffff' : '#94a3b8';
                ctx.lineWidth = 2;
                ctx.fillRect(pos.x - sw/2, pos.y - sw/2, sw, sw);
                ctx.strokeRect(pos.x - sw/2, pos.y - sw/2, sw, sw);

                // Grate treads
                ctx.strokeStyle = 'rgba(255,255,255,0.4)';
                ctx.lineWidth = 1;
                for (let t = -sw/2 + 4; t < sw/2; t += 5) {
                    ctx.beginPath();
                    ctx.moveTo(pos.x - sw/2 + 2, pos.y + t);
                    ctx.lineTo(pos.x + sw/2 - 2, pos.y + t);
                    ctx.stroke();
                }

                ctx.fillStyle = '#94a3b8';
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('STAIRS', pos.x, pos.y + sw/2 + 10);

            } else if (nodeId.includes('shower')) {
                // DECONTAMINATION SHOWER
                ctx.fillStyle = '#065f46';
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 14, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 12px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🚿', pos.x, pos.y);

            } else {
                // Standard Pipe Junction / Catwalk Node
                const rad = 10 + (pop / cap) * 6;
                ctx.fillStyle = this.getHazardColor(hazard);
                ctx.strokeStyle = isSelected ? '#ffffff' : '#64748b';
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, rad, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            // Occupancy Badge if populated
            if (pop > 0) {
                const badgeW = 24;
                const badgeH = 12;
                ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
                ctx.strokeStyle = pop > cap ? '#ef4444' : '#38bdf8';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(pos.x + 8, pos.y - 16, badgeW, badgeH, 3);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = pop > cap ? '#f87171' : '#38bdf8';
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`${pop}`, pos.x + 8 + badgeW/2, pos.y - 16 + badgeH/2);
            }

            // Selection / Hover rings
            if (isSelected) {
                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 24, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            } else if (isHovered) {
                ctx.strokeStyle = 'rgba(255,255,255,0.7)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
                ctx.stroke();

                // Detailed Tooltip
                this.drawNodeTooltip(pos.x, pos.y, nodeId, nodeData, hazard, pop, cap);
            }

            ctx.restore();
        }
    }

    drawNFPADiamond(cx, cy, size) {
        const ctx = this.ctx;
        // Health (Blue, Left)
        ctx.fillStyle = '#2563eb';
        ctx.beginPath();
        ctx.moveTo(cx - size, cy); ctx.lineTo(cx, cy - size); ctx.lineTo(cx, cy); ctx.lineTo(cx - size/2, cy + size/2);
        ctx.fill();
        // Flammability (Red, Top)
        ctx.fillStyle = '#dc2626';
        ctx.beginPath();
        ctx.moveTo(cx, cy - size); ctx.lineTo(cx + size, cy); ctx.lineTo(cx, cy);
        ctx.fill();
        // Instability (Yellow, Right)
        ctx.fillStyle = '#ca8a04';
        ctx.beginPath();
        ctx.moveTo(cx + size, cy); ctx.lineTo(cx, cy + size); ctx.lineTo(cx, cy);
        ctx.fill();
    }

    drawNodeTooltip(x, y, nodeId, data, haz, pop, cap) {
        const ctx = this.ctx;
        const tw = 160;
        const th = 68;
        const tx = x + 16;
        const ty = y - 35;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(tx, ty, tw, th, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`ID: ${nodeId}`, tx + 10, ty + 18);

        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`Type: ${data.node_type} (Floor ${data.floor})`, tx + 10, ty + 32);
        ctx.fillText(`Personnel: ${pop} / ${cap}`, tx + 10, ty + 46);

        ctx.fillStyle = haz > 0.5 ? '#ef4444' : (haz > 0.2 ? '#f59e0b' : '#10b981');
        ctx.fillText(`Hazard Score: ${(haz * 100).toFixed(0)}%`, tx + 10, ty + 60);
    }

    drawIndustrialWorkers(time) {
        if (!this.state || !this.state.evacuees) return;
        const ctx = this.ctx;

        for (const ev of this.state.evacuees) {
            if (ev.floor !== this.currentFloor) continue;
            const pos = this.worldToScreen(ev.position[0], ev.position[1]);

            ctx.save();
            if (ev.status === 'casualty') {
                // Incapacitated worker / casualty: Red triage cross
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
                ctx.fill();
                // Triage cross
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(pos.x - 3, pos.y); ctx.lineTo(pos.x + 3, pos.y);
                ctx.moveTo(pos.x, pos.y - 3); ctx.lineTo(pos.x, pos.y + 3);
                ctx.stroke();

            } else if (ev.status === 'evacuated') {
                // Mustered safely at exit: ISO Blue worker with safety halo
                ctx.fillStyle = '#3b82f6';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 4.5, 0, Math.PI * 2);
                ctx.fill();

            } else {
                // ACTIVE INDUSTRIAL WORKER: Safety hardhat with brim & visor
                const helmetRadius = 4.0;
                
                // Yellow Safety Hardhat
                ctx.fillStyle = '#facc15';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, helmetRadius, 0, Math.PI * 2);
                ctx.fill();

                // Hardhat brim & reflective silver stripe
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y - 1, helmetRadius - 1, 0, Math.PI);
                ctx.stroke();

                // Small directional flow chevron when moving
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, helmetRadius + 2, 0, Math.PI * 2);
                ctx.stroke();
            }
            ctx.restore();
        }
    }

    drawIndustrialHUD(time) {
        const ctx = this.ctx;
        ctx.save();

        // Top-Right Industrial Telemetry HUD Box
        const hudW = 250;
        const hudH = 106;
        const hudX = this.canvas.width - hudW - 15;
        const hudY = 15;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(hudX, hudY, hudW, hudH, 6);
        ctx.fill();
        ctx.stroke();

        // Facility Header
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText('INDUSTRIAL TELEMETRY // UNIT 4', hudX + 12, hudY + 18);

        // Wind Vector Indicator
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px "JetBrains Mono", monospace';
        let windStr = 'WIND: 4.2 m/s @ NE (45°)';
        if (this.state && this.state.metrics) {
            const spd = (this.state.metrics.wind_speed || 4.2).toFixed(1);
            let ang = 45;
            if (this.state.metrics.wind_angle !== undefined) {
                ang = Math.round(this.state.metrics.wind_angle);
            } else if (this.state.metrics.wind_x !== undefined && this.state.metrics.wind_y !== undefined) {
                ang = Math.round((Math.atan2(this.state.metrics.wind_y, this.state.metrics.wind_x) * 180 / Math.PI + 360) % 360);
            }
            let dirStr = 'NE';
            if (ang >= 337.5 || ang < 22.5) dirStr = 'E';
            else if (ang < 67.5) dirStr = 'NE';
            else if (ang < 112.5) dirStr = 'N';
            else if (ang < 157.5) dirStr = 'NW';
            else if (ang < 202.5) dirStr = 'W';
            else if (ang < 247.5) dirStr = 'SW';
            else if (ang < 292.5) dirStr = 'S';
            else dirStr = 'SE';
            windStr = `WIND: ${spd} m/s @ ${dirStr} (${ang}°)`;
        }
        ctx.fillText(windStr, hudX + 12, hudY + 36);

        // Gas Concentration Sensor
        let maxHaz = 0;
        let dominantThreat = 'NORMAL';
        if (this.state && this.state.nodes) {
            for (const n of Object.values(this.state.nodes)) {
                if ((n.hazard_score || 0) > maxHaz) maxHaz = n.hazard_score;
            }
        }
        if (this.state && this.state.active_threats && this.state.active_threats.length > 0) {
            dominantThreat = this.state.active_threats[0].threat_type;
        }

        if (maxHaz > 0.1) {
            ctx.fillStyle = maxHaz > 0.5 ? '#ef4444' : '#eab308';
            ctx.fillText(`STATUS: ALERT - ${dominantThreat} (${(maxHaz * 500).toFixed(0)} PPM)`, hudX + 12, hudY + 54);
        } else {
            ctx.fillStyle = '#10b981';
            ctx.fillText('STATUS: ALL SECTORS SAFE (0 PPM)', hudX + 12, hudY + 54);
        }

        // Active Signage Count
        let activeSigns = 0;
        if (this.buildingData && this.buildingData.edges && this.state && this.state.nodes) {
            for (const n of Object.values(this.state.nodes)) {
                if (n.sign_directions) {
                    for (const s of Object.values(n.sign_directions)) {
                        if (s === 'REDIRECT') activeSigns++;
                    }
                }
            }
        }
        ctx.fillStyle = '#34d399';
        ctx.fillText(`DYNAMIC SIGNAGE: ${activeSigns > 0 ? activeSigns + ' ACTIVE' : 'STANDBY'}`, hudX + 12, hudY + 72);

        // Routing Policy Mode & Latency
        const isAI = this.state && this.state.metrics && this.state.metrics.policy_mode === 'ai';
        const lat = (this.state && this.state.metrics && this.state.metrics.inference_latency_ms !== undefined) ? 
                    `${this.state.metrics.inference_latency_ms.toFixed(1)}ms` : '1.8ms';
        ctx.fillStyle = isAI ? '#38bdf8' : '#94a3b8';
        ctx.fillText(`ROUTING: ${isAI ? 'AI GAT-GRU (' + lat + ')' : 'STATIC BASELINE'}`, hudX + 12, hudY + 90);

        ctx.restore();
    }
}

