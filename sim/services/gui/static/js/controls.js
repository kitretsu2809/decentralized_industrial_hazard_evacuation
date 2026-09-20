class ControlPanel {
    constructor(websocketManager) {
        this.ws = websocketManager;
        this.selectedThreatType = 'GAS';
        this.intensity = 0.85;
        this.spreadRate = 0.20;
        this.selectedNodeId = 'reactor_1';
        this.selectedFloor = 1;
    }

    init() {
        // Threat type select
        const typeSelect = document.getElementById('threat-type');
        if (typeSelect) {
            typeSelect.addEventListener('change', (e) => {
                this.selectedThreatType = e.target.value;
            });
            this.selectedThreatType = typeSelect.value;
        }

        // Equipment node target select
        const nodeSelect = document.getElementById('node-target-select');
        if (nodeSelect) {
            nodeSelect.addEventListener('change', (e) => {
                this.selectedNodeId = e.target.value;
                const opt = e.target.selectedOptions[0];
                if (opt && opt.dataset.floor) {
                    this.selectedFloor = parseInt(opt.dataset.floor);
                }
            });
            this.selectedNodeId = nodeSelect.value;
        }

        // Sliders
        const intensitySlider = document.getElementById('threat-intensity');
        const intensityVal = document.getElementById('intensity-val');
        if (intensitySlider && intensityVal) {
            intensitySlider.addEventListener('input', (e) => {
                this.intensity = parseFloat(e.target.value);
                intensityVal.innerText = this.intensity.toFixed(2);
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

        // Inject button
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

        // Delegated click listener for active disasters list (prevents dropped clicks)
        const listContainer = document.getElementById('active-disasters-list');
        if (listContainer) {
            listContainer.addEventListener('click', async (e) => {
                const btn = e.target.closest('.btn-neutralize');
                if (!btn || btn.disabled) return;
                
                const nodeId = btn.dataset.nodeId;
                const threatType = btn.dataset.threatType;
                const floor = parseInt(btn.dataset.floor) || 1;
                const threatId = btn.dataset.threatId;

                btn.disabled = true;
                btn.innerText = 'Neutralizing...';
                btn.style.opacity = '0.6';

                const card = btn.closest('.disaster-card');
                if (card) {
                    card.style.opacity = '0.35';
                    card.style.transition = 'all 0.2s ease';
                }

                await this.deleteDisaster(nodeId, threatType, floor, threatId);
            });
        }
    }

    onNodeSelected(nodeId, floor) {
        this.selectedNodeId = nodeId;
        this.selectedFloor = floor;

        const nodeSelect = document.getElementById('node-target-select');
        if (nodeSelect) {
            for (let i = 0; i < nodeSelect.options.length; i++) {
                if (nodeSelect.options[i].value === nodeId) {
                    nodeSelect.selectedIndex = i;
                    break;
                }
            }
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
            console.log("Injected disaster:", data);
            
            // Visual feedback on button
            const btn = document.getElementById('btn-inject');
            if (btn) {
                const originalText = btn.innerText;
                btn.innerText = "✓ INJECTED!";
                btn.style.background = "var(--status-safe)";
                setTimeout(() => {
                    btn.innerText = originalText;
                    btn.style.background = "";
                }, 1500);
            }
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
            await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (err) {
            console.error(`Failed to send control ${command}:`, err);
        }
    }

    async deleteDisaster(nodeId, threatType, floor, threatId = null) {
        const payload = {
            threat_id: threatId || "",
            threat_type: threatType,
            target_node: nodeId,
            floor: floor,
            intensity: 0.0,
            spread_rate: 0.0,
            metadata: {}
        };
        try {
            await fetch('/api/inject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (err) {
            console.error("Failed to delete disaster:", err);
        }
    }

    updateState(state) {
        if (!state) return;

        const activeThreats = state.active_threats || [];
        const hasThreats = activeThreats.length > 0;

        // 1. Update DEFCON Industrial Alert Status
        const defconBadge = document.getElementById('defcon-badge');
        const defconText = document.getElementById('defcon-text');
        if (defconBadge && defconText) {
            if (hasThreats) {
                defconBadge.classList.add('alert-active');
                defconText.innerText = `⚠️ EMERGENCY DEFCON-1: ${activeThreats.length} HAZARD(S) ACTIVE`;
            } else {
                defconBadge.classList.remove('alert-active');
                defconText.innerText = "STATUS: NOMINAL PROCESS STANDBY";
            }
        }

        // 2. Update Active Hazards Count
        const countEl = document.getElementById('metric-hazard-count');
        if (countEl) {
            countEl.innerText = `${activeThreats.length} Threats`;
            countEl.style.color = hasThreats ? "#ef4444" : "#10b981";
        }

        // 3. Update Active Disasters List (Stable DOM elements: prevents dropped clicks)
        const listContainer = document.getElementById('active-disasters-list');
        if (listContainer) {
            const threatsKey = activeThreats.map(t => `${t.threat_id || t.node_id}_${t.threat_type}_${t.floor}`).sort().join('|');

            if (threatsKey !== this.lastThreatsKey) {
                this.lastThreatsKey = threatsKey;
                if (!hasThreats) {
                    listContainer.innerHTML = '<span class="helper-text">No active threats detected. Complex nominal.</span>';
                } else {
                    listContainer.innerHTML = '';
                    activeThreats.forEach(threat => {
                        const card = document.createElement('div');
                        card.className = 'disaster-card';
                        
                        let icon = '🔥';
                        let borderCol = 'var(--threat-fire)';
                        if (threat.threat_type.includes('GAS')) { icon = '💨'; borderCol = 'var(--threat-gas)'; }
                        else if (threat.threat_type.includes('SPILL')) { icon = '⚠️'; borderCol = 'var(--threat-spill)'; }
                        else if (threat.threat_type.includes('EXPLOSION')) { icon = '💥'; borderCol = 'var(--threat-blast)'; }
                        else if (threat.threat_type.includes('COLLAPSE')) { icon = '🏗️'; borderCol = 'var(--threat-collapse)'; }
                        
                        card.style.borderLeftColor = borderCol;
                        
                        const info = document.createElement('div');
                        info.className = 'disaster-card-info';
                        info.innerHTML = `<strong>${icon} ${threat.threat_type}</strong>
                            <span>Unit: <code>${threat.node_id}</code> [L${threat.floor}]</span>
                            <small class="threat-intensity-label">Intensity: ${(threat.hazard_score * 100).toFixed(0)}%</small>`;
                        
                        const delBtn = document.createElement('button');
                        delBtn.className = 'btn-neutralize';
                        delBtn.innerText = 'Neutralize';
                        delBtn.title = 'Extinguish / Neutralize this disaster';
                        delBtn.dataset.nodeId = threat.node_id;
                        delBtn.dataset.threatType = threat.threat_type;
                        delBtn.dataset.floor = threat.floor;
                        delBtn.dataset.threatId = threat.threat_id || '';
                        
                        card.appendChild(info);
                        card.appendChild(delBtn);
                        listContainer.appendChild(card);
                    });
                }
            } else if (hasThreats) {
                // Update intensity values smoothly without destroying buttons
                const cards = listContainer.querySelectorAll('.disaster-card');
                activeThreats.forEach((threat, idx) => {
                    if (cards[idx]) {
                        const small = cards[idx].querySelector('.threat-intensity-label');
                        if (small) small.innerText = `Intensity: ${(threat.hazard_score * 100).toFixed(0)}%`;
                    }
                });
            }
        }

        // 4. Update Atmospheric Gas Sensor Array
        let maxGasHazard = 0.0;
        let maxFireHazard = 0.0;
        if (state.nodes) {
            for (const n of Object.values(state.nodes)) {
                if (n.hazard_score > maxGasHazard) maxGasHazard = n.hazard_score;
            }
        }
        for (const t of activeThreats) {
            if (t.threat_type.includes('GAS')) maxGasHazard = Math.max(maxGasHazard, t.hazard_score);
            if (t.threat_type.includes('FIRE') || t.threat_type.includes('EXPLOSION')) maxFireHazard = Math.max(maxFireHazard, t.hazard_score);
        }

        const h2sVal = (maxGasHazard * 48.5).toFixed(1);
        const cl2Val = (maxGasHazard * 14.2).toFixed(1);
        const lelVal = (Math.max(maxGasHazard, maxFireHazard) * 42.0).toFixed(1);
        const heatVal = (0.4 + maxFireHazard * 6.8).toFixed(1);

        const elH2s = document.getElementById('val-h2s');
        if (elH2s) elH2s.innerHTML = `${h2sVal} <small style="font-size:0.6rem;">PPM</small>`;
        const fillH2s = document.getElementById('fill-h2s');
        if (fillH2s) {
            fillH2s.style.width = `${Math.min(100, maxGasHazard * 100)}%`;
            fillH2s.style.background = maxGasHazard > 0.5 ? '#ef4444' : '#84cc16';
        }

        const elCl2 = document.getElementById('val-cl2');
        if (elCl2) elCl2.innerHTML = `${cl2Val} <small style="font-size:0.6rem;">PPM</small>`;
        const fillCl2 = document.getElementById('fill-cl2');
        if (fillCl2) {
            fillCl2.style.width = `${Math.min(100, maxGasHazard * 100)}%`;
            fillCl2.style.background = maxGasHazard > 0.4 ? '#ef4444' : '#a3e635';
        }

        const elLel = document.getElementById('val-lel');
        if (elLel) elLel.innerHTML = `${lelVal} <small style="font-size:0.6rem;">%</small>`;
        const fillLel = document.getElementById('fill-lel');
        if (fillLel) {
            fillLel.style.width = `${Math.min(100, Math.max(maxGasHazard, maxFireHazard) * 100)}%`;
            fillLel.style.background = maxFireHazard > 0.5 ? '#ef4444' : '#f59e0b';
        }

        const elHeat = document.getElementById('val-heat');
        if (elHeat) elHeat.innerHTML = `${heatVal} <small style="font-size:0.6rem;">kW/m²</small>`;
        const fillHeat = document.getElementById('fill-heat');
        if (fillHeat) {
            fillHeat.style.width = `${Math.min(100, (heatVal / 8.0) * 100)}%`;
            fillHeat.style.background = maxFireHazard > 0.5 ? '#ef4444' : '#f59e0b';
        }

        // Update Floating HUD Gas Chip
        const hudGas = document.getElementById('hud-gas-val');
        if (hudGas) {
            hudGas.innerText = `H₂S ${h2sVal} ppm | Cl₂ ${cl2Val} ppm`;
            hudGas.style.color = maxGasHazard > 0.4 ? '#ef4444' : (maxGasHazard > 0.1 ? '#facc15' : '#10b981');
        }

        // Update Floating HUD Wind Chip and Needle
        const hudWind = document.getElementById('hud-wind-val');
        const sideWind = document.getElementById('sidebar-wind-text');
        const needle = document.getElementById('compass-needle');
        if (state.metrics && (state.metrics.wind_speed !== undefined || state.metrics.wind_x !== undefined)) {
            const spd = (state.metrics.wind_speed || 4.2).toFixed(1);
            let ang = 45;
            if (state.metrics.wind_angle !== undefined) {
                ang = Math.round(state.metrics.wind_angle);
            } else if (state.metrics.wind_x !== undefined && state.metrics.wind_y !== undefined) {
                ang = Math.round((Math.atan2(state.metrics.wind_y, state.metrics.wind_x) * 180 / Math.PI + 360) % 360);
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
            
            const windLabel = `${dirStr} (${ang}°) @ ${spd} m/s`;
            if (hudWind) hudWind.innerText = windLabel;
            if (sideWind) sideWind.innerText = windLabel;
            if (needle) needle.style.transform = `rotate(${ang}deg)`;
        }

        // 5. Update Equipment Integrity HUD Table
        const equipRows = document.getElementById('equipment-rows');
        if (equipRows && state.nodes) {
            const units = [
                { id: 'reactor_1', name: 'Reactor R-101', zone: 'Process' },
                { id: 'reactor_2', name: 'Reactor R-102', zone: 'Process' },
                { id: 'pump_house', name: 'Distillation Tower', zone: 'Process' },
                { id: 'tank_farm_a', name: 'Tank Farm TK-01', zone: 'Storage' },
                { id: 'tank_farm_b', name: 'Tank Farm TK-02', zone: 'Storage' },
                { id: 'hazmat_basin', name: 'Hazmat Basin', zone: 'Containment' },
                { id: 'compressor_shed', name: 'Compressor Bay', zone: 'Mechanical' },
                { id: 'muster_point_alpha', name: 'Muster Alpha', zone: 'Safety Exit' },
                { id: 'muster_point_bravo', name: 'Muster Bravo', zone: 'Safety Exit' }
            ];

            let html = '';
            for (const u of units) {
                const node = state.nodes[u.id];
                const hazard = node ? node.hazard_score : 0.0;
                let statusText = 'NOMINAL';
                let statusClass = 'status-nominal';

                if (hazard > 0.7) {
                    statusText = 'CRITICAL';
                    statusClass = 'status-danger';
                } else if (hazard > 0.25) {
                    statusText = 'WARNING';
                    statusClass = 'status-danger';
                }

                // Check if any attached edge is severed
                if (node && node.edge_states) {
                    if (Object.values(node.edge_states).includes('SEVERED')) {
                        statusText = 'SEVERED';
                        statusClass = 'status-severed';
                    }
                }

                html += `<tr>
                    <td><strong>${u.id}</strong><br><small>${u.name}</small></td>
                    <td>${u.zone}</td>
                    <td><span class="status-tag ${statusClass}">${statusText}</span></td>
                    <td class="mono-digits">${(hazard * 100).toFixed(0)}%</td>
                </tr>`;
            }
            equipRows.innerHTML = html;
        }

        // 6. Update Policy Mode & Inference Telemetry in HUD and Deck
        if (state.metrics) {
            const policyVal = document.getElementById('hud-policy-val');
            const latencyVal = document.getElementById('train-val-latency');
            const mode = state.metrics.policy_mode || 'ai';

            if (policyVal) {
                if (mode === 'ai') {
                    policyVal.innerText = 'AI: ST-TBA-GAT';
                    policyVal.style.color = '#38bdf8';
                } else {
                    policyVal.innerText = 'STATIC BASELINE';
                    policyVal.style.color = '#94a3b8';
                }
            }

            if (latencyVal && state.metrics.inference_latency_ms !== undefined) {
                latencyVal.innerText = `${state.metrics.inference_latency_ms.toFixed(1)} ms`;
            }

            // If training status is exposed on websocket metrics
            if (state.metrics.training_status) {
                const ts = state.metrics.training_status;
                const fill = document.getElementById('train-progress-fill');
                const epEl = document.getElementById('train-ep-counter');
                const statText = document.getElementById('train-status-text');
                const dot = document.getElementById('train-dot');
                if (fill && ts.progress !== undefined) fill.style.width = `${Math.round(ts.progress * 100)}%`;
                if (epEl && ts.current_episode !== undefined) epEl.innerText = `Ep: ${ts.current_episode}/${ts.total_episodes}`;
                if (statText) statText.innerText = ts.is_training ? 'TRAINING IN PROGRESS...' : 'MODEL READY (WEIGHTS SYNCED)';
                if (dot) dot.classList.toggle('training', !!ts.is_training);
            }
        }
    }
}
