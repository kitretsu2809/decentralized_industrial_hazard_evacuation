class MetricsPanel {
    constructor() {
        this.previousValues = {
            evacuated: 0,
            casualties: 0,
            congestion: 0
        };
    }

    update(state) {
        if (!state || !state.metrics) return;

        // Evacuated
        const evacuated = Math.floor(state.metrics.evacuated || 0);
        const total = Math.floor(state.metrics.total_people || 0);
        this.animateNumber('metric-evacuated', this.previousValues.evacuated, evacuated, 500);
        
        const totalEl = document.querySelector('#metric-evacuated').nextSibling;
        if (totalEl) totalEl.textContent = `/${total}`;
        
        this.previousValues.evacuated = evacuated;

        // Casualties
        const casualties = Math.floor(state.metrics.casualties || 0);
        this.animateNumber('metric-casualties', this.previousValues.casualties, casualties, 500);
        this.previousValues.casualties = casualties;

        // Congestion
        const congestion = state.metrics.max_congestion || 0;
        const congBar = document.getElementById('metric-congestion');
        if (congBar) {
            congBar.style.width = `${Math.min(congestion * 100, 100)}%`;
            if (congestion > 0.8) {
                congBar.style.background = 'var(--status-danger)';
            } else if (congestion > 0.5) {
                congBar.style.background = 'var(--status-caution)';
            } else {
                congBar.style.background = 'var(--status-safe)';
            }
        }

        // Time
        const elapsed = state.metrics.elapsed_time || 0; // seconds
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = Math.floor(elapsed % 60).toString().padStart(2, '0');
        const timeEl = document.getElementById('metric-time');
        if (timeEl) {
            timeEl.innerText = `${mins}:${secs}`;
        }

        // Active Threats
        const threatsContainer = document.getElementById('active-threats');
        if (threatsContainer && state.active_threats) {
            threatsContainer.innerHTML = '';
            
            // Deduplicate threat types for badges
            const activeTypes = new Set(state.active_threats.map(t => t.threat_type));
            
            activeTypes.forEach(type => {
                const badge = document.createElement('div');
                badge.className = 'threat-badge';
                
                let icon = '⚠️', color = '#ef4444';
                if (type === 'FIRE') { icon = '🔥'; color = 'var(--threat-fire)'; }
                if (type === 'COLLAPSE') { icon = '🏗️'; color = 'var(--threat-collapse)'; }
                if (type === 'GAS') { icon = '💨'; color = 'var(--status-caution)'; }
                if (type === 'WATER') { icon = '💧'; color = 'var(--accent-blue)'; }
                if (type === 'ANIMAL') { icon = '🦁'; color = 'var(--threat-animal)'; }
                if (type === 'WEAPON') { icon = '⚔️'; color = 'var(--threat-weapon)'; }
                
                badge.style.background = `${color}33`; // 20% opacity
                badge.style.color = color;
                badge.style.border = `1px solid ${color}`;
                badge.innerHTML = `${icon} ${type}`;
                
                threatsContainer.appendChild(badge);
            });
        }
    }

    animateNumber(elementId, from, to, duration) {
        const el = document.getElementById(elementId);
        if (!el || from === to) {
            if (el) el.innerText = to;
            return;
        }
        
        const start = performance.now();
        const step = (timestamp) => {
            const progress = Math.min((timestamp - start) / duration, 1);
            const current = Math.floor(from + (to - from) * progress);
            el.innerText = current;
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                el.innerText = to;
            }
        };
        requestAnimationFrame(step);
    }
}
