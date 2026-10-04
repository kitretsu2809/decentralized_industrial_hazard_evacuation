/**
 * Driver.js Interactive Guided Tour Controller for Digital Twin Evacuation System.
 * 
 * Implements 5-step automated presentation tour:
 *  - Step 1: 2.5D multi-story building topology & vertical egress shafts.
 *  - Step 2: Real-time sensor telemetry (CO concentration, crowd pressure, throughput).
 *  - Step 3: Trigger Scenario 1 injection (catastrophic explosion at Reactor 2).
 *  - Step 4: Baseline failure mode (doorway stampede arching and casualties).
 *  - Step 5: ST-TBA-GAT execution (dynamic flow splitting, early diversion, conflict-free egress).
 * 
 * Exposes deterministic control hooks on window:
 *  - window.startAutomatedTour(options)
 *  - window.injectDisasterScenario(scenarioId)
 *  - window.setSimulationSpeed(multiplier)
 */

(function () {
  'use strict';

  // Helper to load Driver.js from CDN if not already loaded
  function loadDriverJs(callback) {
    ensureCustomTheme();

    if (window.driver && window.driver.js) {
      callback();
      return;
    }

    // CSS
    if (!document.getElementById('driver-js-css')) {
      const link = document.createElement('link');
      link.id = 'driver-js-css';
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.css';
      document.head.appendChild(link);
    }

    // Ensure custom theme is always loaded after driver-js-css
    ensureCustomTheme();

    // JS
    if (!document.getElementById('driver-js-script')) {
      const script = document.createElement('script');
      script.id = 'driver-js-script';
      script.src = 'https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.iife.js';
      script.onload = () => {
        ensureCustomTheme();
        if (callback) callback();
      };
      document.head.appendChild(script);
    } else {
      const existing = document.getElementById('driver-js-script');
      existing.addEventListener('load', () => {
        ensureCustomTheme();
        if (callback) callback();
      });
    }
  }

  function ensureCustomTheme() {
    let style = document.getElementById('driver-custom-theme');
    if (!style) {
      style = document.createElement('style');
      style.id = 'driver-custom-theme';
      document.head.appendChild(style);
    }
    style.textContent = `
.driver-popover {
  box-sizing: border-box !important;
  background: rgba(15, 23, 42, 0.96) !important;
  backdrop-filter: blur(20px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
  border: 1px solid rgba(56, 189, 248, 0.38) !important;
  border-radius: 14px !important;
  box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.85), 0 0 24px rgba(56, 189, 248, 0.22) !important;
  color: #e2e8f0 !important;
  padding: 18px 20px 16px !important;
  min-width: 300px !important;
  max-width: 360px !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
}

.driver-popover-title,
header.driver-popover-title,
#driver-popover-title {
  box-sizing: border-box !important;
  font-family: inherit !important;
  font-size: 14.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.25px !important;
  color: #38bdf8 !important;
  background: transparent !important;
  background-color: transparent !important;
  height: auto !important;
  min-height: unset !important;
  max-height: unset !important;
  overflow: visible !important;
  margin: 0 0 10px 0 !important;
  padding: 0 24px 8px 0 !important;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16) !important;
  line-height: 1.45 !important;
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
  text-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
}

.driver-popover-description,
#driver-popover-description {
  box-sizing: border-box !important;
  font-family: inherit !important;
  font-size: 12.5px !important;
  line-height: 1.65 !important;
  color: #cbd5e1 !important;
  background: var(--surface2, #1e293b) !important;
  background-color: var(--surface2, #1e293b) !important;
  padding: 10px 12px !important;
  border-radius: 8px !important;
  border: 1px solid rgba(148, 163, 184, 0.16) !important;
  margin: 0 0 14px 0 !important;
  font-weight: 400 !important;
}

.driver-popover-footer {
  margin-top: 12px !important;
  padding-top: 10px !important;
  border-top: 1px solid rgba(148, 163, 184, 0.12) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
}

.driver-popover-progress-text {
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.4px !important;
  color: #38bdf8 !important;
  background: rgba(56, 189, 248, 0.12) !important;
  border: 1px solid rgba(56, 189, 248, 0.28) !important;
  padding: 3px 9px !important;
  border-radius: 12px !important;
}

.driver-popover-navigation-btns {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
}

.driver-popover-footer button {
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  font-family: inherit !important;
  font-size: 11.5px !important;
  font-weight: 600 !important;
  border-radius: 7px !important;
  padding: 5px 12px !important;
  transition: all 0.18s ease !important;
  text-shadow: none !important;
}

.driver-popover-prev-btn {
  background: rgba(30, 41, 59, 0.85) !important;
  border: 1px solid rgba(148, 163, 184, 0.25) !important;
  color: #94a3b8 !important;
}

.driver-popover-prev-btn:hover {
  background: rgba(51, 65, 85, 0.95) !important;
  border-color: rgba(148, 163, 184, 0.45) !important;
  color: #f1f5f9 !important;
}

.driver-popover-next-btn {
  background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
  border: 1px solid #38bdf8 !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(56, 189, 248, 0.35) !important;
}

.driver-popover-next-btn:hover {
  filter: brightness(1.1) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(56, 189, 248, 0.55) !important;
}

.driver-popover-close-btn {
  all: unset !important;
  box-sizing: border-box !important;
  position: absolute !important;
  top: 10px !important;
  right: 10px !important;
  width: 26px !important;
  height: 26px !important;
  cursor: pointer !important;
  font-size: 16px !important;
  font-weight: 500 !important;
  color: #94a3b8 !important;
  border-radius: 6px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: all 0.18s ease !important;
}

.driver-popover-close-btn:hover {
  color: #ef4444 !important;
  background: rgba(239, 68, 68, 0.14) !important;
}

.driver-popover-arrow {
  content: "" !important;
  position: absolute !important;
  border: 6px solid transparent !important;
}
.driver-popover-arrow-side-left {
  left: 100% !important;
  border-left-color: rgba(15, 23, 42, 0.96) !important;
}
.driver-popover-arrow-side-right {
  right: 100% !important;
  border-right-color: rgba(15, 23, 42, 0.96) !important;
}
.driver-popover-arrow-side-top {
  top: 100% !important;
  border-top-color: rgba(15, 23, 42, 0.96) !important;
}
.driver-popover-arrow-side-bottom {
  bottom: 100% !important;
  border-bottom-color: rgba(15, 23, 42, 0.96) !important;
}

.driver-active-element {
  outline: 2px solid #38bdf8 !important;
  box-shadow: 0 0 20px rgba(56, 189, 248, 0.6) !important;
  border-radius: 6px !important;
}
    `;
    document.head.appendChild(style);
  }

  // ── Global Deterministic Control Hooks ─────────────────────────────────────

  window.setSimulationSpeed = function (multiplier) {
    const mult = Math.max(0.1, Math.min(10.0, Number(multiplier) || 1.0));
    console.log(`[TourController] Setting simulation speed to: ${mult}x`);
    const speedSlider = document.getElementById('speed-slider');
    const speedVal = document.getElementById('speed-val');
    if (speedSlider) {
      speedSlider.value = mult;
      speedSlider.dispatchEvent(new Event('input'));
    }
    if (speedVal) {
      speedVal.textContent = `${mult.toFixed(1)}×`;
    }
    if (window.simSocket && window.simSocket.readyState === WebSocket.OPEN) {
      window.simSocket.send(JSON.stringify({ action: 'set_speed', speed: mult }));
    }
    return mult;
  };

  window.injectDisasterScenario = function (scenarioId) {
    console.log(`[TourController] Injecting disaster scenario: ${scenarioId}`);
    const scenarios = {
      scenario_1: { node_id: 'reactor_2', hazard_type: 'EXPLOSION', intensity: 0.95 },
      scenario_2: { node_id: 'tank_farm_a', hazard_type: 'GAS_RELEASE', intensity: 0.85 },
      scenario_3: { node_id: 'compressor_shed', hazard_type: 'THERMAL_FIRE', intensity: 0.90 }
    };

    const sc = scenarios[scenarioId] || scenarios['scenario_1'];

    // Send disaster injection payload over websocket or HTTP
    if (window.simSocket && window.simSocket.readyState === WebSocket.OPEN) {
      window.simSocket.send(JSON.stringify({
        action: 'inject_disaster',
        node_id: sc.node_id,
        hazard_type: sc.hazard_type,
        intensity: sc.intensity
      }));
    } else {
      fetch('/api/disaster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sc)
      }).catch(err => console.warn('[TourController] Disaster injection fallback fetch:', err));
    }

    // The server arms the evacuation alarm atomically with hazard injection.
    // Sending a second toggle here would race the WebSocket command and can
    // accidentally cancel the alarm during a recorded demonstration.
    return sc;
  };

  window.startAutomatedTour = function (options = {}) {
    loadDriverJs(() => {
      const driverObj = window.driver.js.driver({
        showProgress: true,
        animate: true,
        allowClose: true,
        overlayColor: 'rgba(15, 23, 42, 0.75)',
        nextBtnText: 'Next Step →',
        prevBtnText: '← Previous',
        doneBtnText: 'Finish Tour 🚀',
        steps: [
          {
            element: '.canvas-wrap',
            popover: {
              title: '📐 Step 1: 2.5D Digital Twin & Topology',
              description: 'The industrial facility is modeled as a 3-floor cyber-physical graph with 36 edge router nodes, reinforced stairwells, and emergency exits. Occupants navigate continuous 2D floor plans under the Helbing Social Force Model.',
              side: 'left',
              align: 'start'
            }
          },
          {
            element: '.header-center',
            popover: {
              title: '📡 Step 2: Real-Time Sensor Telemetry & Mesh Gossip',
              description: 'Each edge router node continuously monitors toxic gas ppm, thermal gradients, and optical crowd density. Routers synchronize states via an ultra-compact 4-byte sparse delta gossip protocol over local ad-hoc wireless mesh.',
              side: 'bottom',
              align: 'center'
            }
          },
          {
            element: '.btn-alarm',
            popover: {
              title: '💥 Step 3: Catastrophic Disaster Outbreak (Reactor 2)',
              description: 'Injecting Scenario 1: A severe explosion ruptures Reactor 2 on Floor 1, releasing lethal toxic plumes and cutting off the primary North Stairwell egress shaft.',
              side: 'right',
              align: 'center',
              onHighlightStarted: () => {
                window.injectDisasterScenario('scenario_1');
              }
            }
          },
          {
            element: '#policy-status',
            popover: {
              title: '⚠️ Step 4: Classical Greedy Bottleneck Breakdown',
              description: 'Under the centralized Dijkstra baseline, occupants can converge on the single surviving stairwell, creating doorway arching, flow collapse, and elevated crowd risk. The measured comparison should be used instead of this illustrative scenario text.',
              side: 'bottom',
              align: 'center'
            }
          },
          {
            element: '.status-pill.iot-pill',
            popover: {
              title: '🧠 Step 5: ST-TBA-GAT Dynamic Flow Splitting',
              description: 'ST-TBA-GAT evaluates 1-hop topology-bound attention and GRU temporal velocity to perform game-theoretic mixed-strategy flow splitting. Adjacent signs coordinate early diversions, balancing corridor throughput to ensure conflict-free egress.',
              side: 'bottom',
              align: 'center',
              onHighlightStarted: () => {
                const polText = document.getElementById('policy-status-text');
                if (polText) polText.textContent = 'AI: ST-TBA-GAT (ACTIVE)';
              }
            }
          }
        ]
      });

      driverObj.drive();
    });
  };

  // Expose Tour Controller instance globally
  window.TourController = {
    start: window.startAutomatedTour,
    inject: window.injectDisasterScenario,
    setSpeed: window.setSimulationSpeed
  };

  console.log('✅ Driver.js Tour Controller initialized with window hooks.');
})();
