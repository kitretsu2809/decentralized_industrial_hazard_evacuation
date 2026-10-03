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

    // JS
    if (!document.getElementById('driver-js-script')) {
      const script = document.createElement('script');
      script.id = 'driver-js-script';
      script.src = 'https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.iife.js';
      script.onload = () => {
        if (callback) callback();
      };
      document.head.appendChild(script);
    } else {
      const existing = document.getElementById('driver-js-script');
      existing.addEventListener('load', () => {
        if (callback) callback();
      });
    }
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
      window.simSocket.send(JSON.stringify({ type: 'set_speed', speed: mult }));
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
        type: 'inject_disaster',
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

    const alarmBtn = document.getElementById('btn-alarm');
    if (alarmBtn && !alarmBtn.classList.contains('active')) {
      alarmBtn.click();
    }
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
              title: 'Step 1: 2.5D Digital Twin & Topology',
              description: 'The industrial facility is modeled as a 3-floor cyber-physical graph with 36 edge router nodes, reinforced stairwells, and emergency exits. Occupants navigate continuous 2D floor plans under the Helbing Social Force Model.',
              side: 'left',
              align: 'start'
            }
          },
          {
            element: '.header-center',
            popover: {
              title: 'Step 2: Real-Time Sensor Telemetry & Mesh Gossip',
              description: 'Each edge router node continuously monitors toxic gas ppm, thermal gradients, and optical crowd density. Routers synchronize states via an ultra-compact 4-byte sparse delta gossip protocol over local ad-hoc wireless mesh.',
              side: 'bottom',
              align: 'center'
            }
          },
          {
            element: '.btn-alarm',
            popover: {
              title: 'Step 3: Catastrophic Disaster Outbreak (Reactor 2)',
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
              title: 'Step 4: Classical Greedy Bottleneck Breakdown',
              description: 'Under classical dynamic shortest paths (D* Lite), all 250 occupants greedily converge on the single surviving stairwell, triggering severe doorway arching (jamming density ρ > 3.5 ped/m²), flow collapse to 0.05 m/s, and lethal crowd asphyxia.',
              side: 'bottom',
              align: 'center'
            }
          },
          {
            element: '.status-pill.iot-pill',
            popover: {
              title: 'Step 5: ST-TBA-GAT Dynamic Flow Splitting',
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
