/**
 * Dependency-free guided demonstration for the industrial safety digital twin.
 * The tour changes real simulator modes; it never substitutes cosmetic UI state
 * for a backend routing decision.
 */
(function () {
  "use strict";

  window.setSimulationSpeed = function (multiplier) {
    const value = Math.max(0.5, Math.min(8, Number(multiplier) || 1));
    const slider = document.getElementById("speed-slider");
    if (slider) {
      slider.value = value;
      slider.dispatchEvent(new Event("input"));
    }
    return value;
  };

  window.injectDisasterScenario = function (scenarioId) {
    const scenarios = {
      scenario_1: {node_id: "reactor_2", hazard_type: "EXPLOSION", intensity: 0.95},
      scenario_2: {node_id: "tank_farm_a", hazard_type: "GAS_RELEASE", intensity: 0.85},
      scenario_3: {node_id: "compressor_shed", hazard_type: "FIRE", intensity: 0.90},
    };
    const scenario = scenarios[scenarioId] || scenarios.scenario_1;
    if (window.simSocket && window.simSocket.readyState === WebSocket.OPEN) {
      window.simSocket.send(JSON.stringify({action: "inject_disaster", ...scenario}));
    } else {
      fetch("/api/disaster", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(scenario),
      }).catch(() => {});
    }
    return scenario;
  };

  const steps = [
    {
      selector: ".twin-stage",
      title: "Facility topology digital twin",
      description: "A three-level industrial graph maps rooms, corridors, stair shafts, exits, personnel, hazards, and directional sign actuators onto one live operational view.",
    },
    {
      selector: ".rail-right",
      title: "Safety outcomes and assurance",
      description: "The operator sees accountability, movement states, active hazard zones, controller source, and safety-interlock status instead of an unexplained model score.",
    },
    {
      selector: ".command-section",
      title: "Exercise incident declaration",
      description: "This demonstration injects a Reactor 2 explosion into the digital twin. Injection atomically raises the facility alarm and starts the evacuation workflow.",
      enter: () => window.injectDisasterScenario("scenario_1"),
    },
    {
      selector: ".policy-section",
      title: "Measured operational baseline",
      description: "Dynamic Dijkstra routing is engaged as the hazard-aware comparison baseline. Its sign commands and pedestrian movement come from the same backend state.",
      enter: () => window.setRoutingMode && window.setRoutingMode("dijkstra"),
    },
    {
      selector: ".policy-section",
      title: "Adaptive ST-TBA-GAT guidance",
      description: "The trained policy is requested for distributed sign control. If its checkpoint is unavailable or rejected, the server retains the safe-path baseline and reports the fallback.",
      enter: () => window.setRoutingMode && window.setRoutingMode("marl"),
    },
  ];

  let activeIndex = -1;
  let entered = new Set();
  let overlay = null;
  let card = null;

  function removeFocus() {
    document.querySelectorAll(".tour-focus").forEach(el => el.classList.remove("tour-focus"));
  }

  function stopTour() {
    removeFocus();
    if (overlay) overlay.remove();
    if (card) card.remove();
    overlay = null;
    card = null;
    activeIndex = -1;
    entered = new Set();
  }

  function renderStep(index) {
    activeIndex = Math.max(0, Math.min(steps.length - 1, index));
    const step = steps[activeIndex];
    removeFocus();
    const target = document.querySelector(step.selector);
    if (target) {
      target.classList.add("tour-focus");
      target.scrollIntoView({behavior: "smooth", block: "center", inline: "center"});
    }
    if (step.enter && !entered.has(activeIndex)) {
      entered.add(activeIndex);
      step.enter();
    }

    card.replaceChildren();
    const meta = document.createElement("div");
    meta.className = "tour-meta";
    meta.textContent = `SYSTEM WALKTHROUGH · ${activeIndex + 1}/${steps.length}`;
    const title = document.createElement("h2");
    title.textContent = step.title;
    const description = document.createElement("p");
    description.textContent = step.description;
    const actions = document.createElement("div");
    actions.className = "tour-actions";

    const close = document.createElement("button");
    close.className = "tour-close";
    close.textContent = "Close";
    close.addEventListener("click", stopTour);
    actions.appendChild(close);

    if (activeIndex > 0) {
      const previous = document.createElement("button");
      previous.textContent = "Back";
      previous.addEventListener("click", () => renderStep(activeIndex - 1));
      actions.appendChild(previous);
    }

    const next = document.createElement("button");
    next.className = "primary";
    next.textContent = activeIndex === steps.length - 1 ? "Finish" : "Next";
    next.addEventListener("click", () => {
      if (activeIndex === steps.length - 1) stopTour();
      else renderStep(activeIndex + 1);
    });
    actions.appendChild(next);
    card.append(meta, title, description, actions);
  }

  window.startAutomatedTour = function () {
    stopTour();
    overlay = document.createElement("div");
    overlay.className = "tour-overlay";
    overlay.addEventListener("click", stopTour);
    card = document.createElement("section");
    card.className = "tour-card";
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-modal", "true");
    card.setAttribute("aria-label", "Guided system demonstration");
    document.body.append(overlay, card);
    renderStep(0);
  };

  window.TourController = {
    start: window.startAutomatedTour,
    inject: window.injectDisasterScenario,
    setSpeed: window.setSimulationSpeed,
    stop: stopTour,
  };
})();
