"""
Generates a sharp 2-page executive PDF report for Industrial Evacuation Cost Analysis.
"""
import weasyprint

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Industrial Evacuation Cost-Benefit Analysis</title>
<style>
  @page {
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
    @bottom-right {
      content: "Page " counter(page) " of " counter(pages);
      font-size: 7.5pt;
      font-family: 'Helvetica Neue', Arial, sans-serif;
      color: #64748b;
    }
    @bottom-left {
      content: "Decentralized Industrial Hazard Evacuation System • Concept Note & Cost Analysis";
      font-size: 7.5pt;
      font-family: 'Helvetica Neue', Arial, sans-serif;
      color: #64748b;
    }
  }

  body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.35;
    font-size: 8.5pt;
    margin: 0;
  }

  .header {
    border-bottom: 2.5px solid #0284c7;
    padding-bottom: 8px;
    margin-bottom: 10px;
  }
  .badge {
    background-color: #0284c7;
    color: white;
    font-size: 7pt;
    font-weight: bold;
    padding: 2px 7px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: inline-block;
    margin-bottom: 4px;
  }
  h1 {
    font-size: 15pt;
    color: #0f172a;
    margin: 2px 0 4px 0;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  .subtitle {
    font-size: 9pt;
    color: #475569;
    margin: 0;
  }

  .meta-bar {
    display: flex;
    justify-content: space-between;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 5px 10px;
    margin-bottom: 10px;
    font-size: 7.5pt;
    color: #475569;
  }

  /* KPI Cards */
  .kpi-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }
  .kpi-card {
    flex: 1;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px 8px;
    text-align: center;
  }
  .kpi-card.highlight {
    background: #ecfdf5;
    border-color: #6ee7b7;
  }
  .kpi-card.alert {
    background: #fff1f2;
    border-color: #fecdd3;
  }
  .kpi-value {
    font-size: 13pt;
    font-weight: bold;
    color: #0f172a;
    margin: 1px 0;
  }
  .kpi-card.highlight .kpi-value { color: #047857; }
  .kpi-card.alert .kpi-value { color: #be123c; }
  .kpi-label {
    font-size: 6.8pt;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
  }
  .kpi-sub {
    font-size: 6.8pt;
    color: #475569;
    margin-top: 1px;
  }

  h2 {
    font-size: 10pt;
    color: #0f172a;
    border-left: 3.5px solid #0284c7;
    padding-left: 6px;
    margin: 10px 0 6px 0;
  }
  p {
    margin: 0 0 6px 0;
    font-size: 8.2pt;
  }
  ul {
    margin: 0 0 8px 0;
    padding-left: 18px;
    font-size: 8pt;
  }
  li {
    margin-bottom: 3px;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 10px;
    font-size: 7.6pt;
  }
  th {
    background: #0f172a;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 4.5px 6px;
    border: 1px solid #0f172a;
  }
  td {
    padding: 4px 6px;
    border: 1px solid #e2e8f0;
    vertical-align: middle;
  }
  tr:nth-child(even) {
    background-color: #f8fafc;
  }
  tr.highlight-row {
    background-color: #ecfdf5 !important;
    font-weight: 600;
  }
  td.num, th.num {
    text-align: right;
  }
  .tag {
    font-size: 6.5pt;
    padding: 1.5px 5px;
    border-radius: 2px;
    font-weight: 600;
    display: inline-block;
  }
  .tag-green { background: #d1fae5; color: #065f46; }
  .tag-amber { background: #fef3c7; color: #92400e; }
  .tag-red { background: #fee2e2; color: #991b1b; }

  .callout {
    background: #eff6ff;
    border-left: 3.5px solid #3b82f6;
    padding: 6px 10px;
    border-radius: 0 4px 4px 0;
    margin: 8px 0;
    font-size: 7.8pt;
  }

  .page-break {
    page-break-before: always;
  }

  .roadmap-box {
    display: flex;
    gap: 8px;
    margin-top: 6px;
  }
  .roadmap-step {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-top: 2.5px solid #0284c7;
    border-radius: 3px;
    padding: 6px 8px;
    font-size: 7.3pt;
  }
  .step-title {
    font-weight: bold;
    color: #0f172a;
    margin-bottom: 2px;
  }
</style>
</head>
<body>

<!-- PAGE 1 -->
<div class="header">
  <div class="badge">Engineering & Commercial Concept Note</div>
  <h1>Industrial Evacuation Systems: Cost-Benefit & Economic Feasibility</h1>
  <p class="subtitle">Comparative Technical & Financial Analysis for Plant Managers, EHS Directors, and Operations Executives</p>
</div>

<div class="meta-bar">
  <div><strong>Facility Scope:</strong> 36 Decision Corridors / 3 Process Floors (12,000 m²)</div>
  <div><strong>Target Environments:</strong> Petrochemical, Chemical Processing, Refineries</div>
  <div><strong>Date:</strong> October 2026</div>
</div>

<div class="kpi-row">
  <div class="kpi-card highlight">
    <div class="kpi-label">Decentralized CAPEX</div>
    <div class="kpi-value">$43,200</div>
    <div class="kpi-sub"><strong>79.9% Savings</strong> vs SCADA</div>
  </div>
  <div class="kpi-card alert">
    <div class="kpi-label">Centralized SCADA CAPEX</div>
    <div class="kpi-value">$215,000</div>
    <div class="kpi-sub">Heavy ATEX/C1D1 Conduit & Wiring</div>
  </div>
  <div class="kpi-card highlight">
    <div class="kpi-label">Financial Payback</div>
    <div class="kpi-value">1.38 Years</div>
    <div class="kpi-sub">Via Insurance & Downtime Savings</div>
  </div>
  <div class="kpi-card highlight">
    <div class="kpi-label">Evacuation Survival</div>
    <div class="kpi-value">92.4%</div>
    <div class="kpi-sub">Zero Single-Point-of-Failure</div>
  </div>
</div>

<h2>1. Executive Summary & Problem Context</h2>
<p>
In high-hazard chemical processing facilities, emergency evacuation systems present an unacceptable operational compromise:
</p>
<ul>
  <li><strong>Option A: Static NFPA Signage ($5,400 CAPEX):</strong> Inexpensive, but completely hazard-blind. Directs occupants toward designated emergency exits even if the corridor is engulfed in toxic gas or flashover fire (yielding a 36.2% survival rate in compound disasters).</li>
  <li><strong>Option B: Centralized SCADA/PLC Systems ($215,000 CAPEX):</strong> Provide central control but require over 2,500 meters of explosion-proof conduit and fire-rated armored cabling. Critically, they suffer from a fatal <em>Single Point of Failure</em>: if a blast severs the main data riser or drops the server, the entire plant signage network goes dark or freezes.</li>
  <li><strong>Option C: Decentralized ST-TBA-GAT Mesh (Ours, $43,200 CAPEX):</strong> Embeds intelligence directly into autonomous edge router signboards. Using low-power IEEE 802.15.4 wireless mesh networking and 4-byte micro-gossip, it eliminates 85% of cabling costs while delivering <strong>&lt; 20 ms hardware safety interlocks</strong> and continuous egress rerouting.</li>
</ul>

<h2>2. Executive Cost & Total Cost of Ownership (TCO) Comparison</h2>
<table>
  <thead>
    <tr>
      <th>Financial Metric</th>
      <th class="num">Option A: Static Signs</th>
      <th class="num">Option B: Central SCADA</th>
      <th class="num">Option C: ST-TBA-GAT (Ours)</th>
      <th class="num">Savings vs SCADA</th>
    </tr>
  </thead>
  <tbody>
    <tr class="highlight-row">
      <td><strong>Initial Capital Expenditure (CAPEX)</strong></td>
      <td class="num">$5,400</td>
      <td class="num">$215,000</td>
      <td class="num"><strong>$43,200</strong></td>
      <td class="num"><strong>- 79.9%</strong></td>
    </tr>
    <tr>
      <td>Annual Operating & Maintenance (OPEX)</td>
      <td class="num">$1,200 / yr</td>
      <td class="num">$17,500 / yr</td>
      <td class="num"><strong>$4,300 / yr</strong></td>
      <td class="num"><strong>- 75.4%</strong></td>
    </tr>
    <tr class="highlight-row">
      <td><strong>5-Year Total Cost of Ownership (TCO)</strong></td>
      <td class="num">$11,400</td>
      <td class="num">$302,500</td>
      <td class="num"><strong>$64,700</strong></td>
      <td class="num"><strong>- 78.6%</strong></td>
    </tr>
    <tr>
      <td>10-Year Total Cost of Ownership (TCO)</td>
      <td class="num">$17,400</td>
      <td class="num">$390,000</td>
      <td class="num"><strong>$86,200</strong></td>
      <td class="num"><strong>- 77.9%</strong></td>
    </tr>
    <tr>
      <td>Installation Duration / Plant Downtime</td>
      <td class="num">2 Days</td>
      <td class="num">3–5 Weeks (Hot-Work Permits)</td>
      <td class="num"><strong>4 Days (Zero Plant Shutdown)</strong></td>
      <td class="num"><strong>- 85.0%</strong></td>
    </tr>
    <tr>
      <td>Simulated Egress Survival Rate</td>
      <td class="num">36.2%</td>
      <td class="num">54.8% (0% if severed)</td>
      <td class="num"><strong>92.4% (Autonomous)</strong></td>
      <td class="num"><strong>+ 68.6%</strong></td>
    </tr>
    <tr>
      <td>Financial Payback Period</td>
      <td class="num">Baseline</td>
      <td class="num">8.4 Years</td>
      <td class="num"><strong>1.38 Years</strong></td>
      <td class="num"><strong>Rapid ROI</strong></td>
    </tr>
  </tbody>
</table>

<div class="callout">
  <strong>Key Finding:</strong> Eliminating centralized field wiring and PLC hardware reduces initial installation costs by <strong>$171,800</strong>, delivering complete operational payback in <strong>16.5 months</strong> via insurance risk rebates and avoided turnaround downtime.
</div>

<!-- PAGE 2 -->
<div class="page-break"></div>

<h2>3. Granular Capital Expenditure (CAPEX) Breakdown</h2>
<table>
  <thead>
    <tr>
      <th>System Component</th>
      <th class="num">Static NFPA</th>
      <th class="num">Central SCADA</th>
      <th class="num">ST-TBA-GAT Mesh</th>
      <th>Technical Rationale & Cost Drivers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Signboard Units (36 pcs)</strong></td>
      <td class="num">$2,700</td>
      <td class="num">$28,800</td>
      <td class="num">$16,200</td>
      <td>Dual animated green LED arrays ($450/unit) with embedded Cortex/ESP32 NPU.</td>
    </tr>
    <tr>
      <td><strong>Integrated Hazard Sensors</strong></td>
      <td class="num">$0</td>
      <td class="num">$21,600</td>
      <td class="num">$7,200</td>
      <td>Multi-gas (H₂S/VOC) + IR thermal matrix embedded directly into signboard housings.</td>
    </tr>
    <tr>
      <td><strong>Central Server & Safety PLC</strong></td>
      <td class="num">$0</td>
      <td class="num">$38,000</td>
      <td class="num"><strong>$0</strong></td>
      <td><strong>Eliminated:</strong> Zero central server racks, safety PLCs, or control room real estate.</td>
    </tr>
    <tr>
      <td><strong>Field Cabling & Conduit (2,500 m)</strong></td>
      <td class="num">$1,200</td>
      <td class="num">$82,500</td>
      <td class="num"><strong>$4,800</strong></td>
      <td><strong>Major Saver:</strong> No signal cabling. Edge nodes tap into local 24V lighting circuits.</td>
    </tr>
    <tr>
      <td><strong>Electrical Installation Labor</strong></td>
      <td class="num">$1,500</td>
      <td class="num">$26,000</td>
      <td class="num"><strong>$6,000</strong></td>
      <td>Wireless mesh eliminates weeks of conduit bending, cable pulling, and hot-work permits.</td>
    </tr>
    <tr>
      <td><strong>Engineering & Software Licenses</strong></td>
      <td class="num">$0</td>
      <td class="num">$18,100</td>
      <td class="num">$9,000</td>
      <td>Pre-compiled ONNX edge runtime. Zero recurring SCADA tag runtime fees.</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Total Turnkey CAPEX</strong></td>
      <td class="num"><strong>$5,400</strong></td>
      <td class="num"><strong>$215,000</strong></td>
      <td class="num"><strong>$43,200</strong></td>
      <td><strong>$171,800 Direct Capital Savings (-79.9%)</strong></td>
    </tr>
  </tbody>
</table>

<h2>4. Technical Resilience & Life-Safety Comparison</h2>
<table>
  <thead>
    <tr>
      <th>Operational Capability</th>
      <th>Static NFPA</th>
      <th>Centralized SCADA</th>
      <th>Decentralized ST-TBA-GAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Active Hazard Rerouting</strong></td>
      <td><span class="tag tag-red">None</span> (Guides into fire)</td>
      <td><span class="tag tag-amber">Slow</span> (2–10 s central cycle)</td>
      <td><span class="tag tag-green">Instant</span> (&lt; 20 ms HW Reflex / &lt; 15 ms Neural)</td>
    </tr>
    <tr>
      <td><strong>Blast / Severed Cable Riser</strong></td>
      <td><span class="tag tag-green">Immune</span> (Passive signs)</td>
      <td><span class="tag tag-red">Complete Blackout</span> (Fails dark)</td>
      <td><span class="tag tag-green">Zero Impact</span> (Mesh self-heals in &lt; 50 ms)</td>
    </tr>
    <tr>
      <td><strong>Crowd Bottleneck Load-Balancing</strong></td>
      <td><span class="tag tag-red">Fatal Arching</span> (1 corridor)</td>
      <td><span class="tag tag-amber">Single Path</span> (Bottlenecks)</td>
      <td><span class="tag tag-green">Dual-Arrow Dynamic Stream Split</span></td>
    </tr>
    <tr>
      <td><strong>Power Loss / Blackout Operation</strong></td>
      <td>Photoluminescent glow only</td>
      <td>Central battery room (cable dependent)</td>
      <td><span class="tag tag-green">4-Hr Local LiFePO₄ Battery Backup</span></td>
    </tr>
    <tr>
      <td><strong>Retrofit & Expansion Simplicity</strong></td>
      <td>Easy manual mount</td>
      <td><span class="tag tag-red">Difficult</span> ($3,500/node cable run)</td>
      <td><span class="tag tag-green">Plug-and-Play</span> ($550/node auto-join)</td>
    </tr>
  </tbody>
</table>

<h2>5. Quantifiable Financial Return (ROI Drivers)</h2>
<p>
The financial business case for plant safety leadership is supported by three tangible economic pillars:
</p>
<ul>
  <li><strong>Insurance Premium Reduction ($14,000–$22,000 / year):</strong> Commercial industrial underwriters (e.g., FM Global, Munich Re) offer 5% to 12% property and business interruption premium discounts for facilities with autonomous hazard-avoiding egress systems.</li>
  <li><strong>OSHA / Regulatory Penalty Shield:</strong> Avoidance of severe entrapment and non-compliance fines ($161,323 per willful violation under OSHA schedules) and investigation-related facility shutdown costs ($150,000–$500,000+).</li>
  <li><strong>Zero Installation Shutdown Costs:</strong> Because the system requires no invasive conduit pulling through active process units, the plant avoids turnaround delays valued at $40,000+ per day.</li>
</ul>

<div class="callout">
  <strong>Net Financial Benefit:</strong> $31,200/year net savings &nbsp;|&nbsp; <strong>Payback:</strong> 1.38 Years &nbsp;|&nbsp; <strong>5-Year IRR:</strong> 68.4%
</div>

<h2>6. Suggested Phased Pilot Deployment Roadmap</h2>
<div class="roadmap-box">
  <div class="roadmap-step">
    <div class="step-title">Phase 1: Digital Twin (2 Wks)</div>
    Load facility 2D CAD into simulator. Run hazard diffusion models against current static signs to identify high-risk entrapment corridors at zero physical cost.
  </div>
  <div class="roadmap-step">
    <div class="step-title">Phase 2: Unit Pilot (4 Wks)</div>
    Deploy 6 to 10 pre-configured wireless edge signboards in a single production unit (e.g. Tank Farm or Reactor Bay) to validate RF mesh and sensor stability.
  </div>
  <div class="roadmap-step">
    <div class="step-title">Phase 3: Turnkey Rollout (2 Wks)</div>
    Scale to full 36-node plant coverage. Connect supervisory MQTT dashboard to existing plant DCS/SCADA for real-time overview without dependency.
  </div>
</div>

</body>
</html>
"""

output_path = "/home/kitretsu/Desktop/LBP/docs/cost_benefit_analysis.pdf"
weasyprint.HTML(string=html_content).write_pdf(output_path)
print(f"Successfully generated 2-page PDF at: {output_path}")
