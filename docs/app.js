const $ = (selector) => document.querySelector(selector);

// Mobile navigation
const menuButton = $(".menu-button");
const nav = $("#site-nav");
menuButton?.addEventListener("click", () => {
  const open = nav.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
});
nav?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  nav.classList.remove("open");
  menuButton?.setAttribute("aria-expanded", "false");
}));

// Scroll reveals
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

// Deterministic illustrative market chart. This is intentionally not presented as live data.
const points = Array.from({ length: 72 }, (_, index) => {
  const wave = Math.sin(index / 5.2) * 13 + Math.sin(index / 2.8) * 4;
  const drift = index * 0.16;
  return 122 - wave - drift + (index > 46 ? (index - 46) * 1.05 : 0);
});
const chartWidth = 640;
const chartHeight = 230;
const chartBottom = 224;
const toPath = (values) => values.map((y, index) => `${index ? "L" : "M"}${(index / (values.length - 1) * chartWidth).toFixed(1)} ${y.toFixed(1)}`).join(" ");
const linePath = toPath(points);
$("#price-path")?.setAttribute("d", linePath);
$("#area-path")?.setAttribute("d", `${linePath} L${chartWidth} ${chartBottom} L0 ${chartBottom} Z`);

const priceDot = $("#price-dot");
const spotPrice = $("#spot-price");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let chartTick = reducedMotion ? points.length - 1 : 0;
const animateChart = () => {
  const index = chartTick % points.length;
  const x = index / (points.length - 1) * chartWidth;
  const y = points[index];
  priceDot?.setAttribute("cx", x.toFixed(1));
  priceDot?.setAttribute("cy", y.toFixed(1));
  if (spotPrice) spotPrice.textContent = `$${(85.420 + (105 - y) * 0.0013).toFixed(3)}`;
  chartTick += 1;
  if (!reducedMotion) window.setTimeout(animateChart, 190);
};
animateChart();

// Illustrative countdown loops so the visual never implies a real market clock.
let secondsLeft = 261;
window.setInterval(() => {
  secondsLeft = secondsLeft > 0 ? secondsLeft - 1 : 899;
  const minutes = Math.floor(secondsLeft / 60);
  const seconds = secondsLeft % 60;
  const countdown = $("#countdown");
  if (countdown) countdown.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}, 1000);

// Fee-aware edge lab using the repository's general event-contract fee logic.
const probabilityInput = $("#probability");
const askInput = $("#ask-price");
const contractsInput = $("#contracts");
const money = (value) => `${value >= 0 ? "+" : "−"}$${Math.abs(value).toFixed(2)}`;
const centicentsUp = (value) => Math.ceil((value - Number.EPSILON) * 10000) / 10000;

function setRangeFill(input) {
  const progress = (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min)) * 100;
  input.style.setProperty("--fill", `${progress}%`);
}

function updateCalculator() {
  const probability = Number(probabilityInput.value) / 100;
  const ask = Number(askInput.value) / 100;
  const contracts = Number(contractsInput.value);
  const fee = centicentsUp(0.07 * contracts * ask * (1 - ask));
  const cost = ask * contracts;
  const expectedPnl = (probability - ask) * contracts - fee;
  const edge = expectedPnl / contracts;
  const breakEven = ask + fee / contracts;
  const qualifies = edge >= 0.08;

  $("#probability-output").textContent = `${Math.round(probability * 100)}%`;
  $("#ask-output").textContent = `${Math.round(ask * 100)}¢`;
  $("#contracts-output").textContent = String(contracts);
  $("#cost-output").textContent = `$${cost.toFixed(2)}`;
  $("#fee-output").textContent = `$${fee.toFixed(2)}`;
  $("#ev-output").textContent = money(expectedPnl);
  $("#breakeven-output").textContent = `${(breakEven * 100).toFixed(1)}%`;
  $("#edge-output").textContent = `${edge >= 0 ? "+" : "−"}${Math.abs(edge * 100).toFixed(2)}¢`;
  const verdict = $("#lab-verdict");
  verdict.textContent = qualifies ? "QUALIFIES" : "PASS";
  verdict.classList.toggle("positive", qualifies);
  verdict.classList.toggle("negative", !qualifies);
  [probabilityInput, askInput, contractsInput].forEach(setRangeFill);
}

[probabilityInput, askInput, contractsInput].forEach((input) => input?.addEventListener("input", updateCalculator));
updateCalculator();
