const canvas = document.getElementById('signal-canvas');
const ctx = canvas.getContext('2d');
let w = 0;
let h = 0;
let t = 0;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  w = canvas.width = Math.floor(window.innerWidth * dpr);
  h = canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawWave(offset, amp, speed, color, width) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  ctx.beginPath();
  for (let x = 0; x <= vw; x += 6) {
    const y = vh * offset + Math.sin(x * 0.014 + t * speed) * amp + Math.sin(x * 0.033 + t * speed * 0.7) * (amp * 0.34);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function animate() {
  t += 0.012;
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.globalCompositeOperation = 'lighter';
  drawWave(0.26, 24, 1.1, 'rgba(79, 255, 212, 0.28)', 1.4);
  drawWave(0.48, 32, 0.8, 'rgba(123, 97, 255, 0.22)', 1.2);
  drawWave(0.68, 26, 1.35, 'rgba(255, 79, 216, 0.18)', 1.2);
  ctx.globalCompositeOperation = 'source-over';
  requestAnimationFrame(animate);
}

resize();
window.addEventListener('resize', resize);
animate();
