// A generated ambient map for the glass. The browser uses the same bitmap as
// the page's edge glow and as the GPU's ambient texture, so the contour detail
// is actually refracted by the glass instead of sitting behind it as a flat wash.

function parseColor(value) {
  if (!value) return [1, 1, 1, 0];
  const text = value.trim();
  if (text.startsWith('#')) {
    const hex = text.slice(1);
    const n = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex.slice(0, 6);
    return [parseInt(n.slice(0, 2), 16) / 255, parseInt(n.slice(2, 4), 16) / 255, parseInt(n.slice(4, 6), 16) / 255, 1];
  }
  const nums = text.match(/[\d.]+/g);
  if (!nums || nums.length < 3) return [1, 1, 1, 0];
  return [Number(nums[0]) / 255, Number(nums[1]) / 255, Number(nums[2]) / 255, nums.length > 3 ? Number(nums[3]) : 1];
}

function cssColor(style, name) {
  return parseColor(style.getPropertyValue(name));
}

export function createAmbientLayer(host = document.getElementById('ambient')) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  let version = 0;
  const draw = () => {
    const style = getComputedStyle(document.documentElement);
    const theme = document.documentElement.dataset.theme || 'dark';
    const light = theme === 'light';
    const width = Math.max(720, Math.min(1280, Math.round((innerWidth || 1200) / 2)));
    const height = Math.max(420, Math.round(width * (innerHeight || 800) / (innerWidth || 1200)));
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);

    // Keep the centre clear: the ambient only occupies the side bands.
    const mask = context.createLinearGradient(0, 0, width, 0);
    mask.addColorStop(0, 'rgba(0,0,0,1)');
    mask.addColorStop(0.26, 'rgba(0,0,0,1)');
    mask.addColorStop(0.42, 'rgba(0,0,0,0)');
    mask.addColorStop(0.58, 'rgba(0,0,0,0)');
    mask.addColorStop(0.74, 'rgba(0,0,0,1)');
    mask.addColorStop(1, 'rgba(0,0,0,1)');
    context.globalCompositeOperation = 'destination-in';
    context.fillStyle = mask;
    context.fillRect(0, 0, width, height);
    context.globalCompositeOperation = 'source-atop';

    // A little grain keeps the bevel from looking like a flat outline.
    let seed = 0x9e3779b9;
    const random = () => {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
      return (seed >>> 0) / 4294967296;
    };
    for (let i = 0; i < 3200; i++) {
      const x = random() * width;
      const y = random() * height;
      const alpha = random() * (light ? 0.045 : 0.035);
      context.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      context.fillRect(x, y, 1, 1);
    }
    context.globalCompositeOperation = 'source-over';

    if (host) {
      host.style.backgroundImage = `url(${canvas.toDataURL('image/png')})`;
      host.style.backgroundSize = '100% 100%';
      host.style.backgroundRepeat = 'no-repeat';
    }
    version += 1;
    return canvas;
  };

  draw();
  return {
    canvas,
    get version() { return version; },
    refresh: draw,
  };
}
