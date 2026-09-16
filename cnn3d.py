import json

import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components


# --------------------------------------------------
# 1. Activation model
# --------------------------------------------------

def get_activation_model(model):
    """Returns a model whose outputs are every layer's output."""
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in model.layers]
    )


# --------------------------------------------------
# 2. Build the payload sent to the browser
# --------------------------------------------------

def _to_bytes(a, gamma=0.55, lo=None, hi=None):
    """
    Scale an array to 0-255 ints for texture use.

    lo/hi: pass explicit bounds to normalize against a shared scale
    (e.g. across every conv layer) instead of this array's own
    min/max. If omitted, uses the array's own min/max.

    gamma < 1 brightens midtones so low-magnitude activations
    are still visible instead of reading as near-black.
    """
    a = np.asarray(a, dtype="float32")
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi

    if hi - lo < 1e-8:
        norm = np.zeros_like(a)
    else:
        norm = (a - lo) / (hi - lo)

    norm = np.clip(norm, 0.0, 1.0)
    norm = np.power(norm, gamma)

    return (norm * 255).astype("uint8").flatten().tolist()


def _stats(a):
    """min/mean/max of the raw (pre-normalization) values."""
    a = np.asarray(a, dtype="float32")
    return {
        "min": round(float(a.min()), 3),
        "mean": round(float(a.mean()), 3),
        "max": round(float(a.max()), 3)
    }


def build_payload(model, activation_model, image_array):
    """
    image_array: (1, 28, 28, 1) float32, already normalized.
    Returns a dict of layers ready for JSON serialization.

    Conv/pool feature maps share ONE global min/max (computed across
    all of them) so brightness is comparable layer to layer, not just
    channel to channel within a layer. The input image keeps its own
    0-1 pixel scale since it's a different quantity (raw pixels vs
    learned activations) and mixing the two scales would just make
    the input look faint for no informative reason.
    """
    outputs = activation_model.predict(image_array, verbose=0)

    input_img = image_array[0, :, :, 0]
    layers = [{
        "name": "input",
        "kind": "maps",
        "w": 28,
        "h": 28,
        "maps": [_to_bytes(input_img, gamma=1.0)],
        "stats": _stats(input_img)
    }]

    # pass 1: collect every conv/pool output to get one shared scale
    conv_outs = [
        out[0] for layer, out in zip(model.layers, outputs)
        if out[0].ndim == 3
    ]
    if conv_outs:
        g_lo = min(float(a.min()) for a in conv_outs)
        g_hi = max(float(a.max()) for a in conv_outs)
    else:
        g_lo, g_hi = 0.0, 1.0

    for layer, out in zip(model.layers, outputs):
        out = out[0]  # drop batch dimension

        if out.ndim == 3:
            h, w, c = out.shape
            layers.append({
                "name": layer.name,
                "kind": "maps",
                "w": int(w),
                "h": int(h),
                "maps": [
                    _to_bytes(out[:, :, i], lo=g_lo, hi=g_hi)
                    for i in range(c)
                ],
                "stats": _stats(out)
            })

        elif out.ndim == 1 and layer.__class__.__name__ == "Dense":
            v = np.asarray(out, dtype="float32")
            hi = float(v.max())
            hi = hi if hi > 1e-6 else 1.0

            layers.append({
                "name": layer.name,
                "kind": "vector",
                "values": (v / hi).tolist(),  # 0-1, drives bar height/color
                "raw": [round(float(x), 3) for x in v],  # actual numbers to label
                "stats": _stats(v)
            })

    return {"layers": layers, "cmap": "viridis"}


# --------------------------------------------------
# 2b. Kernel weights (what the network learned,
#     separate from activations, which are what fired
#     for one specific input)
# --------------------------------------------------

def build_weight_payload(model):
    """
    Visualizes each Conv2D layer's learned filters as 3x3 grids.

    A conv1 filter is (3,3,1) -> shown directly. A conv2 filter is
    (3,3,16) because it reads all 16 conv1 channels at once; there's
    no single 3x3 picture for that, so it's averaged across the
    input-channel axis to give one representative 3x3 pattern per
    filter. That's a real simplification, flagged in the layer name.

    Weights can be negative, so this uses a diverging (blue-white-red)
    colormap centered on zero rather than the viridis one used for
    activations, which only shows magnitude.
    """
    layers = []

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Conv2D):
            continue

        w = layer.get_weights()[0]  # (kh, kw, in_c, out_c)
        kh, kw, in_c, out_c = w.shape

        filters = w[:, :, 0, :] if in_c == 1 else w.mean(axis=2)
        max_abs = float(np.max(np.abs(filters))) or 1.0

        maps = []
        for i in range(out_c):
            k = filters[:, :, i]
            norm = (k / max_abs + 1.0) / 2.0  # [-max_abs, max_abs] -> [0, 1]
            maps.append((np.clip(norm, 0, 1) * 255).astype("uint8").flatten().tolist())

        name = layer.name + (" kernels (avg over input ch)" if in_c > 1 else " kernels")

        layers.append({
            "name": name,
            "kind": "maps",
            "w": int(kw),
            "h": int(kh),
            "maps": maps,
            "stats": _stats(filters)
        })

    return {"layers": layers, "cmap": "diverging"}


# --------------------------------------------------
# 3. Render
# --------------------------------------------------

_HTML = r"""
<div id="wrap" style="width:100%;height:540px;background:#12161f;border-radius:8px;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __PAYLOAD__;

const wrap = document.getElementById("wrap");
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  45, wrap.clientWidth / wrap.clientHeight, 0.1, 500
);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
wrap.appendChild(renderer.domElement);

// ---------- helpers ----------

// simple viridis-like colormap: dark purple -> teal -> yellow
const CMAP = [
  [0.267,0.004,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
  [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
  [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
  [0.741,0.873,0.150],[0.993,0.906,0.144]
];
function colormap(t) {
  t = Math.max(0, Math.min(1, t));
  const n = CMAP.length - 1;
  const idx = Math.min(n - 1, Math.floor(t * n));
  const frac = t * n - idx;
  const a = CMAP[idx], b = CMAP[idx + 1];
  return [
    a[0] + (b[0]-a[0]) * frac,
    a[1] + (b[1]-a[1]) * frac,
    a[2] + (b[2]-a[2]) * frac
  ];
}

// diverging colormap for weights: blue (negative) -> white (zero) -> red (positive)
function divergingColor(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {
    const f = t / 0.5;
    return [f, f, 1];
  } else {
    const f = (t - 0.5) / 0.5;
    return [1, 1 - f, 1 - f];
  }
}

const ACTIVE_CMAP = DATA.cmap === "diverging" ? divergingColor : colormap;

function mapTexture(arr, w, h) {
  const buf = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // flip vertically: texture origin is bottom-left
      const v = arr[(h - 1 - y) * w + x] / 255;
      const [r, g, b] = ACTIVE_CMAP(v);
      const i = (y * w + x) * 4;
      buf[i] = r * 255; buf[i+1] = g * 255; buf[i+2] = b * 255; buf[i+3] = 255;
    }
  }
  const tex = new THREE.DataTexture(buf, w, h, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

function label(lines, x) {
  const c = document.createElement("canvas");
  c.width = 300; c.height = 90;
  const ctx = c.getContext("2d");
  ctx.textAlign = "center";
  ctx.fillStyle = "#d7e2f0";
  ctx.font = "bold 26px sans-serif";
  ctx.fillText(lines[0], 150, 34);
  if (lines[1]) {
    ctx.fillStyle = "#8ba3c7";
    ctx.font = "20px sans-serif";
    ctx.fillText(lines[1], 150, 62);
  }
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), transparent:true})
  );
  sprite.position.set(x, -3.4, 0);
  sprite.scale.set(4.6, 1.4, 1);
  scene.add(sprite);
}

// small always-visible numeric tag, local to a group
function valueTag(group, text, x, y, z) {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 48;
  const ctx = c.getContext("2d");
  ctx.textAlign = "center";
  ctx.fillStyle = "#e8eef7";
  ctx.font = "bold 24px sans-serif";
  ctx.fillText(text, 64, 32);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), transparent:true})
  );
  sprite.position.set(x, y, z);
  sprite.scale.set(0.9, 0.34, 1);
  group.add(sprite);
  return sprite;
}

// ---------- build layers ----------

const X_STEP = 5;
const groups = [];

DATA.layers.forEach((layer, li) => {
  const g = new THREE.Group();
  g.position.x = li * X_STEP;

  if (layer.kind === "maps") {
    const size = Math.max(layer.w, layer.h) * 0.12;
    const gap = 0.3;
    layer.maps.forEach((arr, j) => {
      const mat = new THREE.MeshBasicMaterial({
        map: mapTexture(arr, layer.w, layer.h),
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
      });
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
      plane.position.z = (j - (layer.maps.length - 1) / 2) * gap;
      g.add(plane);
    });
  } else {
    const n = layer.values.length;
    const spread = Math.min(n * 0.14, 4.5);
    const showTags = n <= 12;  // avoid clutter on the 64-unit hidden layer
    layer.values.forEach((v, j) => {
      const hgt = 0.15 + v * 2.2;
      const [r, gg, b] = colormap(0.15 + v * 0.85);
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(r, gg, b),
        transparent: true,
        opacity: 0
      });
      const zPos = (j - (n - 1) / 2) * (spread / n);
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.12, hgt, 0.12), mat);
      bar.position.set(0, hgt / 2 - 1, zPos);
      g.add(bar);

      if (showTags) {
        const raw = layer.raw[j];
        const text = layer.name.includes("dense_") && n === 10
          ? Math.round(raw * 100) + "%"
          : raw.toFixed(2);
        valueTag(g, text, 0, hgt - 1 + 0.3, zPos);
      }
    });
  }

  scene.add(g);
  groups.push(g);

  const statsLine = layer.stats
    ? `min ${layer.stats.min}  mean ${layer.stats.mean}  max ${layer.stats.max}`
    : null;
  label([layer.name, statsLine], li * X_STEP);
});

// centre the whole stack
const centre = (DATA.layers.length - 1) * X_STEP / 2;
scene.position.x = -centre;

// ---------- camera + orbit ----------

let theta = 0.9, phi = 1.25, radius = DATA.layers.length * 4.5;

function place() {
  camera.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(0, 0, 0);
}
place();

let dragging = false, lx = 0, ly = 0;

renderer.domElement.addEventListener("mousedown", e => {
  dragging = true; lx = e.clientX; ly = e.clientY;
});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  theta -= (e.clientX - lx) * 0.006;
  phi = Math.max(0.15, Math.min(3.0, phi - (e.clientY - ly) * 0.006));
  lx = e.clientX; ly = e.clientY;
  place();
});
renderer.domElement.addEventListener("wheel", e => {
  e.preventDefault();
  radius = Math.max(5, Math.min(160, radius + e.deltaY * 0.05));
  place();
}, {passive:false});

// ---------- reveal animation ----------

const START = performance.now();
const DELAY = 450;   // ms between layers
const FADE = 500;    // ms fade-in

function animate() {
  requestAnimationFrame(animate);
  const t = performance.now() - START;

  groups.forEach((g, i) => {
    const local = Math.max(0, Math.min(1, (t - i * DELAY) / FADE));
    g.children.forEach(child => {
      child.material.opacity = local;
    });
  });

  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
});
</script>
"""


def render(payload, height=560):
    components.html(
        _HTML.replace("__PAYLOAD__", json.dumps(payload)),
        height=height
    )


def render_weights(payload, height=420):
    """Same scene renderer, fed a build_weight_payload() result."""
    components.html(
        _HTML.replace("__PAYLOAD__", json.dumps(payload)),
        height=height
    )