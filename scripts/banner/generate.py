#!/usr/bin/env python3
"""Generate the animated GitHub profile banners.

Run from the repository root:
    python scripts/banner/generate.py

Requires: pip install -r scripts/banner/requirements.txt
Requires: a portrait photo at assets/source/portrait.png
"""

from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/portrait.png"
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"

W, H = 1180, 610
INTRO_SECONDS = 3.2
TRANSITION_SECONDS = 1.3
LOGO_HOLD_SECONDS = 4.0
TRAVELLER_COUNT = 900
HOLD_PARTICLE_COUNT = 2_400
SEED = 314159
LOGO_SUFFIXES = {".png", ".webp"}
LOGO_ALIASES = {"kube": "kubernetes"}
PREFERRED_LOGO_ORDER = ("teverde",)

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Pedro Giovanni Ricra Figueroa"),
    (1, "role", "Ing. de Sistemas · Full-Stack Developer"),
    (1, "focus", "Angular · Spring Boot · Node.js · MySQL · REST APIs"),
    (0, "stack", ""),
    (2, "frontend", "Angular 19 + Material"),
    (2, "backend", "Node.js + Express · Spring Boot + JPA"),
    (2, "database", "MySQL · SQL Server"),
    (2, "tools", "Git · GitHub · Docker · Postman"),
    (0, "contact", ""),
    (1, "email", "pedro.ricra.figueroa@gmail.com"),
    (1, "linkedin", "/in/pedro-giovanni-ricra-figueroa"),
    (1, "github", "TeVerde29"),
    (1, "city", "Pucallpa"),
]

THEMES = {
    "dark": {
        "bg":      "#0B1410",
        "panel":   "#101F16",
        "panel2":  "#14281B",
        "line":    "#274434",
        "muted":   "#7FA08A",
        "text":    "#EFF6EC",
        "portrait":"#A8D86B",   # te verde claro
        "chrome":  "#C9E8B8",   # te verde niebla
        "accent":  "#A8D86B",
        "shadow":  "#02080A",
    },
    "light": {
        "bg":      "#EFF5E8",
        "panel":   "#FFFFFF",
        "panel2":  "#E4F0D8",
        "line":    "#C2D9AF",
        "muted":   "#6B8A6B",
        "text":    "#1B2E1B",
        "portrait":"#3E7A34",
        "chrome":  "#5A8F4C",
        "accent":  "#3E7A34",
        "shadow":  "#B9CFAA",
    },
}


def edge_connected(mask: np.ndarray) -> np.ndarray:
    """Return true values in ``mask`` that connect to an image edge."""
    connected = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = mask.shape
    for x in range(width):
        queue.extend(((0, x), (height - 1, x)))
    for y in range(height):
        queue.extend(((y, 0), (y, width - 1)))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not mask[y, x]:
            continue
        connected[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width:
                queue.append((ny, nx))
    return connected


def normalize_logo(image: Image.Image) -> Image.Image:
    """Centre an icon on the sampling canvas, extracting opaque white backgrounds."""
    icon = image.convert("RGBA")
    alpha = icon.getchannel("A")
    if alpha.getextrema() == (255, 255):
        # Remove only light pixels connected to the canvas edge. This keeps light
        # details enclosed by a dark outline (for example, Linux's belly) opaque.
        gray = np.asarray(ImageOps.grayscale(icon))
        background = edge_connected(gray >= 220)
        visible = ~background
        # Close tiny anti-aliased gaps in the outline, then fill enclosed light
        # areas so an opaque source becomes a complete single-colour silhouette.
        visible = np.asarray(
            Image.fromarray((visible * 255).astype("uint8"))
            .filter(ImageFilter.MaxFilter(3))
        ) > 0
        exterior = edge_connected(~visible)
        icon.putalpha(Image.fromarray(np.where(exterior, 0, 255).astype("uint8")))
    icon.thumbnail((370, 370), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    offset = ((400 - icon.width) // 2, (400 - icon.height) // 2)
    canvas.alpha_composite(icon, offset)
    if canvas.getchannel("A").getbbox() is None:
        raise ValueError("icon has no visible pixels")
    return canvas


def load_logo_files() -> dict[str, Image.Image]:
    """Load PNG and WebP silhouettes supplied in scripts/banner/logos."""
    loaded: dict[str, Image.Image] = {}
    for path in sorted(LOGOS.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LOGO_SUFFIXES:
            continue
        name = LOGO_ALIASES.get(path.stem.lower(), path.stem.lower())
        with Image.open(path) as image:
            loaded[name] = normalize_logo(image)
    return loaded


def make_logos() -> dict[str, Image.Image]:
    """Load user-supplied PNG/WebP icon files in a stable animation order."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    logos = load_logo_files()
    if not logos:
        raise SystemExit(f"No PNG or WebP icons found in {LOGOS.relative_to(ROOT)}")
    ordered_names = [name for name in PREFERRED_LOGO_ORDER if name in logos]
    ordered_names.extend(name for name in sorted(logos) if name not in ordered_names)
    return {name: logos[name] for name in ordered_names}


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg diffusion; True means a lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Return sampled x/y banner coordinates from a 340x385 dither grid."""
    source = Image.open(SOURCE).convert("RGBA")
    # Tighter head + shoulders crop so face detail fills the VISUAL.MAP frame.
    w, h = source.size
    crop_w = int(w * 0.80)
    crop_h = int(crop_w * (385 / 340))
    left = int(w * 0.10)
    top = int(h * 0.048)
    crop = source.crop((left, top, left + crop_w, top + crop_h)).resize((340, 385), Image.Resampling.LANCZOS)
    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    # Put the portrait over a solid background to process lighting
    if theme == "dark":
        bg = Image.new("RGBA", crop.size, "black")
        bg.alpha_composite(crop)
        prepared = ImageOps.grayscale(bg.convert("RGB"))
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
        prepared = ImageEnhance.Brightness(prepared).enhance(1.05)
        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=1))
        select_lit = True
    else:
        bg = Image.new("RGBA", crop.size, "white")
        bg.alpha_composite(crop)
        prepared = ImageOps.grayscale(bg.convert("RGB"))
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
        prepared = ImageEnhance.Brightness(prepared).enhance(1.05)
        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=1))
        select_lit = False

    bits = floyd_steinberg(np.asarray(prepared))
    active = bits if select_lit else ~bits
    if theme == "dark":
        active &= alpha > 0.08

    # Keep the full 300×340 lattice — skipping 2×2 cells was the soft/blurry look.
    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack((74 + xs, 139 + ys)).astype(np.float32)
    if len(points) > 18000:
        points = points[rng.choice(len(points), 18000, replace=False)]
    return points


def logo_silhouette_points(image: Image.Image) -> np.ndarray:
    """Return every visible logo pixel in the portrait frame's coordinate space."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    return np.column_stack((60 + xs * 0.92, 147 + ys * 0.92)).astype(np.float32)


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Choose travellers from a silhouette in the portrait frame's visual space."""
    points = logo_silhouette_points(image)
    if not len(points):
        raise ValueError("icon has no visible pixels")
    chosen = rng.choice(len(points), count, replace=len(points) < count)
    return points[chosen]


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def time_num(value: float) -> str:
    """Format SMIL timeline values without collapsing adjacent keyframes."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def particle_path(points: np.ndarray) -> str:
    """Render independent one-pixel particles without joining adjacent dots."""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    return "".join(f"M{x} {y}h1" for x, y in unique)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Stable monospace width used both for textLength and leader placement."""
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_points: dict[str, np.ndarray],
    logo_hold_particles: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    targets: list[np.ndarray] = []
    current = source
    for name, points in logo_points.items():
        current = transport(current, points[:n])
        targets.append(current)

    # Three seconds of portrait, then transitions and full-logo holds. Returning
    # to the portrait keeps the loop seamless.
    times = [0.0, 3.0]
    frames = [source, source]
    for target in targets:
        times.extend((times[-1] + TRANSITION_SECONDS,
                      times[-1] + TRANSITION_SECONDS + LOGO_HOLD_SECONDS))
        frames.extend((target, target))
    times.append(times[-1] + TRANSITION_SECONDS)
    frames.append(source)
    loop_seconds = times[-1]
    loop_duration = time_num(loop_seconds)
    key_times = ";".join(time_num(v / loop_seconds) for v in times)
    opacity_values = ";".join(["0", "0"] + ["1"] * (len(frames) - 3) + ["0"])
    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">TeVerde29's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a dithered portrait and '
        "Full-Stack tool silhouettes.</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">vim profile.yml</text>',
        # Left visual frame.
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">340×385 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        # Loop layer stays visible at t=0 so camo/static first frames still show the face.
        # Intro duplicate below shimmers on top, then hands off at 3.2s.
        '<g opacity="1">',
    ]

    # Dense portrait drift moves toward the first logo in the current sequence.
    first_centroid = targets[0].mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 4, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (first_centroid - centroid) * 0.18 + noise[band]
        drift_positions = ["0 0", "0 0", f"{num(delta[0])} {num(delta[1])}",
                           f"{num(delta[0])} {num(delta[1])}"]
        drift_positions.extend(["0 0"] * (len(frames) - len(drift_positions)))
        drift_opacity = [".94", ".94", "0", "0"]
        drift_opacity.extend(["0"] * (len(frames) - len(drift_opacity) - 1))
        drift_opacity.append(".94")
        drift_position_values = ";".join(drift_positions)
        drift_opacity_values = ";".join(drift_opacity)
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{drift_position_values}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            f'values="{drift_opacity_values}"/></path>'
        )

    # Optimal-transport travellers, represented as tiny path squares (never glyphs).
    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )

    # Travellers give the transition its motion. A denser particle cloud takes
    # over when they arrive, preserving the dithered look during each logo hold.
    for index, (name, particles) in enumerate(logo_hold_particles.items()):
        visible = ["0"] * len(frames)
        visible[index * 2 + 2] = ".82"
        visible[index * 2 + 3] = ".82"
        parts.append(
            f'<path d="{particle_path(particles)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{";".join(visible)}"/></path>'
        )
    parts.append("</g>")

    # One-shot scattered intro: sixty random, interleaved point groups.
    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/>'
            "</path>"
        )
    parts.extend(
        [
            "</g>",
            # Small frame telemetry.
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · FS/SERPENTINE</text>',
            # Right information panel (Vim YAML editor view).
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing=".5">profile.yml</text>',
            f'<text x="580" y="111" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[YAML]</text>',
            f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1062" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700">@TeVerde29</text>',
        ]
    )

    row_y = 152.0
    for idx, (indent, key, value) in enumerate(YAML_ROWS, 1):
        line_num = f"{idx:2d}"
        if indent == 0:
            content = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            text_x = 525.0
        else:
            content = (
                f'<tspan fill="{t["portrait"]}">{html.escape(key)}: </tspan>'
                f'<tspan fill="{t["text"]}">{html.escape(value)}</tspan>'
            )
            text_x = 542.0 if indent == 1 else 560.0

        parts.extend(
            [
                f'<text x="506" y="{num(row_y)}" text-anchor="end" fill="{t["muted"]}" opacity=".45" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="15">'
                f"{line_num}</text>",
                f'<text x="{num(text_x)}" y="{num(row_y)}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="16">'
                f"{content}</text>",
            ]
        )
        row_y += 26.5

    # Vim status line at bottom of panel
    parts.extend(
        [
            f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
            f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}" rx="0 0 5 5"/>',
            f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["portrait"]}"/>',
            f'<text x="521" y="547" text-anchor="middle" fill="{t["bg"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11" font-weight="700">NORMAL</text>',
            f'<text x="569" y="547" fill="{t["text"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="600">profile.yml</text>',
            f'<text x="740" y="547" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[utf-8]</text>',
            f'<text x="1134" y="547" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">14L, 380B  100%  14:1</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source portrait: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    logos = make_logos()

    portraits: dict[str, np.ndarray] = {}
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        portraits[theme] = points

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in logos.items()
        }
        hold_particles = {
            name: sample_logo_points(image, rng, HOLD_PARTICLE_COUNT)
            for name, image in logos.items()
        }
        svg = render_svg(theme, portraits[theme], sampled, hold_particles, rng)
        output = ASSETS / f"banner-{theme}.v9.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portraits[theme]):,} portrait dots, "
            f"{TRAVELLER_COUNT} travellers"
        )

    print(f"animation sequence: {', '.join(logos)}")


if __name__ == "__main__":
    main()
