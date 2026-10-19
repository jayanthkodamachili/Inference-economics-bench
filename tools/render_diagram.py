"""Render the committed architecture SVG.

Hand written rather than produced by a diagramming library, for the reasons in ADR-004: a
committed SVG with intrinsic dimensions and an explicit light background renders the same
on GitHub in either theme, which a Mermaid fence does not reliably do.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

W, H = 960, 500
INK = "#1b1f23"
MUTED = "#57606a"
BAD = "#b42318"
GOOD = "#116329"
LINE = "#8c959f"


def box(x, y, w, h, title, subtitle="", stroke=MUTED, fill="#ffffff", bold=False):
    weight = "600" if bold else "400"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + w / 2}" y="{y + (20 if subtitle else h / 2 + 4)}" '
        f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" '
        f'font-weight="{weight}" fill="{INK}">{title}</text>',
    ]
    if subtitle:
        for index, line in enumerate(subtitle.split("|")):
            parts.append(
                f'<text x="{x + w / 2}" y="{y + 38 + index * 14}" text-anchor="middle" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="11" '
                f'fill="{MUTED}">{line}</text>'
            )
    return "".join(parts)


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
        f'stroke-width="1.5" marker-end="url(#a)"{dash}/>'
    )


def label(x, y, text, fill=MUTED, size=11, anchor="middle", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def render() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Architecture of inference-economics-bench">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>'
        "</marker></defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        label(W / 2, 26, "One run produces both numbers, because one of them is useless alone",
              INK, 14, weight="600"),
    ]

    parts.append(box(24, 150, 180, 76, "one corpus",
                     "unigram signal plus|ordered pairs a bag of|words cannot see",
                     bold=True))

    models = ["majority", "logreg_unigram", "logreg_ngram2", "transformer", "svd_boosting"]
    for index, name in enumerate(models):
        y = 60 + index * 54
        parts.append(box(276, y, 196, 40, name, "", stroke=MUTED))
        parts.append(arrow(204, 188, 276, y + 20))

    parts.append(label(374, 352, "all five take a string, so all five pay to parse it",
                       MUTED, 10))

    parts.append(box(540, 96, 180, 54, "accuracy",
                     "deterministic", stroke=MUTED, bold=True))
    parts.append(box(540, 186, 180, 54, "cost per call",
                     "latency x machine price", stroke=MUTED, bold=True))
    for index in range(len(models)):
        parts.append(arrow(472, 80 + index * 54, 540, 123))
        parts.append(arrow(472, 80 + index * 54, 540, 213))

    parts.append(box(778, 140, 158, 70, "the frontier",
                     "who is dominated,|and at what budget",
                     stroke=GOOD, fill="#f2fbf4", bold=True))
    parts.append(arrow(720, 123, 778, 165))
    parts.append(arrow(720, 213, 778, 185))

    parts.append(label(857, 232, "exit 4: dominated", BAD, 10))

    parts.append(
        box(130, 400, 700, 66,
            "ranked by accuracy and ranked by cost are not the same ranking",
            "the most accurate model here costs 33 times what the next one does, "
            "for two accuracy points", bold=True)
    )
    parts.append(arrow(857, 210, 700, 400))
    parts.append(arrow(630, 240, 420, 400))

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
