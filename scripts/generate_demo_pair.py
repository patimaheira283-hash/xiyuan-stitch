from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xiyuan_mvp.image_io import write_image


def main() -> None:
    rng = np.random.default_rng(2026)
    base = rng.integers(0, 40, size=(520, 1200, 3), dtype=np.uint8)
    for x in range(30, 1170, 60):
        cv2.line(base, (x, 10), (x, 510), (100 + x % 120, 180, 220), 2)
    for y in range(30, 500, 55):
        cv2.line(base, (10, y), (1190, y), (220, 100 + y % 120, 160), 2)
    for index in range(24):
        center = (50 + index * 45, 80 + (index * 83) % 360)
        cv2.circle(base, center, 8 + index % 7, (230, 230, 230), -1)
    cv2.putText(
        base,
        "XIYUAN MVP",
        (410, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4,
    )
    first = base[:, :760].copy()
    second = cv2.convertScaleAbs(base[:, 440:].copy(), alpha=1.03, beta=4)
    output = Path("data")
    write_image(output / "demo_a.png", first)
    write_image(output / "demo_b.png", second)
    print(f"Demo pair written to {output.resolve()}")


if __name__ == "__main__":
    main()

