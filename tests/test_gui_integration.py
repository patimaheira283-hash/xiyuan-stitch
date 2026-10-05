from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from xiyuan_mvp.gui import MainWindow
from xiyuan_mvp.image_io import write_image

from tests.test_pipeline_smoke import _synthetic_pair


def test_gui_worker_completes_traditional_pipeline(tmp_path) -> None:
    image_a, image_b = _synthetic_pair()
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    write_image(path_a, image_a)
    write_image(path_b, image_b)

    app = QApplication.instance() or QApplication([])
    window = MainWindow(output_root=tmp_path / "runs")
    window.image_a_path = str(path_a)
    window.image_b_path = str(path_b)
    window._start(False)

    deadline = time.monotonic() + 10
    while window.result is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    assert window.result is not None
    assert window.progress_bar.value() == 100
    assert window.mask_view.mask is not None
    assert window.mask_view.mask.any()
    assert window.tabs.count() == 7
    if window.worker is not None:
        window.worker.wait(1000)
    window.close()
