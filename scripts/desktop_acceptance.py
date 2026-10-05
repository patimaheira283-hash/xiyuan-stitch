"""Exercise import -> worker -> manual edit -> export cloud job and original-size PNG."""
import json
import os
from pathlib import Path
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication, QFileDialog
from xiyuan_mvp.gui import MainWindow
from xiyuan_mvp.image_io import read_image
from xiyuan_mvp.run_io import load_run, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'outputs/desktop-acceptance'
    output.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    window=MainWindow(output_root=output/'runs')
    window.set_input('a',str(ROOT/'data/demo_a.png'))
    window.set_input('b',str(ROOT/'data/demo_b.png'))
    window.show()
    window._start(False)
    deadline=time.monotonic()+45
    while window._is_busy() and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.result is not None,window.log.toPlainText()
    # Deterministic manual edit for the round-trip acceptance artifact.
    painted = window.mask_view.mask
    painted[:painted.shape[0]//4] = 0
    window.mask_view.set_image(window.initial_view._image, painted)
    window.controlnet.setCurrentText('canny')
    window.strength.setValue(.25)
    window.ai_opacity.setValue(.2)
    saved_dialog=QFileDialog.getSaveFileName
    try:
        QFileDialog.getSaveFileName=lambda *a,**k:(str(output/'desktop-job.zip'),'ZIP')
        window._export_job()
        assert (output/'desktop-job.zip').is_file(),window.log.toPlainText()
        window.tabs.setCurrentWidget(window.traditional_view)
        QFileDialog.getSaveFileName=lambda *a,**k:(str(output/'export.png'),'PNG')
        window._save()
        assert read_image(output/'export.png').shape==window.result.final_image.shape
    finally:QFileDialog.getSaveFileName=saved_dialog
    app.processEvents()
    window.grab().save(str(output/'desktop.png'))
    write_json(output/'acceptance.json',{'status':'passed','tabs':window.tabs.count(),
        'manual_edit':'top quarter of automatic seam removed','job':'desktop-job.zip',
        'export_dimensions':list(window.result.final_image.shape),'metrics':window.result.metrics})
    window.close()


if __name__=='__main__':main()
