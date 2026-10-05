import os
from pathlib import Path
import time
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QFileDialog
from xiyuan_mvp.gui import MainWindow
from xiyuan_mvp.image_io import read_image
from xiyuan_mvp.run_io import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'outputs/desktop-neural-acceptance';output.mkdir(exist_ok=True)
    source=ROOT/'data/neural-validation/cases/fine_grained_wood_local_11'
    app=QApplication.instance() or QApplication([])
    window=MainWindow(output_root=output/'runs')
    window.set_input('a',str(source/'a.png'));window.set_input('b',str(source/'b.png'))
    window.show()
    def wait():
        deadline=time.monotonic()+60
        while window._is_busy() and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        app.processEvents();assert not window._is_busy(),'worker timeout'
    window._start(False);wait();assert window.result is not None,window.log.toPlainText()
    baseline=window.result.traditional_image.copy()
    painted=window.mask_view.mask;painted[:painted.shape[0]//5]=0
    window.mask_view.set_image(window.initial_view._image,painted)
    window.repair_engine.setCurrentIndex(window.repair_engine.findData('neural_alignment'))
    window._start(True);wait()
    assert window.result.metrics['use_ai'],window.log.toPlainText()
    result=window.result
    np.testing.assert_array_equal(result.final_image[result.mask.soft_mask==0],baseline[result.mask.soft_mask==0])
    np.testing.assert_array_equal(result.mask.binary_mask,painted)
    window.tabs.setCurrentWidget(window.final_view)
    original=QFileDialog.getSaveFileName
    try:
        QFileDialog.getSaveFileName=lambda *a,**k:(str(output/'export.png'),'PNG')
        window._save()
    finally:QFileDialog.getSaveFileName=original
    assert read_image(output/'export.png').shape==result.final_image.shape
    app.processEvents();window.grab().save(str(output/'desktop.png'))
    write_json(output/'acceptance.json',{'status':'passed','engine':'neural_alignment','device':'cpu',
        'painted_mask_preserved':True,'outside_mask_unchanged':True,'export_shape':list(result.final_image.shape),
        'metrics':result.metrics,'run':str(window.last_output)})
    window.close()


if __name__=='__main__':main()
