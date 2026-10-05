"""Exercise the real desktop controls and models; usable from a frozen build."""
from pathlib import Path
import socket
import sys
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QScrollArea
from .image_io import read_image
from . import __version__
from .run_io import load_run, write_json


def run_workflow(app, window, data, output, *, offline=False):
    data, output = Path(data), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    original_connect = socket.socket.connect
    original_open, original_save = QFileDialog.getOpenFileName, QFileDialog.getSaveFileName
    def no_network(*args, **kwargs):
        raise RuntimeError('Offline acceptance attempted a network connection')
    if offline: socket.socket.connect = no_network
    rows = []
    try:
        window.output_root = output.parent/'runs'
        window.registration_method.setCurrentText('loftr')
        window.repair_engine.setCurrentIndex(window.repair_engine.findData('neural_alignment'))
        window.show(); app.processEvents()
        def wait():
            deadline = time.monotonic()+180
            while window._is_busy() and time.monotonic()<deadline:
                app.processEvents(); time.sleep(.01)
            app.processEvents()
            assert not window._is_busy(), 'Worker timed out'
            assert window.progress_bar.value()==100, window.log.toPlainText()
        for cid in ('fine_grained_wood_local_11', 'brick_wall_005_local_11', 'dark_wooden_planks_gradient'):
            destination = output.parent/cid; destination.mkdir(exist_ok=True)
            for index, name in enumerate(('a.png','b.png')):
                path = str(data/cid/name)
                QFileDialog.getOpenFileName = lambda *a, _p=path, **k: (_p, 'PNG')
                window.file_actions[index].trigger()
            QTest.mouseClick(window.run_button, Qt.LeftButton); wait()
            assert window.result is not None
            assert window.result.metrics['feature_method'].lower()=='loftr'
            baseline = window.result.repair_base_image.copy()
            assert window.result.metrics.get('clear_fusion_enabled'), 'Clear fusion was not exercised'
            clear_export = destination/'clear-export.png'
            window.tabs.setCurrentWidget(window.initial_view)
            QFileDialog.getSaveFileName = lambda *a, **k: (str(clear_export), 'PNG')
            QTest.mouseClick(window.quick_export_button, Qt.LeftButton)
            np.testing.assert_array_equal(read_image(clear_export),baseline)
            for scroll in window.findChildren(QScrollArea):scroll.verticalScrollBar().setValue(0)
            app.processEvents()
            window.grab().save(str(destination/'clear-desktop.png'))
            mask = window.mask_view.mask
            ys, xs = np.where(mask>0); mid = len(xs)//2
            window.tabs.setCurrentWidget(window.mask_view); app.processEvents()
            window.mask_view.fit_image()
            point = window.mask_view.mapFromScene(QPointF(float(xs[mid]), float(ys[mid])))
            window.brush_size.setValue(44)
            QTest.mouseClick(window.erase_button, Qt.LeftButton)
            QTest.mousePress(window.mask_view.viewport(), Qt.LeftButton, pos=point)
            QTest.mouseMove(window.mask_view.viewport(), point+window.mask_view.mapFromScene(QPointF(20,20))-window.mask_view.mapFromScene(QPointF(0,0)))
            QTest.mouseRelease(window.mask_view.viewport(), Qt.LeftButton, pos=point)
            assert not np.array_equal(mask, window.mask_view.mask), 'Actual eraser stroke changed no pixels'
            QTest.mouseClick(window.undo_button, Qt.LeftButton)
            np.testing.assert_array_equal(mask, window.mask_view.mask)
            QTest.mouseClick(window.mask_view.viewport(), Qt.LeftButton, pos=point)
            painted = window.mask_view.mask
            assert not np.array_equal(mask, painted)
            window.grab().save(str(destination/'painted.png'))
            QTest.mouseClick(window.ai_button, Qt.LeftButton); wait()
            result = window.result
            assert result.metrics['use_ai'] and result.metrics['repair_engine']=='raft_large'
            assert not result.metrics.get('test_double')
            np.testing.assert_array_equal(result.mask.binary_mask, cv2.bitwise_and(painted,result.repair_region))
            np.testing.assert_array_equal(result.final_image[result.mask.soft_mask==0],baseline[result.mask.soft_mask==0])
            export = destination/'export.png'
            QFileDialog.getSaveFileName = lambda *a, **k: (str(export), 'PNG')
            QTest.mouseClick(window.save_button, Qt.LeftButton)
            np.testing.assert_array_equal(read_image(export), result.final_image)
            # Both comparison tabs must expose the actual computed baselines.
            np.testing.assert_array_equal(window.poisson_view._image,result.poisson_image)
            np.testing.assert_array_equal(window.traditional_view._image,result.traditional_image)
            saved = window.last_output/'run.json'
            restored, _ = load_run(saved)
            np.testing.assert_array_equal(restored.final_image,result.final_image)
            app.processEvents(); window.grab().save(str(destination/'desktop.png'))
            rows.append({'id':cid, 'status':'passed', 'mask_edit_and_undo':True,
                'outside_mask_unchanged':True, 'export_pixels_identical':True,
                'export_shape':list(result.final_image.shape),'metrics':result.metrics,'clear_export':str(clear_export.resolve()),
                'run':str(saved.resolve()), 'export':str(export.resolve())})
            write_json(output, {'status':'running', 'records':rows})
        motion=data/'motion-run/run.json'
        if motion.exists():
            destination=output.parent/'protected-motion';destination.mkdir(exist_ok=True)
            QFileDialog.getOpenFileName=lambda *a,**k:(str(motion),'JSON')
            window._open_run();app.processEvents()
            assert window.protect_detail.isChecked()
            baseline=window.result.repair_base_image.copy()
            QTest.mouseClick(window.ai_button,Qt.LeftButton);wait()
            result=window.result
            assert result.metrics['model_executed'] and result.metrics['repair_applied']
            assert result.metrics['repair_revision']=='clarity-guided-raft-v1'
            assert not np.array_equal(result.final_image,baseline)
            np.testing.assert_array_equal(result.final_image[result.mask.soft_mask==0],baseline[result.mask.soft_mask==0])
            export=destination/'export.png'
            QFileDialog.getSaveFileName=lambda *a,**k:(str(export),'PNG')
            QTest.mouseClick(window.quick_export_button,Qt.LeftButton)
            np.testing.assert_array_equal(read_image(export),result.final_image)
            for scroll in window.findChildren(QScrollArea):scroll.verticalScrollBar().setValue(0)
            app.processEvents();window.grab().save(str(destination/'desktop.png'))
            record={'id':'protected-motion','status':'passed','prepared_geometry':'known crop coordinates; not an automatic registration test',
                'metrics':result.metrics,'export_pixels_identical':True,'outside_mask_unchanged':True,'export':str(export.resolve())}
            window.protect_detail.setChecked(False)
            QTest.mouseClick(window.ai_button,Qt.LeftButton);wait()
            assert window.result.metrics['repair_engine']=='raft_large' and 'repair_revision' not in window.result.metrics
            record['legacy_option_exercised']=True
            rows.append(record)
        write_json(output, {'status':'passed', 'version':__version__, 'frozen':bool(getattr(sys,'frozen',False)),
            'network_blocked':offline, 'gui_tabs':window.tabs.count(), 'records':rows})
    finally:
        socket.socket.connect = original_connect
        QFileDialog.getOpenFileName, QFileDialog.getSaveFileName = original_open, original_save
        window.close()
