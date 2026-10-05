from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import importlib.util
import os
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import yaml

from . import __version__
from .config import load_config, validate_config
from .image_io import read_image, write_image
from .pipeline import StitchPipeline
from .run_io import load_run, save_failure, save_run, write_json
from .seam_mask import generate_seam_mask

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPixmap, QKeySequence, QFont, QFontDatabase
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
        QScrollArea, QSpinBox, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget,
        QListWidget, QListWidgetItem, QAbstractItemView, QFrame,
    )
except ImportError as exc:
    raise SystemExit('缺少 PySide6，请先安装 .[gui] 可选依赖。') from exc


def _to_qimage(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


class ImageView(QGraphicsView):
    mask_changed = Signal()

    def __init__(self, editable=False):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#111a24"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setAlignment(Qt.AlignCenter)
        self.setViewportMargins(10, 10, 10, 10)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(240, 200)
        self._image = None
        self._mask = None
        self._editable = editable
        self._erase = False
        self._brush_size = 48
        self._drawing = False
        self._last_point = None
        self._pan = None
        self._history = []
        self._image_item = None
        self._mask_item = None
        self._auto_fit = True

    @property
    def mask(self):
        return None if self._mask is None else self._mask.copy()

    def set_image(self, image, mask=None):
        self.scene().clear()
        self._image = None if image is None else image.copy()
        self._mask = None if mask is None else mask.copy()
        self._history = []
        self._image_item = None
        self._mask_item = None
        if image is not None:
            self._image_item = self.scene().addPixmap(QPixmap.fromImage(_to_qimage(image)))
            self.scene().setSceneRect(0, 0, image.shape[1], image.shape[0])
            if self._editable and self._mask is None:
                self._mask = np.zeros(image.shape[:2], np.uint8)
            self._mask_item = self.scene().addPixmap(QPixmap())
            self._mask_item.setZValue(1)
            self._update_overlay()
        self.fit_image()

    def fit_image(self):
        if self._image is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._auto_fit = True

    def _update_overlay(self):
        if self._mask_item is not None and self._mask is not None:
            rgba = np.zeros((*self._mask.shape, 4), np.uint8)
            rgba[..., 0] = 245
            rgba[..., 1] = 75
            rgba[..., 3] = (self._mask.astype(float) * 0.48).astype(np.uint8)
            h, w = self._mask.shape
            qim = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888).copy()
            self._mask_item.setPixmap(QPixmap.fromImage(qim))

    def set_brush(self, size, erase):
        self._brush_size = size
        self._erase = erase

    def _remember(self):
        if self._mask is not None:
            self._history.append(self._mask.copy())
            self._history = self._history[-15:]

    def clear_mask(self):
        if self._mask is not None:
            self._remember()
            self._mask.fill(0)
            self._update_overlay()
            self.mask_changed.emit()

    def undo_mask(self):
        if self._history:
            self._mask = self._history.pop()
            self._update_overlay()
            self.mask_changed.emit()

    def _paint_at(self, point):
        if not self._editable or self._mask is None:
            return
        p = self.mapToScene(point)
        current = (int(p.x()), int(p.y()))
        if not (0 <= current[0] < self._mask.shape[1] and 0 <= current[1] < self._mask.shape[0]):
            self._last_point = None
            return
        color = 0 if self._erase else 255
        if self._last_point is not None:
            cv2.line(self._mask, self._last_point, current, color, self._brush_size)
        cv2.circle(self._mask, current, max(1, self._brush_size//2), color, -1)
        self._last_point = current
        self._update_overlay()
        self.mask_changed.emit()

    def wheelEvent(self, event):
        if self._image is not None:
            factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            if 0.02 < self.transform().m11() * factor < 30:
                self.scale(factor, factor)
                self._auto_fit = False
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self._editable:
            self._remember()
            self._drawing = True
            self._last_point = None
            self._paint_at(event.position().toPoint())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        if self._pan is not None:
            delta = point-self._pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self._pan = point
            self._auto_fit = False
        elif self._drawing:
            self._paint_at(point)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self._pan = None
        self._last_point = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_image()


class PipelineWorker(QThread):
    progress = Signal(str, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, pipeline, image_a, image_b, use_ai, custom_mask, prepared=None, output=None, sequence=None):
        super().__init__()
        self.pipeline, self.image_a, self.image_b = pipeline, image_a, image_b
        self.use_ai, self.custom_mask, self.prepared = use_ai, custom_mask, prepared
        self.output = output
        self.sequence = sequence
        self.config_snapshot = deepcopy(pipeline.config)

    def run(self):
        try:
            if self.sequence and len(self.sequence)>2 and not self.use_ai:
                result = self.pipeline.run_many(self.sequence,cancelled=self.isInterruptionRequested,
                                               progress=lambda message,value:self.progress.emit(message,value))
            else:
                result = self.pipeline.run(
                    self.image_a, self.image_b, use_ai=self.use_ai, custom_mask=self.custom_mask,
                    prepared=self.prepared, cancelled=self.isInterruptionRequested,
                    progress=lambda message, value: self.progress.emit(message, value),
                )
            if self.output:
                save_run(self.output, result, self.config_snapshot)
            self.succeeded.emit(result)
        except Exception as exc:
            if self.output:
                try:
                    save_failure(self.output, exc, self.config_snapshot)
                except OSError:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, output_root=None):
        app = QApplication.instance()
        if app is not None and not app.property("xiyuan_fonts_configured"):
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
            app.setFont(QFont("Microsoft YaHei", 9))
            app.setProperty("xiyuan_fonts_configured", True)
        super().__init__()
        self.setWindowTitle(f"曦源 {__version__} · 图像无缝拼接")
        self.resize(1380, 900)
        self.config = load_config()
        self.config['clear_fusion']['enabled']=True
        if importlib.util.find_spec('torchvision') is not None:
            self.config['inpainting']['engine']='neural_alignment'
        if getattr(sys,'frozen',False) and (Path(sys._MEIPASS)/'model_cache').is_dir():
            self.config['inpainting']['engine']='neural_alignment'
            if (Path(sys._MEIPASS)/'model_cache/hub/checkpoints/loftr_outdoor.ckpt').is_file():
                self.config['registration'].update(method='loftr',loftr_device='cpu')
        self.pipeline = StitchPipeline(self.config)
        self.image_a_path = None
        self.image_b_path = None
        self.result = None
        self.worker = None
        self.output_root = Path(output_root or "outputs/gui")
        self.last_output = None
        self._closing = False
        self._build_ui()
        self._load_controls()

    def _build_ui(self):
        toolbar = QToolBar("文件")
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        self.file_actions = []
        for label, handler in (
            ("导入图片 A", lambda: self._choose_image("a")),
            ("导入图片 B", lambda: self._choose_image("b")),
            ("导入多图序列", self._choose_sequence),
            ("打开运行记录", self._open_run), ("导入配置", self._import_config),
            ("导出配置", self._export_config),
        ):
            action = QAction(label, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)
            self.file_actions.append(action)
        undo = QAction("撤销画笔", self)
        undo.setShortcut(QKeySequence.Undo)
        undo.triggered.connect(lambda: self.mask_view.undo_mask() if not self._is_busy() else None)
        self.addAction(undo)
        root = QWidget()
        root.setObjectName("appRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("heroHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(14)
        header_copy = QVBoxLayout()
        header_copy.setSpacing(2)
        eyebrow = QLabel("XIYUAN  ·  IMAGE STITCHING WORKBENCH")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("曦源图像拼接")
        title.setObjectName("heroTitle")
        subtitle = QLabel("从重叠图片开始，完成配准、接缝编辑、结构修复与原尺寸导出")
        subtitle.setObjectName("heroSubtitle")
        header_copy.addWidget(eyebrow)
        header_copy.addWidget(title)
        header_copy.addWidget(subtitle)
        header_layout.addLayout(header_copy, 1)
        version_badge = QLabel(f"v{__version__}")
        version_badge.setObjectName("versionBadge")
        version_badge.setAlignment(Qt.AlignCenter)
        version_badge.setToolTip("当前软件版本")
        header_layout.addWidget(version_badge, 0, Qt.AlignTop)
        layout.addWidget(header)

        self.workflow_hint = QLabel("① 导入图片  ② 自动配准  ③ 编辑接缝  ④ 运行修复  ⑤ 导出结果")
        self.workflow_hint.setObjectName("workflowHint")
        layout.addWidget(self.workflow_hint)

        self.path_label = QLabel("请导入两张具有重叠区域的图片")
        self.path_label.setObjectName("pathSummary")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)
        split = QSplitter()
        panel = QWidget()
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(360)
        controls = QVBoxLayout(panel)
        panel_heading = QLabel("工作台控制")
        panel_heading.setObjectName("panelHeading")
        controls.addWidget(panel_heading)
        self.sequence_group = QGroupBox("多图顺序（拖动调整，相邻图片需重叠）")
        sequence_layout=QVBoxLayout(self.sequence_group)
        self.sequence_list=QListWidget()
        self.sequence_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.sequence_list.setMaximumHeight(105)
        self.sequence_list.model().rowsMoved.connect(self._sequence_reordered)
        sequence_layout.addWidget(self.sequence_list)
        self.sequence_group.hide()
        controls.addWidget(self.sequence_group)
        fusion_heading = QLabel("01  ·  输入与融合")
        fusion_heading.setObjectName("sectionHeading")
        controls.addWidget(fusion_heading)
        self.clear_fusion = QCheckBox('清晰融合：优先保留原图细节')
        self.clear_fusion.setObjectName("featureCheck")
        self.clear_fusion.setToolTip('本机图像算法，无需 GPU。比较两图清晰度，仅在差异明确时选择更清楚的纹理；影响整个重叠区域。更改后重新配准。')
        controls.addWidget(self.clear_fusion)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.run_button = QPushButton("开始配准")
        self.run_button.setObjectName("primaryAction")
        self.run_button.setToolTip("读取两张图片，自动配准并生成融合底图")
        self.ai_button = QPushButton("运行修复")
        self.ai_button.setObjectName("accentAction")
        self.ai_button.setToolTip("使用当前接缝区域和修复设置生成结果")
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.ai_button)
        controls.addLayout(action_row)
        self.cancel_button = QPushButton("取消当前任务")
        self.cancel_button.setObjectName("cancelAction")
        self.cancel_button.setToolTip("停止当前计算，已完成的输入不会被删除")
        self.run_button.clicked.connect(lambda: self._start(False))
        self.ai_button.clicked.connect(lambda: self._start(True))
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.cancel_button)
        self.quick_export_button=QPushButton('导出当前结果')
        self.quick_export_button.setObjectName("secondaryAction")
        self.quick_export_button.clicked.connect(self._save)
        controls.addWidget(self.quick_export_button)
        settings_heading = QLabel("02  ·  修复方式与参数")
        settings_heading.setObjectName("sectionHeading")
        controls.addWidget(settings_heading)
        self.settings_group = QGroupBox("修复参数")
        form = QFormLayout(self.settings_group)
        self.mode = QComboBox()
        self.repair_engine = QComboBox()
        self.repair_engine.addItem('生成修复（GPU）','diffusion')
        self.repair_engine.addItem('结构修复（本机，两图）','neural_alignment')
        self.repair_engine.addItem('结构＋生成（GPU，两图）','hybrid')
        self.repair_engine.setToolTip('结构修复通过神经网络对齐原有纹理；生成修复使用扩散模型补绘内容。')
        self.hybrid_policy = QComboBox()
        self.hybrid_policy.addItem('始终生成（对照）', 'always')
        self.hybrid_policy.addItem('仅结构有收益时生成（推荐）', 'on_structural_gain')
        self.hybrid_policy.setToolTip('门控策略会跳过没有确认结构收益、或残差已经很低的扩散生成，减少不必要的纹理改动。')
        self.hybrid_min_residual = QDoubleSpinBox()
        self.hybrid_min_residual.setRange(0, 128)
        self.hybrid_min_residual.setDecimals(2)
        self.hybrid_min_residual.setSingleStep(.5)
        self.protect_detail=QCheckBox('保护清晰细节')
        self.protect_detail.setToolTip('根据两图清晰度共同校正错位；已有可靠清晰纹理或无法确认改善时保留底图。关闭后可对照旧版 RAFT 修复。')
        self.mode.addItems(["质量（20 步）", "预览（10 步）"])
        self.mode.currentIndexChanged.connect(lambda i: self.steps.setValue(10 if i else 20))
        self.steps = QSpinBox()
        self.steps.setRange(2, 150)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.1, 1.0)
        self.strength.setSingleStep(0.05)
        self.seam_width = QSpinBox()
        self.seam_width.setRange(3, 512)
        self.patch_size = QComboBox()
        self.patch_size.addItems(["512", "768", "256"])
        self.patch_layout = QComboBox()
        self.patch_layout.addItem("单个 512 Patch（当前）", "single")
        self.patch_layout.addItem("原尺寸重叠分块（实验）", "native_tiled")
        self.prompt = QPlainTextEdit()
        self.prompt.setMaximumHeight(76)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("本地模型目录或模型标识")
        self.allow_cpu = QCheckBox("允许较慢的 CPU 推理")
        self.registration_method = QComboBox()
        self.registration_method.addItems(["sift", "loftr", "auto"])
        self.controlnet = QComboBox()
        self.controlnet.addItems(["none", "canny", "tile"])
        self.controlnet.currentTextChanged.connect(self._update_controlnet_mask_control)
        self.controlnet_mask_mode = QComboBox()
        self.controlnet_mask_mode.addItem("原始边缘", "none")
        self.controlnet_mask_mode.addItem("抑制接缝内边缘", "context_edges")
        self.controlnet_mask_mode.addItem("渐弱接缝边缘", "faded_edges")
        self.use_lcm = QCheckBox("LCM 快速预览（8 步）")
        self.refinement = QCheckBox("实验性颜色 / 局部几何校正（需重新配准）")
        self.clear_fusion.toggled.connect(lambda checked:self.refinement.setChecked(False) if checked else None)
        self.refinement.toggled.connect(lambda checked:self.clear_fusion.setChecked(False) if checked else None)
        self.ai_opacity = QDoubleSpinBox()
        self.ai_opacity.setRange(0, 1)
        self.ai_opacity.setSingleStep(0.1)
        self.ai_boundary_guard = QCheckBox("接缝边界保护（实验）")
        self.ai_boundary_guard.setToolTip("检测生成结果是否让接缝边界变硬；发现退化时自动降低 AI 混合比例。")
        self.use_lcm.toggled.connect(lambda value: self.steps.setValue(8 if value else 20))
        self.allow_cpu.setToolTip("无 NVIDIA GPU 时建议导出 GPU 任务，使用免费笔记本运行。")
        for label, widget in (("配准算法",self.registration_method),("结构约束",self.controlnet),
                              ("Canny 接缝保护",self.controlnet_mask_mode),
                              ("模式", self.mode), ("步数", self.steps), ("随机种子", self.seed),
                              ("重绘强度", self.strength), ("接缝宽度", self.seam_width),
                              ("Patch 尺寸", self.patch_size), ("生成取样", self.patch_layout), ("提示词", self.prompt),
                              ("模型位置", self.model_path)):
            form.addRow(label, widget)
        form.addRow(self.allow_cpu)
        form.insertRow(0,'修复方式',self.repair_engine)
        form.insertRow(1,'结构修复',self.protect_detail)
        form.insertRow(2,'结构＋生成策略',self.hybrid_policy)
        form.insertRow(3,'生成最低残差',self.hybrid_min_residual)
        form.addRow(self.use_lcm)
        form.addRow(self.refinement)
        form.addRow("AI 混合比例", self.ai_opacity)
        form.addRow(self.ai_boundary_guard)
        self.repair_engine.currentIndexChanged.connect(self._engine_changed)
        self.hybrid_policy.currentIndexChanged.connect(self._hybrid_policy_changed)
        controls.addWidget(self.settings_group)
        mask_heading = QLabel("03  ·  接缝编辑")
        mask_heading.setObjectName("sectionHeading")
        controls.addWidget(mask_heading)
        self.mask_group = QGroupBox("接缝画笔")
        mask_layout = QVBoxLayout(self.mask_group)
        mask_layout.setSpacing(8)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(2, 300)
        self.brush_size.setValue(48)
        self.brush_size.setPrefix("画笔像素  ")
        self.brush_size.valueChanged.connect(lambda v: self.mask_view.set_brush(v, self.mask_view._erase))
        mask_layout.addWidget(self.brush_size)
        row = QHBoxLayout()
        self.paint_button, self.erase_button = QPushButton("画笔"), QPushButton("橡皮")
        for button in (self.paint_button, self.erase_button):
            button.setCheckable(True)
            button.setAutoExclusive(True)
        self.paint_button.setChecked(True)
        self.paint_button.clicked.connect(lambda: self._set_brush_mode(False))
        self.erase_button.clicked.connect(lambda: self._set_brush_mode(True))
        row.addWidget(self.paint_button)
        row.addWidget(self.erase_button)
        mask_layout.addLayout(row)
        row = QHBoxLayout()
        self.clear_button = QPushButton("清空")
        self.undo_button = QPushButton("撤销")
        self.reset_mask_button = QPushButton("自动区域")
        self.clear_button.clicked.connect(lambda: self.mask_view.clear_mask())
        self.undo_button.clicked.connect(lambda: self.mask_view.undo_mask())
        self.reset_mask_button.clicked.connect(self._reset_mask)
        for widget in (self.clear_button, self.undo_button, self.reset_mask_button):
            row.addWidget(widget)
        mask_layout.addLayout(row)
        controls.addWidget(self.mask_group)
        export_heading = QLabel("04  ·  导出与分享")
        export_heading.setObjectName("sectionHeading")
        controls.addWidget(export_heading)
        self.save_button = QPushButton("导出当前查看的图片")
        self.save_button.setObjectName("secondaryAction")
        self.save_button.clicked.connect(self._save)
        self.job_button = QPushButton("导出免费 GPU 任务")
        self.job_button.setObjectName("secondaryAction")
        self.job_button.clicked.connect(self._export_job)
        controls.addWidget(self.save_button)
        controls.addWidget(self.job_button)
        hint = QLabel("操作提示\n先配准，再在接缝页调整红色区域。\n滚轮缩放，中键 / 右键拖动画布。\n双击画布适应窗口，Ctrl+Z 撤销画笔。")
        hint.setObjectName("helpCard")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(320)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("resultTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.source_a_view, self.source_b_view = ImageView(), ImageView()
        self.traditional_view, self.poisson_view = ImageView(), ImageView()
        self.initial_view = ImageView()
        self.mask_view, self.final_view = ImageView(True), ImageView()
        tab_help = {
            "原图 A": "输入图片 A。双击画布适应窗口，滚轮缩放。",
            "原图 B": "输入图片 B。双击画布适应窗口，滚轮缩放。",
            "羽化融合": "快速、平滑的传统融合基线。",
            "泊松融合": "颜色连续性优先的传统融合基线。",
            "清晰融合 / 修复底图": "当前送入结构修复或生成修复的底图。",
            "接缝编辑": "用画笔添加、橡皮擦除需要处理的接缝区域。",
            "AI 结果": "当前 AI 修复结果；导出时使用原始画布尺寸。",
        }
        for view, label in ((self.source_a_view, "原图 A"), (self.source_b_view, "原图 B"),
                            (self.traditional_view, "羽化融合"), (self.poisson_view, "泊松融合"),
                            (self.initial_view, "清晰融合 / 修复底图"), (self.mask_view, "接缝编辑"), (self.final_view, "AI 结果")):
            self.tabs.addTab(view, label)
            self.tabs.setTabToolTip(self.tabs.count() - 1, tab_help[label])
        split.addWidget(scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 1020])
        layout.addWidget(split, 1)
        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 8, 12, 8)
        status_layout.setSpacing(6)
        self.status_label = QLabel("就绪 · 原始分辨率导出")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("taskProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        layout.addWidget(status_panel)
        self.log = QPlainTextEdit()
        self.log.setObjectName("runLog")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(84)
        self.log.setPlaceholderText("运行记录与错误信息")
        layout.addWidget(self.log)
        self.setCentralWidget(root)
        self.setStyleSheet(self._style_sheet())
        self._set_busy(False)

    @staticmethod
    def _style_sheet():
        return """
        QWidget#appRoot { background: #f4f7fa; color: #1d2a35; }
        QMainWindow { background: #f4f7fa; }
        QToolBar#mainToolbar {
            background: #ffffff; border: 0; border-bottom: 1px solid #dce5eb;
            padding: 5px 10px; spacing: 3px;
        }
        QToolBar#mainToolbar QToolButton {
            color: #425466; background: transparent; border: 0; border-radius: 6px;
            padding: 7px 9px;
        }
        QToolBar#mainToolbar QToolButton:hover { background: #e9f3f5; color: #0f6875; }
        QFrame#heroHeader {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0f6674, stop:1 #214f78);
            border-radius: 12px;
        }
        QLabel#eyebrow { color: #bce7e5; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
        QLabel#heroTitle { color: white; font-size: 26px; font-weight: 700; }
        QLabel#heroSubtitle { color: #d6eef0; font-size: 12px; }
        QLabel#versionBadge {
            background: rgba(255,255,255,0.16); color: white; border: 1px solid rgba(255,255,255,0.28);
            border-radius: 10px; padding: 5px 10px; font-weight: 700;
        }
        QLabel#workflowHint {
            background: #e8f5f4; color: #146873; border: 1px solid #cce8e6;
            border-radius: 7px; padding: 8px 12px; font-weight: 600;
        }
        QLabel#pathSummary {
            background: #ffffff; color: #536575; border: 1px solid #dce5eb;
            border-radius: 7px; padding: 8px 12px;
        }
        QLabel#panelHeading { color: #123f50; font-size: 18px; font-weight: 700; padding: 2px 2px 6px; }
        QLabel#sectionHeading { color: #728391; font-size: 10px; font-weight: 800; padding: 8px 2px 1px; }
        QScrollArea { border: 0; background: transparent; }
        QGroupBox {
            background: #ffffff; border: 1px solid #dce5eb; border-radius: 9px;
            margin-top: 10px; padding: 14px 10px 10px; font-weight: 700; color: #304553;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #2e5c68; background: #f4f7fa; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
            background: #fbfdfe; color: #243746; border: 1px solid #cbd8df; border-radius: 5px; padding: 5px 7px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
            border: 1px solid #3b9aa3; background: #ffffff;
        }
        QComboBox::drop-down { border: 0; width: 22px; }
        QCheckBox { spacing: 7px; padding: 3px 1px; color: #324b59; }
        QCheckBox#featureCheck { background: #edf8f6; border: 1px solid #c9e8e2; border-radius: 6px; padding: 7px; font-weight: 600; }
        QCheckBox::indicator { width: 15px; height: 15px; }
        QPushButton {
            background: #ffffff; color: #31505e; border: 1px solid #c8d6de; border-radius: 6px;
            padding: 7px 10px; min-height: 18px; font-weight: 600;
        }
        QPushButton:hover { background: #edf7f7; border-color: #56aab0; color: #155e68; }
        QPushButton:pressed { background: #dff0ef; }
        QPushButton:disabled { background: #eef1f3; color: #9aa8b1; border-color: #e0e6ea; }
        QPushButton#primaryAction { background: #0f7581; color: white; border-color: #0f7581; }
        QPushButton#primaryAction:hover { background: #0a6470; }
        QPushButton#accentAction { background: #d8744a; color: white; border-color: #d8744a; }
        QPushButton#accentAction:hover { background: #bd5f38; }
        QPushButton#cancelAction { color: #a85443; }
        QPushButton#secondaryAction { background: #f3f8f9; color: #2c6770; border-color: #bdd9dc; }
        QPushButton:checked { background: #d8eff0; border: 1px solid #26929b; color: #0b6973; }
        QListWidget { background: #fbfdfe; border: 1px solid #cbd8df; border-radius: 5px; padding: 3px; }
        QListWidget::item { padding: 5px; border-radius: 4px; }
        QListWidget::item:selected { background: #dceff0; color: #155d66; }
        QTabWidget#resultTabs::pane { background: #111a24; border: 1px solid #263743; border-radius: 9px; }
        QTabWidget#resultTabs QTabBar::tab {
            background: #e9eef2; color: #5e707d; border: 0; border-top-left-radius: 6px; border-top-right-radius: 6px;
            padding: 8px 11px; margin-right: 2px;
        }
        QTabWidget#resultTabs QTabBar::tab:selected { background: #111a24; color: #e8f6f6; font-weight: 700; }
        QTabWidget#resultTabs QTabBar::tab:hover:!selected { background: #d9e8eb; color: #275d67; }
        QFrame#statusPanel { background: #ffffff; border: 1px solid #dce5eb; border-radius: 8px; }
        QLabel#statusLabel { color: #3b5260; font-weight: 600; }
        QProgressBar#taskProgress { background: #edf1f3; border: 0; border-radius: 4px; height: 7px; }
        QProgressBar#taskProgress::chunk { background: #1b8b91; border-radius: 4px; }
        QPlainTextEdit#runLog { background: #1c2832; color: #c1d0d6; border: 0; border-radius: 7px; padding: 7px; font-family: Consolas; font-size: 11px; }
        QLabel#helpCard { background: #edf2f5; color: #657681; border: 1px solid #d9e3e8; border-radius: 7px; padding: 9px; line-height: 1.45; }
        QSplitter::handle { background: #dce5eb; width: 6px; }
        QScrollBar:vertical { background: #edf1f3; width: 10px; margin: 2px; border-radius: 5px; }
        QScrollBar::handle:vertical { background: #b7c8d0; border-radius: 5px; min-height: 30px; }
        """

    def _load_controls(self):
        ai = self.config["inpainting"]
        self.clear_fusion.setChecked(self.config.get('clear_fusion',{}).get('enabled',False))
        self.protect_detail.setChecked(self.config.get('neural_alignment',{}).get('preserve_detail',True))
        self.repair_engine.setCurrentIndex(max(0,self.repair_engine.findData(ai.get('engine','diffusion'))))
        self.hybrid_policy.setCurrentIndex(max(0, self.hybrid_policy.findData(ai.get('hybrid_diffusion_policy', 'always'))))
        self.hybrid_min_residual.setValue(float(ai.get('hybrid_min_residual_error', 2.0)))
        self.steps.setValue(ai["steps"])
        self.seed.setValue(ai["seed"])
        self.strength.setValue(ai["strength"])
        self.seam_width.setValue(self.config["mask"]["seam_width"])
        self.patch_size.setCurrentText(str(ai["patch_size"]))
        self.patch_layout.setCurrentIndex(max(0, self.patch_layout.findData(ai.get("patch_layout", "single"))))
        self.prompt.setPlainText(ai["prompt"])
        self.model_path.setText(ai["model_id"])
        self.allow_cpu.setChecked(bool(ai["allow_cpu"]))
        self.registration_method.setCurrentText(self.config["registration"].get("method","sift"))
        self.controlnet.setCurrentText(ai.get("controlnet","none"))
        self.controlnet_mask_mode.setCurrentIndex(max(0, self.controlnet_mask_mode.findData(ai.get("controlnet_mask_mode", "none"))))
        self.use_lcm.blockSignals(True)
        self.use_lcm.setChecked(bool(ai.get("use_lcm",False)))
        self.use_lcm.blockSignals(False)
        self.refinement.setChecked(self.config["refinement"]["enabled"])
        self.ai_opacity.setValue(self.config["blend"]["ai_opacity"])
        self.ai_boundary_guard.setChecked(bool(self.config["blend"].get("ai_boundary_guard", False)))
        self._update_engine_controls()

    def _update_engine_controls(self):
        diffusion = self.repair_engine.currentData() != 'neural_alignment'
        self.protect_detail.setEnabled(self.repair_engine.currentData()!='diffusion')
        for control in (self.controlnet,self.controlnet_mask_mode,self.mode,self.steps,self.seed,self.strength,self.patch_size,self.patch_layout,
                        self.prompt,self.model_path,self.allow_cpu,self.use_lcm,self.ai_opacity,self.ai_boundary_guard):
            control.setEnabled(diffusion)
        hybrid = self.repair_engine.currentData() == 'hybrid'
        self.hybrid_policy.setEnabled(hybrid)
        self.hybrid_min_residual.setEnabled(hybrid and self.hybrid_policy.currentData() == 'on_structural_gain')
        self._update_controlnet_mask_control()

    def _update_controlnet_mask_control(self):
        self.controlnet_mask_mode.setEnabled(
            self.repair_engine.currentData() != 'neural_alignment'
            and self.controlnet.currentText() == 'canny'
        )

    def _engine_changed(self):
        self._update_engine_controls()
        if self.repair_engine.currentData() == 'hybrid':
            self.controlnet.setCurrentText('canny')
            self.steps.setValue(20)
            self.strength.setValue(.25)
            self.ai_opacity.setValue(.05)

    def _hybrid_policy_changed(self):
        self.hybrid_min_residual.setEnabled(
            self.repair_engine.currentData() == 'hybrid'
            and self.hybrid_policy.currentData() == 'on_structural_gain'
        )

    def _sync_config(self):
        self.config["inpainting"].update(
            engine=self.repair_engine.currentData(),
            steps=self.steps.value(), seed=self.seed.value(), strength=self.strength.value(),
            patch_size=int(self.patch_size.currentText()), prompt=self.prompt.toPlainText(),
            patch_layout=self.patch_layout.currentData(),
            model_id=self.model_path.text().strip(), allow_cpu=self.allow_cpu.isChecked(),
            controlnet=self.controlnet.currentText(),use_lcm=self.use_lcm.isChecked(),
            controlnet_mask_mode=self.controlnet_mask_mode.currentData(),
            hybrid_diffusion_policy=self.hybrid_policy.currentData(),
            hybrid_min_residual_error=self.hybrid_min_residual.value(),
            guidance_scale=1.0 if self.use_lcm.isChecked() else 4.0,
        )
        self.config["mask"]["seam_width"] = self.seam_width.value()
        self.config["registration"]["method"] = self.registration_method.currentText()
        self.config["refinement"]["enabled"] = self.refinement.isChecked()
        self.config['clear_fusion']['enabled']=self.clear_fusion.isChecked()
        self.config['neural_alignment']['preserve_detail']=self.protect_detail.isChecked()
        self.config["blend"]["ai_opacity"] = self.ai_opacity.value()
        self.config["blend"]["ai_boundary_guard"] = self.ai_boundary_guard.isChecked()
        validate_config(self.config)
        self.pipeline.config = self.config

    def _choose_image(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.sequence_list.clear()
            self.sequence_group.hide()
            self.set_input(slot, path)

    def _choose_sequence(self):
        paths,_=QFileDialog.getOpenFileNames(self,"选择按重叠顺序排列的图片","","图片 (*.png *.jpg *.jpeg *.tif *.tiff)")
        if len(paths)<2:
            return
        self.set_input("a",paths[0])
        self.set_input("b",paths[-1])
        self.sequence_list.clear()
        for path in paths:
            item=QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole,path)
            item.setToolTip(path)
            self.sequence_list.addItem(item)
        self.sequence_group.show()
        self.path_label.setText(f"已导入 {len(paths)} 张图片；请确认左侧列表中的相邻顺序。")

    def _sequence_reordered(self,*args):
        self.result=None
        self.last_output=None
        for view in (self.traditional_view,self.poisson_view,self.initial_view,self.mask_view,self.final_view):
            view.set_image(None)
        self._set_busy(False)
        self.status_label.setText("图片顺序已变更，请重新配准。")

    def set_input(self, slot, path):
        try:
            image = read_image(path)
            if slot == "a":
                self.image_a_path = str(path)
                self.source_a_view.set_image(image)
            else:
                self.image_b_path = str(path)
                self.source_b_view.set_image(image)
            self.result = None
            for view in (self.traditional_view, self.poisson_view, self.initial_view, self.mask_view, self.final_view):
                view.set_image(None)
            self.last_output = None
            self.path_label.setText(f"A: {self.image_a_path or '未选择'}    B: {self.image_b_path or '未选择'}")
            self.tabs.setCurrentIndex(0 if slot == "a" else 1)
            self._set_busy(False)
        except Exception as exc:
            self._on_error(str(exc))

    def _is_busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _set_busy(self, busy):
        self.run_button.setEnabled(not busy)
        self.ai_button.setEnabled(not busy and self.result is not None)
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(not busy and self.result is not None)
        self.quick_export_button.setEnabled(not busy and self.result is not None)
        self.job_button.setEnabled(not busy and self.result is not None)
        self.settings_group.setEnabled(not busy)
        self.clear_fusion.setEnabled(not busy)
        self.mask_group.setEnabled(not busy and self.result is not None)
        self.mask_view.setEnabled(not busy)
        self.sequence_group.setEnabled(not busy)
        for action in self.file_actions:
            action.setEnabled(not busy)

    def _start(self, use_ai):
        if self._is_busy():
            return
        if not self.image_a_path or not self.image_b_path:
            self._on_error("请先导入图片 A 和图片 B。")
            return
        if use_ai and self.result is None:
            self._on_error("请先完成自动配准。")
            return
        try:
            self._sync_config()
            custom = self.mask_view.mask if use_ai else None
            if use_ai and (custom is None or not np.any(custom)):
                raise ValueError("接缝区域为空，请绘制或重新生成自动区域。")
            self.last_output = self.output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self._set_busy(True)
            self.progress_bar.setValue(0)
            self.worker = PipelineWorker(
                self.pipeline, self.image_a_path, self.image_b_path, use_ai, custom,
                self.result if use_ai else None, self.last_output,
                [self.sequence_list.item(i).data(Qt.UserRole) for i in range(self.sequence_list.count())],
            )
            self.worker.progress.connect(self._on_progress)
            self.worker.succeeded.connect(self._on_result)
            self.worker.failed.connect(self._on_error)
            self.worker.finished.connect(self._worker_finished)
            self.worker.start()
        except Exception as exc:
            self._on_error(str(exc))
            self._set_busy(False)

    def _worker_finished(self):
        self._set_busy(False)
        if self._closing:
            self.close()

    def _cancel(self):
        if self._is_busy():
            self.worker.requestInterruption()
            self.status_label.setText("正在取消，将在当前加载或计算步骤结束后停止。")

    def _on_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)

    def _on_result(self, result):
        self.result = result
        self.source_a_view.set_image(result.registration.image_a)
        self.source_b_view.set_image(result.registration.image_b)
        self.traditional_view.set_image(result.traditional_image)
        self.poisson_view.set_image(result.poisson_image)
        initial = result.repair_base_image if result.repair_base_image is not None else result.traditional_image
        self.initial_view.set_image(initial)
        self.mask_view.set_image(initial, result.mask.binary_mask)
        self.final_view.set_image(result.final_image if result.metrics["use_ai"] else None)
        self.tabs.setCurrentWidget(self.final_view if result.metrics["use_ai"] else
                                   self.initial_view if result.metrics.get('clear_fusion_enabled') else self.mask_view)
        self.status_label.setText(f"完成 · {result.metrics['total_seconds']:.2f} 秒 · {result.final_image.shape[1]}×{result.final_image.shape[0]}")
        if result.metrics.get('clear_fusion_enabled') and not result.metrics['use_ai']:
            selection=result.metrics.get('clear_selection')
            detail={'a':'优先保留原图 A 的纹理','b':'优先保留原图 B 的纹理'}.get(selection,'清晰度相近或已经对齐，保持原有融合')
            if result.metrics.get('clear_noise_guard_applied'):detail+=' · 已避开噪点较多的原图'
            self.status_label.setText(self.status_label.text()+' · '+detail)
        if result.metrics.get('repair_revision') and not result.metrics.get('model_id'):
            reason=result.metrics.get('repair_reason')
            detail={'accepted':'已修正局部错位，并保留清晰细节',
                    'already_aligned':'原图已对齐，保留底图',
                    'clear_source_already_selected':'底图已保留可靠纹理，无需重复修复',
                    'insufficient_alignment_gain':'未确认错位改善，已保留底图',
                    'empty_reliable_region':'没有可靠的局部修复区域，已保留底图'}.get(reason,'结构修复完成')
            self.status_label.setText(self.status_label.text()+' · '+detail)
        if result.metrics.get('diffusion_skipped'):
            self.status_label.setText(self.status_label.text()+' · 已保留当前底图，未运行生成修复')
        elif result.metrics.get('diffusion_executed'):
            self.status_label.setText(self.status_label.text()+' · 已运行生成修复，可切换底图比较')
        self.progress_bar.setValue(100)
        if self.last_output:
            self.log.appendPlainText(f"运行记录：{self.last_output.resolve()}")

    def _on_error(self, message):
        self.status_label.setText("处理未完成，详情见运行记录")
        self.progress_bar.setValue(0)
        self.log.appendPlainText(message)

    def _reset_mask(self):
        if self.result:
            self._sync_config()
            mask = self.result.mask if self.result.metrics.get("sequence_count",2)>2 else generate_seam_mask(self.result.registration, self.config["mask"])
            self.mask_view.set_image(self.initial_view._image, mask.binary_mask)

    def _set_brush_mode(self, erase):
        """Keep the editing mode visible while preserving the existing mask API."""
        self.mask_view.set_brush(self.brush_size.value(), erase)
        self.paint_button.setChecked(not erase)
        self.erase_button.setChecked(erase)
        self.status_label.setText("接缝编辑 · 橡皮擦除模式" if erase else "接缝编辑 · 画笔添加模式")

    def _save(self):
        view = self.tabs.currentWidget()
        if view._image is None:
            self._on_error("当前页没有可导出的图片。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出原始分辨率图片", "xiyuan_result.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            try:
                write_image(path, view._image)
                self.log.appendPlainText(f"已导出：{path}")
            except Exception as exc:
                self._on_error(str(exc))

    def _open_run(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开本机或 GPU 返回的运行记录", "", "运行记录 (run.json)")
        if path:
            try:
                result, report = load_run(path)
                self.config = load_config(overrides=report["config"])
                self.pipeline = StitchPipeline(self.config)
                self.image_a_path = str(Path(path).parent / "input_a.png")
                self.image_b_path = str(Path(path).parent / "input_b.png")
                self.sequence_list.clear()
                for name, artifact in sorted(report["artifacts"].items()):
                    if name.startswith("sequence_"):
                        source_path = str(Path(path).parent / artifact["file"])
                        item = QListWidgetItem(artifact["file"])
                        item.setData(Qt.UserRole, source_path)
                        item.setToolTip(source_path)
                        self.sequence_list.addItem(item)
                self.sequence_group.setVisible(self.sequence_list.count() > 2)
                self.last_output = Path(path).parent
                self._load_controls()
                self._on_result(result)
                self.path_label.setText(f"已打开：{path}")
                self._set_busy(False)
            except Exception as exc:
                self._on_error(str(exc))

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入配置", "", "YAML (*.yaml *.yml)")
        if path:
            try:
                self.config = load_config(path)
                self.pipeline.config = self.config
                self._load_controls()
            except Exception as exc:
                self._on_error(str(exc))

    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出配置", "xiyuan-config.yaml", "YAML (*.yaml)")
        if path:
            try:
                self._sync_config()
                Path(path).write_text(yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            except Exception as exc:
                self._on_error(str(exc))

    def _export_job(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 GPU 任务", "xiyuan-job.zip", "ZIP (*.zip)")
        if not path:
            return
        try:
            self._sync_config()
            mask = self.mask_view.mask
            if mask is None or not np.any(mask):
                raise ValueError("请先选择需要修复的接缝区域。")
            config = deepcopy(self.config)
            config["inpainting"].update(device="cuda", allow_cpu=False)
            # A local Windows model path is not usable on a hosted Linux runtime.
            if Path(config["inpainting"]["model_id"]).exists():
                config["inpainting"]["model_id"] = load_config()["inpainting"]["model_id"]
                config["inpainting"]["revision"] = load_config()["inpainting"]["revision"]
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, image in (("a.png", self.result.registration.image_a),
                                    ("b.png", self.result.registration.image_b), ("mask.png", mask)):
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise ValueError("无法编码输入图片。")
                    archive.writestr(name, encoded.tobytes())
                prepared_folder=self.output_root / "gpu-tasks" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                save_run(prepared_folder,self.result,self.config)
                for artifact in prepared_folder.iterdir():
                    if artifact.is_file():
                        archive.write(artifact,"prepared/"+artifact.name)
                manifest = {"schema_version": 1, "description": "User selected image pair and seam mask; preserved prepared geometry.",
                            "cases": [{"id": "user_pair", "category": "user", "kind": "user_capture",
                                       "source_id": "user_pair", "image_a": "a.png", "image_b": "b.png", "mask": "mask.png",
                                       "prepared_run":"prepared/run.json"}]}
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                archive.writestr("config.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            self.log.appendPlainText(f"已导出 GPU 任务：{path}。在免费 GPU 笔记本上传此文件运行，下载结果后打开 run.json。")
        except Exception as exc:
            self._on_error(str(exc))

    def closeEvent(self, event):
        if self._is_busy():
            self._closing = True
            self._cancel()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
