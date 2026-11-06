"""
High-Performance Image Viewer Window Module.

Provides a DearPyGui-based modular image display canvas supporting GPU texture
streaming, interactive ROI analysis, colormapping, intensity rescaling, transparent
rendering, and image export pipelines.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.automation_manager import automation_manager
from core.config_manager import config
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.tools.clipboard_injector import clipboardinjector


class Image_viewer_win(WindowBase):
    """
    Image_viewer_win:
    A DearPyGui-based modular window to display images with support for:
    - grayscale rescaling
    - color palette application
    - FPS measurement
    - annotated scale overlay
    - clipboard/image export
    - input handling from various sources (file, folder, ndarray)
    - ROI selection and management
    """

    description: str = (
        "High-performance modular image viewer with ROI analysis, colormapping, and texture streaming."
    )

    # 1. Window Configuration
    label: str = "Image Viewer"
    win_width: int = display_scaling.scale(1280)
    win_height: int = display_scaling.scale(900)

    # 2. Input/Output definitions
    accepted_input_types: list[type] = [
        IOTypes.FRAME,
        IOTypes.FILE_PATH,
        IOTypes.FRAME12,
        IOTypes.FRAME16,
        IOTypes.ROI_SAMPLE,
        IOTypes.CMD_DICT,
        IOTypes.FOLDER_PATH,
        IOTypes.FRAME_MASK_PAIR,
        IOTypes.MASK,
    ]
    outputs: dict[str, type] = {
        "Frame": IOTypes.FRAME,
        "ROI Data": IOTypes.ROI_SAMPLE,
        "Cropped Frame": IOTypes.FRAME,
        "Lowdepth Frame": IOTypes.FRAME,
    }

    # 3. Persistent fields
    persistent_fields: list[str] = [
        "label",
        "colorize",
        "intensity_rescaling",
        "fps_counter",
        "image_export",
        "image_depth",
        "show_scale",
        "export_path",
        "categories_count",
        "zero_as_alpha",
    ]

    # Persistent field defaults
    colorize: bool = True
    intensity_rescaling: bool = True
    fps_counter: bool = True
    image_export: bool = True
    image_depth: str = "12bit"
    show_scale: bool = True
    export_path: Optional[str] = None
    categories_count: int = 5
    zero_as_alpha: bool = False
    texture_mode: str = "raw"
    preserve_aspect_ratio: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.use_rgba = True

        # FPS tracking
        self.fps = 0.0
        self.fps_now = time.time()
        self.fps_last = self.fps_now

        # Locks for thread synchronization
        self.lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.update_event = threading.Event()

        # Image reference and display state
        self.last_image: Optional[np.ndarray] = None
        self.negative = False
        self.palette = "Inferno"
        self.imgcount = 0

        # Performance optimization: frame counter for FPS display
        self._frame_count = 0
        self._fps_update_interval = 10

        # Texture mode configuration from config.json (fallback: "raw")
        viewer_cfg = config.get("Viewers", {}).get("image_viewer", {})
        default_mode = viewer_cfg if isinstance(viewer_cfg, str) else viewer_cfg.get("texture_mode", "raw")
        self.texture_mode = str(default_mode).lower() if default_mode else "raw"
        if self.texture_mode not in ("raw", "static"):
            self.texture_mode = "raw"

        # Aspect ratio configuration from config.json (fallback: True)
        par = None
        if isinstance(viewer_cfg, dict):
            par = viewer_cfg.get("preserve_aspect_ratio")
        if par is None:
            par = config.get("Viewers", {}).get("preserve_aspect_ratio", True)

        if kwargs.get("stretch", False):
            self.preserve_aspect_ratio = False
        elif isinstance(par, str):
            self.preserve_aspect_ratio = par.strip().lower() in ("true", "1", "yes", "preserve")
        else:
            self.preserve_aspect_ratio = bool(par)

        # Cast / ensure field types defensively
        raw_colorize = getattr(self, "colorize", None)
        self.colorize: bool = bool(raw_colorize) if raw_colorize is not None else True

        raw_rescaling = getattr(self, "intensity_rescaling", None)
        self.intensity_rescaling: bool = (
            bool(raw_rescaling) if raw_rescaling is not None else True
        )

        raw_fps = getattr(self, "fps_counter", None)
        self.fps_counter: bool = bool(raw_fps) if raw_fps is not None else True

        raw_export = getattr(self, "image_export", None)
        self.image_export: bool = bool(raw_export) if raw_export is not None else True

        raw_depth = getattr(self, "image_depth", None)
        self.image_depth: str = str(raw_depth) if raw_depth is not None else "12bit"

        raw_scale = getattr(self, "show_scale", None)
        self.show_scale: bool = bool(raw_scale) if raw_scale is not None else True

        raw_export_path = getattr(self, "export_path", None)
        self.export_path: Optional[str] = str(raw_export_path) if raw_export_path else None

        raw_cats = getattr(self, "categories_count", None)
        self.categories_count: int = int(raw_cats) if raw_cats is not None else 5

        raw_alpha = getattr(self, "zero_as_alpha", None)
        self.zero_as_alpha: bool = bool(raw_alpha) if raw_alpha is not None else False

        # Texture state
        self.texture_width = 0
        self.texture_height = 0
        self._texture_initialized = False
        self._texture_buffer: Optional[np.ndarray] = None
        self._remap_lut_key: Optional[Tuple[int, int]] = None
        self._remap_lut: Optional[np.ndarray] = None
        self._scale_cache_key: Optional[Tuple[int, int, int, str, bool]] = None
        self._scale_cache_bar: Optional[np.ndarray] = None

        # DPG element tags
        self.texture_tag = f"viewer_texture_{self.UUID}"
        self.series_tag = f"viewer_series_{self.UUID}"
        self.plot_tag = f"viewer_plot_{self.UUID}"
        self.xaxis_tag = f"viewer_xaxis_{self.UUID}"
        self.yaxis_tag = f"viewer_yaxis_{self.UUID}"
        self.fps_tag = f"viewer_fps_{self.UUID}"
        self.name_tag = f"viewer_name_{self.UUID}"
        self.count_tag = f"viewer_count_{self.UUID}"
        self.clipboard_btn = f"viewer_clipboard_{self.UUID}"
        self.save_btn = f"viewer_save_{self.UUID}"
        self.auto_remap_tag = f"viewer_autoremap_{self.UUID}"
        self.show_scale_tag = f"viewer_showscale_{self.UUID}"
        self.inf_tag = f"viewer_inf_{self.UUID}"
        self.sup_tag = f"viewer_sup_{self.UUID}"
        self.texture_reg_tag = f"viewer_tex_reg_{self.UUID}"
        self.crop_limit_tag = f"viewer_crop_limit_{self.UUID}"
        self.zero_as_alpha_tag = f"viewer_zero_alpha_{self.UUID}"
        dpg.add_texture_registry(tag=self.texture_reg_tag, show=False)

        # ROI tags
        self.roi_enable_tag = f"viewer_roi_enable_{self.UUID}"
        self.roi_w_tag = f"viewer_roi_w_{self.UUID}"
        self.roi_h_tag = f"viewer_roi_h_{self.UUID}"
        self.roi_thresh_tag = f"viewer_roi_thresh_{self.UUID}"
        self.roi_show_mask_tag = f"viewer_roi_showmask_{self.UUID}"
        self.roi_live_crop_tag = f"viewer_roi_livecrop_{self.UUID}"
        self.roi_result_tag = f"viewer_roi_result_{self.UUID}"

        # Tags for permission management
        self.mode_combo_tag = f"viewer_mode_combo_{self.UUID}"
        self.btn_clear_rois_tag = f"viewer_clear_rois_{self.UUID}"
        self.export_group_tag = f"viewer_export_group_{self.UUID}"
        self.export_save_group_tag = f"viewer_export_save_group_{self.UUID}"
        self.fps_group_tag = f"viewer_fps_group_{self.UUID}"
        self.roi_thresh_group_tag = f"viewer_roi_thresh_group_{self.UUID}"
        self.palette_combo_tag = f"viewer_palette_combo_{self.UUID}"
        self.depth_combo_tag = f"viewer_depth_combo_{self.UUID}"
        self.scale_group_tag = f"viewer_scale_group_{self.UUID}"

        # Category mapping tags
        self.cat_group_tag = f"viewer_cat_group_{self.UUID}"
        self.cat_count_tag = f"viewer_cat_count_{self.UUID}"
        self.btn_send_palette_tag = f"viewer_send_palette_{self.UUID}"
        self.btn_send_result_tag = f"viewer_send_result_{self.UUID}"

        # Overlay drawing tags
        self.roi_draw_layer = f"viewer_roi_draw_{self.UUID}"

        # Interactive state
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_hovering = False

        # ROI Storage
        self.rois: Dict[Union[int, str], Dict[str, Any]] = {}
        self.last_selected_roi_uuid: Optional[Union[int, str]] = None
        self.roi_index_counter = 1

        # Rendering State
        self.running = True
        self.latest_frame: Optional[np.ndarray] = None

        self._build_interface()
        self.update_permission()

        # Start Render Thread
        self.worker_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.worker_thread.start()

    def update_permission(self) -> None:
        """
        Adapt UI based on app_state.mode.
        In 'user' mode: keep palette, auto rescale, inf/sup, ROI W/H/%/Show, and clipboard button visible.
        Hide expert controls (mode, depth, scale bar, fps, live crop/clear, export save options, categories).
        """
        mode = app_state.mode

        # Elements to toggle (visible in expert mode, hidden in user mode)
        elements = [
            self.mode_combo_tag,
            self.depth_combo_tag,
            self.scale_group_tag,
            self.fps_group_tag,
            self.roi_enable_tag,
            self.roi_live_crop_tag,
            self.btn_clear_rois_tag,
            self.export_save_group_tag,
            self.cat_group_tag,
        ]

        show = (mode != "user")
        for tag in elements:
            if dpg.does_item_exist(tag):
                if show:
                    dpg.show_item(tag)
                else:
                    dpg.hide_item(tag)

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(mode == "user"))

    def _build_interface(self) -> None:
        """Creates the DearPyGui layout using self.window()."""
        scale = display_scaling.scale
        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=0)
                if self.colorize:
                    dpg.add_combo(
                        ("Normal", "Negative"),
                        tag=self.mode_combo_tag,
                        width=scale(120),
                        default_value="Normal",
                        callback=self.set_mode,
                    )
                    with dpg.tooltip(parent=self.mode_combo_tag):
                        dpg.add_text("Normal: Normal colorization\nNegative: Inverted colorization")

                    dpg.add_combo(
                        ("B&W", "Inferno", "Jet", "HSV"),
                        tag=self.palette_combo_tag,
                        width=scale(120),
                        default_value=self.palette,
                        callback=self.set_mode,
                    )
                    with dpg.tooltip(parent=self.palette_combo_tag):
                        dpg.add_text("B&W: Grayscale\nInferno: Perceptually uniform colormap\nJet: Rainbow colormap\nHSV: Hue-saturation-value colormap")

                    dpg.add_combo(
                        ("8bit", "12bit", "16bit_norm", "16bit_abs"),
                        tag=self.depth_combo_tag,
                        width=scale(120),
                        default_value=self.image_depth,
                        callback=self.set_depth,
                    )
                    with dpg.tooltip(parent=self.depth_combo_tag):
                        dpg.add_text("8bit: 8-bit depth\n12bit: 12-bit depth\n16bit_norm: 16-bit normalized\n16bit_abs: 16-bit absolute")

                if self.intensity_rescaling:
                    dpg.add_text("Auto:")
                    dpg.add_checkbox(default_value=True, tag=self.auto_remap_tag, callback=self.update_image_callback)
                    with dpg.tooltip(parent=self.auto_remap_tag):
                        dpg.add_text("Normalize the intensity of the image")

                    with dpg.group(horizontal=True, tag=self.scale_group_tag):
                        dpg.add_text("Scale:")
                        dpg.add_checkbox(default_value=self.show_scale, tag=self.show_scale_tag, callback=self.update_image_callback)
                        with dpg.tooltip(parent=self.show_scale_tag):
                            dpg.add_text("Adds a color scale bar on the right to show the intensity scale")

                    dpg.add_text("Inf:")
                    dpg.add_drag_int(
                        default_value=0,
                        min_value=0,
                        max_value=255,
                        width=scale(70),
                        tag=self.inf_tag,
                        callback=self.update_image_callback,
                    )
                    with dpg.tooltip(parent=self.inf_tag):
                        dpg.add_text("Lower manual threshold for linear intensity remapping")

                    dpg.add_text("Sup:")
                    dpg.add_drag_int(
                        default_value=255,
                        min_value=0,
                        max_value=255,
                        width=scale(70),
                        tag=self.sup_tag,
                        callback=self.update_image_callback,
                    )
                    with dpg.tooltip(parent=self.sup_tag):
                        dpg.add_text("Upper manual threshold for linear intensity remapping")

                if self.fps_counter:
                    with dpg.group(horizontal=True, tag=self.fps_group_tag):
                        dpg.add_text("| FPS:")
                        dpg.add_text("0", tag=self.fps_tag)

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=0)
                dpg.add_checkbox(label="ROI", default_value=True, tag=self.roi_enable_tag, callback=self.update_image_callback)
                with dpg.tooltip(parent=self.roi_enable_tag):
                    dpg.add_text("Enable/disable ROI creation and manipulation")

                dpg.add_text("W:")
                dpg.add_drag_int(width=scale(60), tag=self.roi_w_tag, default_value=25, min_value=1, callback=self.update_image_callback)
                with dpg.tooltip(parent=self.roi_w_tag):
                    dpg.add_text("Width of the ROI in pixels")

                dpg.add_text("H:")
                dpg.add_drag_int(width=scale(60), tag=self.roi_h_tag, default_value=25, min_value=1, callback=self.update_image_callback)
                with dpg.tooltip(parent=self.roi_h_tag):
                    dpg.add_text("Height of the ROI in pixels")

                with dpg.group(horizontal=True, tag=self.roi_thresh_group_tag):
                    dpg.add_text("%")
                    dpg.add_drag_int(width=scale(60), tag=self.roi_thresh_tag, default_value=100, min_value=1, max_value=100, callback=self.update_image_callback)
                with dpg.tooltip(parent=self.roi_thresh_tag):
                    dpg.add_text("Threshold of the pixels to be included in the mean value calculation")

                dpg.add_checkbox(label="Show", tag=self.roi_show_mask_tag, callback=self.update_image_callback)
                with dpg.tooltip(parent=self.roi_show_mask_tag):
                    dpg.add_text("Show/hide the selected pixels")

                dpg.add_checkbox(label="Live Crop", tag=self.roi_live_crop_tag, default_value=False)
                with dpg.tooltip(parent=self.roi_live_crop_tag):
                    dpg.add_text("Enable/disable live cropping of the selected ROI")

                dpg.add_button(label="Clear", tag=self.btn_clear_rois_tag, callback=self.clear_rois)
                with dpg.tooltip(parent=self.btn_clear_rois_tag):
                    dpg.add_text("Clear all ROIs")

                dpg.add_text("Mean Value:", color=(255, 100, 100))
                dpg.add_text("0.0", tag=self.roi_result_tag)

            if self.image_export:
                with dpg.group(horizontal=True, tag=self.export_group_tag):
                    dpg.add_button(label="Clipboard", tag=self.clipboard_btn, callback=self.copy_to_clipboard)
                    with dpg.tooltip(parent=self.clipboard_btn):
                        dpg.add_text("Copy the processed image to the clipboard")

                    with dpg.group(horizontal=True, tag=self.export_save_group_tag):
                        dpg.add_input_text(default_value="Image", width=scale(150), tag=self.name_tag, callback=self.reset_counter)
                        with dpg.tooltip(parent=self.name_tag):
                            dpg.add_text("Name of the image files")

                        dpg.add_text("count: 0", tag=self.count_tag)
                        with dpg.tooltip(parent=self.count_tag):
                            dpg.add_text("Number of images saved")

                        dpg.add_button(label="Save", tag=self.save_btn, callback=self.save_image)

                        self.save_tooltip_text_tag = f"viewer_save_tooltip_text_{self.UUID}"
                        with dpg.tooltip(parent=self.save_btn):
                            initial_tip = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
                            dpg.add_text(initial_tip, tag=self.save_tooltip_text_tag)

                        dpg.add_checkbox(label="Crop limit", default_value=False, tag=self.crop_limit_tag, callback=self.update_image_callback)
                        with dpg.tooltip(parent=self.crop_limit_tag):
                            dpg.add_text("Crops to the minimal bounding box where alpha > 0")

                        dpg.add_checkbox(label="Zero as alpha", default_value=self.zero_as_alpha, tag=self.zero_as_alpha_tag, callback=self.update_image_callback)
                        with dpg.tooltip(parent=self.zero_as_alpha_tag):
                            dpg.add_text("Sets alpha to 0 for all zero-value pixels \n to generate an RGBA image with transparent background")

            # Category mapping group
            with dpg.group(horizontal=True, tag=self.cat_group_tag):
                dpg.add_text("Categories:")
                dpg.add_input_int(default_value=self.categories_count, min_value=2, max_value=20, step=1, width=scale(150), tag=self.cat_count_tag, callback=self._update_categories_count)
                dpg.add_button(label="Send Palette", tag=self.btn_send_palette_tag, callback=self.send_palette_to_clipboard)
                with dpg.tooltip(parent=self.btn_send_palette_tag):
                    dpg.add_text("Sends the current palette to the clipboard")
                dpg.add_button(label="Send Result", tag=self.btn_send_result_tag, callback=self.send_result_to_clipboard)
                with dpg.tooltip(parent=self.btn_send_result_tag):
                    dpg.add_text("Sends the current result to the clipboard")

            # Add mouse handler for ROI interaction
            self.handler_reg_tag = f"viewer_handler_reg_{self.UUID}"
            with dpg.handler_registry(tag=self.handler_reg_tag):
                dpg.add_mouse_move_handler(callback=self.on_mouse_move)
                dpg.add_mouse_click_handler(callback=self.on_mouse_click)

            self.init_viewer(1280, 720)
            self._apply_transparency_theme()

    def on_mouse_move(self, sender: Any = None, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Tracks mouse position for ROI preview."""
        try:
            if dpg.does_item_exist(self.plot_tag) and dpg.is_item_hovered(self.plot_tag):
                mpos = dpg.get_plot_mouse_pos()
                mx = int(mpos[0])
                my = int(mpos[1])
                if not self.is_hovering or mx != self.mouse_x or my != self.mouse_y:
                    self.is_hovering = True
                    self.mouse_x = mx
                    self.mouse_y = my

                    if dpg.get_value(self.roi_show_mask_tag) and self.last_image is not None:
                        self.update_image(self.last_image)

                    self._update_hover_roi()
            elif self.is_hovering:
                self.is_hovering = False
                self._update_hover_roi()
        except Exception:
            self.is_hovering = False

    def on_mouse_click(self, sender: Any = None, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Handles mouse clicks, specifically CTRL+Click or Middle Click for ROI creation."""
        btn = data[0] if isinstance(data, (list, tuple)) else data
        is_ctrl_click = (btn == 0 and dpg.is_key_down(dpg.mvKey_LControl))
        is_middle_click = (btn == 2)

        if not (is_ctrl_click or is_middle_click):
            return

        if not dpg.get_value(self.roi_enable_tag):
            return

        try:
            if not dpg.does_item_exist(self.plot_tag) or not dpg.is_item_hovered(self.plot_tag):
                return
            mpos = dpg.get_plot_mouse_pos()
            pos_x = int(mpos[0])
            pos_y = int(mpos[1])
        except Exception:
            return

        # Create new ROI
        roi_uuid = dpg.generate_uuid()
        name = str(self.roi_index_counter)
        self.roi_index_counter += 1

        w = dpg.get_value(self.roi_w_tag)
        h = dpg.get_value(self.roi_h_tag)

        # Center logic
        x = pos_x - w // 2
        y = pos_y - h // 2

        # Store ROI
        self.rois[roi_uuid] = {
            "uuid": roi_uuid,
            "name": name,
            "rect": [x, y, w, h],
            "selected": True,
            "threshold": dpg.get_value(self.roi_thresh_tag),
        }

        self.last_selected_roi_uuid = roi_uuid
        self._update_single_persistent_roi(roi_uuid)
        self._emit_roi(self.rois[roi_uuid], action="create")
        if self.last_image is not None:
            self._emit_cropped_frame(self.last_image, force_emit=True)

    def _emit_roi(self, roi_data: Optional[Dict[str, Any]], action: str = "select") -> None:
        """Emits ROI_SAMPLE to outputs."""
        if action == "clear" or roi_data is None:
            sample = {"action": "clear", "clear": True}
        else:
            sample = {
                "uuid": roi_data["uuid"],
                "name": roi_data["name"],
                "rect": roi_data["rect"],
                "threshold": roi_data.get("threshold", 100),
                "action": action,
            }

        for module in self.connections.get("ROI Data", []):
            try:
                module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)
            except Exception as e:
                logger.debug(f"[{self.label}] Error emitting ROI: {e}")

    def clear_rois(self, *args: Any, **kwargs: Any) -> None:
        """Clears all ROIs from the viewer and notifies connected modules."""
        self._handle_roi_input({"clear": True})
        self._emit_roi(None, action="clear")
        self.roi_index_counter = 1
        self.last_selected_roi_uuid = None
        self.update_image_callback()

    def _mask_to_alpha(self, mask: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert a boolean or numeric mask into a uint8 alpha channel."""
        if mask.dtype == bool:
            alpha = np.where(mask, np.uint8(255), np.uint8(0))
        elif mask.dtype == np.uint8 and mask.max() > 1:
            alpha = mask
        elif mask.max() <= 1 and np.issubdtype(mask.dtype, np.number):
            alpha = (mask * 255).astype(np.uint8)
        else:
            alpha = np.clip(mask, 0, 255).astype(np.uint8)

        if target_shape is not None and alpha.shape[:2] != target_shape:
            alpha = cv2.resize(alpha, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return alpha

    def _compose_frame_with_mask(self, raw_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Combine a frame and a mask into a 4-channel BGRA image."""
        h, w = raw_frame.shape[:2]
        alpha = self._mask_to_alpha(mask, (h, w))
        base = self.convert_to_lowdepth(raw_frame) if raw_frame.dtype != np.uint8 else raw_frame

        if base.ndim == 2 or (base.ndim == 3 and base.shape[2] == 1):
            bgra = cv2.cvtColor(base, cv2.COLOR_GRAY2BGRA)
            bgra[:, :, 3] = alpha
            return bgra
        elif base.shape[2] == 4:
            bgra = base.copy()
            bgra[:, :, 3] = alpha
            return bgra
        else:
            bgra = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            return bgra

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Node editor input callback.
        Accepts:
          - ROI_SAMPLE (sample dict)
          - CMD_DICT (commands: reset_counter, clear, etc.)
          - FOLDER_PATH (export path)
          - FRAME_MASK_PAIR (tuple/list of [frame, mask])
          - MASK (np.ndarray mask)
          - (frame, name) tuple
          - dict with "frame" and optional "mask"
          - file path string
          - raw ndarray frame (Grayscale, RGB, RGBA)
        """
        # 1. Handle ROI Input
        if kwargs.get("data_type") == IOTypes.ROI_SAMPLE or (
            isinstance(kwargs.get("sample"), dict) and "rect" in kwargs["sample"]
        ):
            self._handle_roi_input(kwargs["sample"])
            return

        # 2. Handle CMD_DICT
        if kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("sample")
            if isinstance(cmd, dict):
                if cmd.get("reset_counter"):
                    self.roi_index_counter = 1
                if cmd.get("clear") or cmd.get("action") == "clear":
                    self._handle_roi_input({"clear": True})
            return

        # 3. Handle FOLDER_PATH
        data_type = kwargs.get("data_type")
        path_val = kwargs.get("path")
        if path_val is None:
            path_val = kwargs.get("data")
        if path_val is None:
            path_val = args[0] if (args and not isinstance(args[0], np.ndarray)) else None

        if data_type == IOTypes.FOLDER_PATH or (isinstance(path_val, str) and os.path.isdir(path_val)):
            if isinstance(path_val, str):
                self.export_path = path_val
                logger.info(f"[{self.label}] Export path set to: {self.export_path}")
                self.update_save_tooltip()
                return

        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        if data is None and "sample" in kwargs:
            data = kwargs["sample"]
        if data is None and "frame" in kwargs:
            data = kwargs["frame"]

        if data is None:
            return

        # 4. Check for FRAME_MASK_PAIR: (frame, mask)
        is_frame_mask_pair = (
            data_type == IOTypes.FRAME_MASK_PAIR
            or (isinstance(data, (tuple, list)) and len(data) == 2 
                and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray))
        )

        if is_frame_mask_pair:
            self.update_image(self._compose_frame_with_mask(data[0], data[1]))
            return

        # 5. Dict with "frame" and optional "mask"
        if isinstance(data, dict) and "frame" in data:
            raw_frame = data["frame"]
            if "mask" in data and data["mask"] is not None and isinstance(data["mask"], np.ndarray):
                self.update_image(self._compose_frame_with_mask(raw_frame, data["mask"]))
                return
            else:
                data = raw_frame

        # 6. Single mask input
        if data_type == IOTypes.MASK and isinstance(data, np.ndarray):
            alpha = self._mask_to_alpha(data)
            frame = cv2.merge([alpha, alpha, alpha, alpha])
            self.update_image(frame)
            return

        # 7. Tuple input (frame, name)
        if isinstance(data, (tuple, list)) and len(data) >= 1 and isinstance(data[0], np.ndarray):
            data = data[0]

        # 8. Filepath string
        if isinstance(data, str) and os.path.exists(data) and not os.path.isdir(data):
            img = cv2.imread(data, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.update_image(img)
                return

        # 9. Raw ndarray frame
        if isinstance(data, np.ndarray):
            self.update_image(data)
            return

    def convert_to_lowdepth(self, frame: np.ndarray) -> np.ndarray:
        """Converts high bit-depth image to 8-bit based on image_depth."""
        match self.image_depth:
            case "8bit":
                return frame
            case "12bit":
                return (frame >> 4).astype(np.uint8)
            case "16bit_norm":
                return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            case "16bit_abs":
                if frame.dtype == np.uint16 and frame.flags.c_contiguous:
                    return np.ascontiguousarray(frame.view(np.uint8)[..., 1::2])
                return (frame >> 8).astype(np.uint8)
            case _:
                return frame

    def get_minmax_values(self, frame: np.ndarray) -> Tuple[int, int]:
        """Returns min and max intensity values. Optimized with OpenCV."""
        if frame is None or frame.size == 0:
            return 0, 255

        if frame.ndim == 2:
            min_val, max_val, _, _ = cv2.minMaxLoc(frame)
        else:
            min_val = np.min(frame)
            max_val = np.max(frame)

        return int(min_val), int(max_val)

    def _lowdepth_to_original(self, val_8bit: float, frame: Optional[np.ndarray]) -> float:
        """Converts an 8-bit value (from sliders) back to the original image depth range."""
        if frame is None:
            return val_8bit

        match self.image_depth:
            case "8bit":
                return val_8bit
            case "12bit":
                return int(val_8bit) << 4
            case "16bit_abs":
                return int(val_8bit) << 8
            case "16bit_norm":
                fmin, fmax = self.get_minmax_values(frame)
                return fmin + (val_8bit / 255.0) * (fmax - fmin)
            case _:
                return val_8bit

    def _get_current_scale_limits(self) -> Tuple[float, float]:
        """Returns (minval, maxval) in original precision based on UI state (Auto vs Manual)."""
        if self.last_image is None:
            return 0.0, 255.0

        if dpg.get_value(self.auto_remap_tag):
            sample = self.last_image[::5, ::5]
            minval = float(np.percentile(sample, 1))
            maxval = float(np.max(sample))
            return minval, maxval
        else:
            ui_inf = dpg.get_value(self.inf_tag)
            ui_sup = dpg.get_value(self.sup_tag)
            minval = float(self._lowdepth_to_original(ui_inf, self.last_image))
            maxval = float(self._lowdepth_to_original(ui_sup, self.last_image))
            return minval, maxval

    def remap_pixel_intensity(self, frame: np.ndarray, low: float, high: float) -> np.ndarray:
        """
        Normalizes frame between [low, high] and scales to 8-bit uint8.
        Pixels <= low are clamped to 0 (black/cleared).
        Pixels >= high are clamped to 255 (saturated/white).
        Pixels between low and high are scaled linearly.
        """
        low_f = float(low)
        high_f = float(high)

        # Fast path: full dynamic range on uint8
        if int(low_f) <= 0 and int(high_f) >= 255 and frame.dtype == np.uint8:
            return frame

        if frame.dtype == np.uint8:
            low_i = max(0, min(255, int(round(low_f))))
            high_i = max(0, min(255, int(round(high_f))))
            lut_key = (low_i, high_i)

            if self._remap_lut is None or self._remap_lut_key != lut_key:
                lut = np.zeros(256, dtype=np.uint8)
                if high_i > low_i:
                    lut[high_i:] = 255
                    ramp = np.linspace(0, 255, high_i - low_i + 1, endpoint=True)
                    lut[low_i:high_i + 1] = np.clip(np.round(ramp), 0, 255).astype(np.uint8)
                else:
                    lut[high_i:] = 255
                self._remap_lut_key = lut_key
                self._remap_lut = lut

            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            return cv2.LUT(frame, self._remap_lut)
        else:
            diff = high_f - low_f
            if diff <= 0:
                diff = 1.0
            clipped = np.clip(frame, low_f, high_f)
            scaled = (clipped.astype(np.float32) - low_f) * (255.0 / diff)
            return np.clip(scaled, 0, 255).astype(np.uint8)

    def add_intensity_scale(
        self, img: np.ndarray, scale_width: int = 30, position: str = "left", in_place: bool = False
    ) -> np.ndarray:
        """Adds vertical or horizontal intensity scale bar to image."""
        h = img.shape[0]
        scale_w = min(display_scaling.scale(scale_width), img.shape[1])
        if scale_w <= 0:
            return img

        n_ch = img.shape[2] if img.ndim == 3 else 1
        cache_key = (h, scale_w, n_ch, self.palette, self.negative)

        if self._scale_cache_bar is None or self._scale_cache_key != cache_key:
            gradient = np.linspace(255, 0, h).reshape((h, 1)).astype(np.uint8)
            scale = np.tile(gradient, (1, scale_w))

            if img.ndim == 3:
                if n_ch == 4:
                    scale_color = self.colorize_frame(scale)
                    if scale_color.ndim == 2:
                        scale_color = cv2.cvtColor(scale_color, cv2.COLOR_GRAY2BGR)
                    alpha_chan = np.full((h, scale_w, 1), 255, dtype=np.uint8)
                    scale = np.concatenate([scale_color[:, :, :3], alpha_chan], axis=-1)
                else:
                    scale_color = self.colorize_frame(scale)
                    if scale_color.ndim == 2 and n_ch == 3:
                        scale_color = cv2.cvtColor(scale_color, cv2.COLOR_GRAY2BGR)
                    scale = scale_color

            self._scale_cache_key = cache_key
            self._scale_cache_bar = scale
        else:
            scale = self._scale_cache_bar

        overlay = img if in_place else img.copy()
        match position:
            case "right":
                overlay[:, -scale_w:] = scale
            case "left":
                overlay[:, :scale_w] = scale
            case "top":
                overlay[:scale_w, :] = scale.T
            case "bottom":
                overlay[-scale_w:, :] = scale.T

        return overlay

    def add_scale_and_annotations(
        self,
        img: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        in_place: bool = False,
    ) -> np.ndarray:
        """Adds scale bar and min/max text to image (supports 2D, BGR, and RGBA)."""
        if img is None or img.size == 0:
            return img

        if min_val is None or max_val is None:
            min_val, max_val = self.get_minmax_values(img)

        img = self.add_intensity_scale(img, in_place=in_place)

        h = img.shape[0]
        text_color = (255, 0, 0, 255) if (img.ndim == 3 and img.shape[2] == 4) else (255, 0, 0)

        f_scale = 0.8 * (display_scaling.scale_factor if hasattr(display_scaling, "scale_factor") else 1.0)
        t_thick = max(1, display_scaling.scale(2))
        x_pos = display_scaling.scale(35)
        y_top = display_scaling.scale(30)
        y_bot = max(y_top, h - display_scaling.scale(15))
        cv2.putText(img, f"{max_val:.0f}", (x_pos, y_top), cv2.FONT_HERSHEY_SIMPLEX, f_scale, text_color, t_thick)
        cv2.putText(img, f"{min_val:.0f}", (x_pos, y_bot), cv2.FONT_HERSHEY_SIMPLEX, f_scale, text_color, t_thick)
        return img

    def calc_fps(self) -> float:
        """Computes FPS (how fast frames arrive/render)."""
        self.fps_now = time.time()
        delta = self.fps_now - self.fps_last
        if delta > 0:
            self.fps = 1.0 / delta
            self.fps_last = self.fps_now
        return self.fps

    def is_crop_limit_enabled(self) -> bool:
        """Returns True if 'Crop limit' checkbox is checked."""
        if hasattr(self, "crop_limit_tag") and dpg.does_item_exist(self.crop_limit_tag):
            return bool(dpg.get_value(self.crop_limit_tag))
        return False

    def is_zero_as_alpha_enabled(self) -> bool:
        """Returns True if 'Zero as alpha' checkbox is checked."""
        if hasattr(self, "zero_as_alpha_tag") and dpg.does_item_exist(self.zero_as_alpha_tag):
            return bool(dpg.get_value(self.zero_as_alpha_tag))
        return getattr(self, "zero_as_alpha", False)

    def apply_zero_as_alpha(self, raw_frame: np.ndarray, processed_frame: np.ndarray) -> np.ndarray:
        """
        If 'Zero as alpha' is enabled, creates an RGBA/BGRA image where any pixel that was 0 in raw_frame
        has alpha set to 0 (fully transparent), and all other pixels have alpha set to 255.
        """
        if not self.is_zero_as_alpha_enabled() or raw_frame is None or processed_frame is None:
            return processed_frame

        if raw_frame.ndim == 2 or (raw_frame.ndim == 3 and raw_frame.shape[2] == 1):
            zero_mask = (raw_frame == 0)
            if zero_mask.ndim == 3:
                zero_mask = zero_mask[:, :, 0]
        elif raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
            zero_mask = (raw_frame[:, :, 0] == 0) & (raw_frame[:, :, 1] == 0) & (raw_frame[:, :, 2] == 0)
        elif raw_frame.ndim == 3 and raw_frame.shape[2] == 4:
            zero_mask = (raw_frame[:, :, 3] == 0) | (
                (raw_frame[:, :, 0] == 0) & (raw_frame[:, :, 1] == 0) & (raw_frame[:, :, 2] == 0)
            )
        else:
            zero_mask = (raw_frame == 0)

        h, w = processed_frame.shape[:2]
        if zero_mask.shape[:2] != (h, w):
            zero_mask = cv2.resize(zero_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        alpha = np.where(zero_mask, np.uint8(0), np.uint8(255))

        if processed_frame.ndim == 2 or (processed_frame.ndim == 3 and processed_frame.shape[2] == 1):
            bgra = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGRA)
            bgra[:, :, 3] = alpha
            return bgra
        elif processed_frame.shape[2] == 3:
            bgra = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            return bgra
        elif processed_frame.shape[2] == 4:
            res = processed_frame.copy()
            res[:, :, 3] = np.minimum(res[:, :, 3], alpha)
            return res

        return processed_frame

    def crop_to_alpha_limit(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops a 4-channel (RGBA/BGRA) frame to the minimal bounding rectangle where alpha > 0.
        Returns original frame if not 4 channels or if alpha mask is completely empty.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 4:
            return frame

        alpha = frame[:, :, 3]
        x, y, w, h = cv2.boundingRect(alpha)
        if w == 0 or h == 0:
            return frame

        return np.ascontiguousarray(frame[y:y + h, x:x + w])

    def _apply_transparency_theme(self) -> None:
        """Applies transparent plot background theme so transparency directly reveals the window background."""
        if not hasattr(self, "_plot_transparent_theme") or not dpg.does_item_exist(self._plot_transparent_theme):
            with dpg.theme() as self._plot_transparent_theme:
                with dpg.theme_component(dpg.mvPlot):
                    dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)

        if dpg.does_item_exist(self.plot_tag):
            dpg.bind_item_theme(self.plot_tag, self._plot_transparent_theme)

    def _on_toggle_rgba(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Retained for backward compatibility. Viewer operates permanently in RGBA."""
        self.use_rgba = True

    def convert_to_texture(self, frame: np.ndarray) -> np.ndarray:
        """Converts image to normalized float32 RGBA array for DearPyGui raw texture."""
        h, w = frame.shape[:2]
        expected_size = h * w * 4

        if self._texture_buffer is None or self._texture_buffer.size != expected_size:
            self._texture_buffer = np.zeros(expected_size, dtype=np.float32)

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        elif frame.shape[2] == 4:
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        elif frame.shape[2] == 3:
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            conv_frame = cv2.cvtColor(frame[:, :, :1], cv2.COLOR_GRAY2RGBA)

        if conv_frame.dtype != np.uint8:
            conv_frame = conv_frame.astype(np.uint8)

        flattened = conv_frame.ravel()

        if len(self._texture_buffer) == len(flattened):
            np.multiply(flattened, 1.0 / 255.0, out=self._texture_buffer, casting="unsafe")
        else:
            self._texture_buffer = flattened.astype(np.float32) * (1.0 / 255.0)

        return self._texture_buffer

    def init_viewer(self, width: int, height: int) -> None:
        """Initializes DearPyGui image display area (main thread safe)."""
        if threading.current_thread() is threading.main_thread():
            self._init_viewer_internal(width, height)
        else:
            automation_manager.post_to_main_thread(self._init_viewer_internal, width, height, wait=True)

    def _init_viewer_internal(self, width: int, height: int) -> None:
        with self._thread_lock:
            self._texture_initialized = False
            self.texture_width = width
            self.texture_height = height

            if dpg.does_item_exist(self.series_tag):
                try:
                    dpg.delete_item(self.series_tag)
                except Exception as e:
                    logger.warning(f"[{self.label}] Error deleting series {self.series_tag}: {e}")

            if dpg.does_item_exist(self.texture_tag):
                try:
                    dpg.delete_item(self.texture_tag)
                except Exception as e:
                    logger.warning(f"[{self.label}] Error deleting texture {self.texture_tag}: {e}")

            expected_size = height * width * 4
            if self._texture_buffer is None or self._texture_buffer.size != expected_size:
                self._texture_buffer = np.zeros(expected_size, dtype=np.float32)

            if self.texture_mode == "static":
                dpg.add_static_texture(
                    width=width,
                    height=height,
                    default_value=self._texture_buffer,
                    tag=self.texture_tag,
                    parent=self.texture_reg_tag,
                )
            else:
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=self._texture_buffer,
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.texture_tag,
                    parent=self.texture_reg_tag,
                )

            if not dpg.does_item_exist(self.plot_tag):
                plot_tag = dpg.add_plot(
                    tag=self.plot_tag,
                    no_menus=True,
                    no_title=True,
                    width=-1,
                    height=-1,
                    equal_aspects=self.preserve_aspect_ratio,
                    parent=self.winID,
                )
                dpg.add_plot_legend(parent=plot_tag)
                dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, tag=self.xaxis_tag, parent=plot_tag)
                dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, tag=self.yaxis_tag, parent=plot_tag)

                # ROI Overlay initialization
                rect_tag = f"roi_rect_{self.UUID}"
                text_tag = f"roi_text_{self.UUID}"
                dpg.draw_rectangle(pmin=(0, 0), pmax=(0, 0), color=(0, 255, 0, 255), thickness=2, tag=rect_tag, show=False, parent=plot_tag)
                dpg.add_plot_annotation(label="ROI", default_value=(0, 0), color=(0, 255, 0, 255), offset=(10, -10), tag=text_tag, show=False, parent=plot_tag)
                self._apply_transparency_theme()
            else:
                dpg.configure_item(self.plot_tag, equal_aspects=self.preserve_aspect_ratio)

            dpg.add_image_series(self.texture_tag, [0, 0], [width, height], tag=self.series_tag, parent=self.yaxis_tag)

            if dpg.does_item_exist(self.xaxis_tag) and dpg.does_item_exist(self.yaxis_tag):
                dpg.set_axis_limits_auto(self.xaxis_tag)
                dpg.set_axis_limits_auto(self.yaxis_tag)
                dpg.fit_axis_data(self.xaxis_tag)
                dpg.fit_axis_data(self.yaxis_tag)

            self._texture_initialized = True

    def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies negative or colormap to the 8-bit image."""
        if self.negative:
            frame = 255 - frame

        if frame.ndim == 3 and frame.shape[2] >= 3:
            return frame

        match self.palette:
            case "B&W":
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            case "Inferno":
                return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
            case "Jet":
                return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            case "HSV":
                return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
            case _:
                return frame

    def remap_and_annotate(self, frame: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
        """Remaps intensities and colorizes frame."""
        if dpg.get_value(self.auto_remap_tag):
            sample = frame[::5, ::5]
            min_val_8bit = int(np.percentile(sample, 1))
            max_val_8bit = int(np.max(sample))
            dpg.set_value(self.sup_tag, max_val_8bit)
            dpg.set_value(self.inf_tag, min_val_8bit)

            if min_val is None:
                min_val = min_val_8bit
            if max_val is None:
                max_val = max_val_8bit

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = self.remap_pixel_intensity(frame, dpg.get_value(self.inf_tag), dpg.get_value(self.sup_tag))

        frame = self.colorize_frame(frame)
        return frame

    def _update_single_persistent_roi(self, r_uuid: Union[int, str]) -> None:
        """Updates or creates a single persistent ROI on the plot (O(1))."""
        if r_uuid not in self.rois:
            return
        r_data = self.rois[r_uuid]
        r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
        t_tag = f"roi_text_{self.UUID}_{r_uuid}"

        x, y, w, h = r_data["rect"]
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        if not dpg.does_item_exist(r_tag):
            dpg.draw_rectangle(pmin=(x1, y1), pmax=(x2, y2), color=(0, 255, 0, 255), thickness=2, parent=self.plot_tag, tag=r_tag)
            dpg.add_plot_annotation(label=r_data["name"], default_value=(x2, y2), color=(0, 255, 0, 255), offset=(10, -10), parent=self.plot_tag, tag=t_tag)

        color = (0, 255, 0, 255) if r_data.get("selected", False) else (0, 100, 0, 255)
        fill_color = (0, 255, 0, 70) if r_data.get("highlighted", False) else (0, 0, 0, 0)
        show = dpg.get_value(self.roi_enable_tag)

        dpg.configure_item(r_tag, pmin=(x1, y1), pmax=(x2, y2), color=color, fill=fill_color, show=show)
        dpg.configure_item(t_tag, default_value=(x2, y2), label=r_data["name"], color=color, show=show)

    def _update_persistent_rois(self) -> None:
        """Updates all persistent ROIs (called on clear/toggle/bulk changes)."""
        for r_uuid in list(self.rois.keys()):
            self._update_single_persistent_roi(r_uuid)

    def _update_hover_roi(self) -> None:
        """Updates ONLY preview/hover ROI rectangle and stats on mouse move (O(1))."""
        rect_tag = f"roi_rect_{self.UUID}"
        text_tag = f"roi_text_{self.UUID}"

        if not dpg.does_item_exist(rect_tag) or not dpg.does_item_exist(text_tag):
            return

        if not dpg.get_value(self.roi_enable_tag) or not self.is_hovering:
            dpg.configure_item(rect_tag, show=False)
            dpg.configure_item(text_tag, show=False)
            return

        w = dpg.get_value(self.roi_w_tag)
        h = dpg.get_value(self.roi_h_tag)

        x = self.mouse_x - w // 2
        y = self.mouse_y - h // 2

        img_w, img_h = self.texture_width, self.texture_height
        if img_w == 0 or img_h == 0:
            return

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 > x1 and y2 > y1:
            dpg.configure_item(rect_tag, pmin=(x1, y1), pmax=(x2, y2), show=True)
            dpg.configure_item(text_tag, default_value=(x2, y2), show=True)
        else:
            dpg.configure_item(rect_tag, show=False)
            dpg.configure_item(text_tag, show=False)
            return

        if self.last_image is not None:
            try:
                y1_arr = max(0, min(img_h - y2, img_h))
                y2_arr = max(0, min(img_h - y1, img_h))
                roi = self.last_image[y1_arr:y2_arr, x1:x2]

                if roi.size > 0:
                    thresh_pct = dpg.get_value(self.roi_thresh_tag)
                    if roi.ndim == 3 and roi.shape[2] >= 3:
                        mask_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
                    else:
                        mask_base = roi

                    if thresh_pct >= 100:
                        mask = None
                    else:
                        p_thresh = np.percentile(mask_base, 100 - thresh_pct)
                        mask = mask_base >= p_thresh

                    if roi.ndim == 3 and roi.shape[2] >= 3:
                        valid_pixels = roi[mask] if mask is not None else roi.reshape(-1, roi.shape[2])
                        if roi.shape[2] == 3:
                            valid_pixels_rgb = valid_pixels[:, ::-1]
                        else:
                            valid_pixels_rgb = valid_pixels[:, :3][:, ::-1]
                        mean_val = np.mean(valid_pixels_rgb, axis=0) if valid_pixels.size > 0 else [0.0, 0.0, 0.0]
                        dpg.set_value(self.roi_result_tag, f"R:{mean_val[0]:.1f} G:{mean_val[1]:.1f} B:{mean_val[2]:.1f}")
                        dpg.configure_item(text_tag, label=f"RGB: {mean_val[0]:.0f},{mean_val[1]:.0f},{mean_val[2]:.0f}")
                    else:
                        valid_pixels = mask_base[mask] if mask is not None else mask_base
                        mean_val = float(np.mean(valid_pixels)) if valid_pixels.size > 0 else 0.0
                        std_val = float(np.std(valid_pixels)) if valid_pixels.size > 0 else 0.0

                        dpg.set_value(self.roi_result_tag, f"{mean_val:.2f} , {std_val:.2f}")
                        dpg.configure_item(text_tag, label=f"{mean_val:.1f} , {std_val:.1f}")
            except Exception:
                pass

    def _update_roi_overlay(self) -> None:
        """Updates both persistent ROIs and hover preview (called on UI callbacks or mode changes)."""
        self._update_persistent_rois()
        self._update_hover_roi()

    def _handle_roi_input(self, sample: Dict[str, Any]) -> None:
        """Handles updates from ROI Manager."""
        if sample.get("clear"):
            for r_uuid in list(self.rois.keys()):
                r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
                t_tag = f"roi_text_{self.UUID}_{r_uuid}"
                if dpg.does_item_exist(r_tag):
                    dpg.delete_item(r_tag)
                if dpg.does_item_exist(t_tag):
                    dpg.delete_item(t_tag)
            self.rois.clear()
            self._update_hover_roi()
            return

        uuid = sample.get("uuid")
        if not uuid:
            return

        action = sample.get("action")

        if action == "rename":
            if uuid in self.rois:
                self.rois[uuid]["name"] = sample["name"]
                self._update_single_persistent_roi(uuid)
        elif action == "select":
            if uuid in self.rois:
                self.rois[uuid]["selected"] = True
                self.last_selected_roi_uuid = uuid
                self._update_single_persistent_roi(uuid)
                if self.last_image is not None:
                    self._emit_cropped_frame(self.last_image, force_emit=True)
        elif action == "unselect":
            if uuid in self.rois:
                self.rois[uuid]["selected"] = False
                self._update_single_persistent_roi(uuid)
        elif action == "delete":
            if uuid in self.rois:
                del self.rois[uuid]
                r_tag = f"roi_rect_{self.UUID}_{uuid}"
                t_tag = f"roi_text_{self.UUID}_{uuid}"
                if dpg.does_item_exist(r_tag):
                    dpg.delete_item(r_tag)
                if dpg.does_item_exist(t_tag):
                    dpg.delete_item(t_tag)
        elif action == "highlight":
            if uuid in self.rois:
                self.rois[uuid]["highlighted"] = True
                self._update_single_persistent_roi(uuid)
        elif action == "unhighlight":
            if uuid in self.rois:
                self.rois[uuid]["highlighted"] = False
                self._update_single_persistent_roi(uuid)
        elif action == "create":
            if uuid not in self.rois:
                self.rois[uuid] = {
                    "uuid": uuid,
                    "name": sample.get("name", "ROI"),
                    "rect": sample.get("rect", [0, 0, 100, 100]),
                    "selected": True,
                    "threshold": sample.get("threshold", 100),
                }
                self.last_selected_roi_uuid = uuid
                self._update_single_persistent_roi(uuid)

    def process_roi(self, frame: np.ndarray, processed_frame: np.ndarray) -> np.ndarray:
        """Calculates ROI stats using raw frame and annotates processed_frame."""
        if not dpg.get_value(self.roi_enable_tag):
            return processed_frame

        show_mask = dpg.get_value(self.roi_show_mask_tag)
        if not show_mask:
            return processed_frame

        x = self.mouse_x - dpg.get_value(self.roi_w_tag) // 2
        y = self.mouse_y - dpg.get_value(self.roi_h_tag) // 2
        w = dpg.get_value(self.roi_w_tag)
        h = dpg.get_value(self.roi_h_tag)
        thresh_pct = dpg.get_value(self.roi_thresh_tag)

        img_h, img_w = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        y1_arr = max(0, min(img_h - y2, img_h))
        y2_arr = max(0, min(img_h - y1, img_h))

        roi = frame[y1_arr:y2_arr, x1:x2]
        if roi.size > 0:
            if roi.ndim == 3 and roi.shape[2] >= 3:
                mask_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            else:
                mask_base = roi

            p_thresh = np.percentile(mask_base, 100 - thresh_pct)
            mask = mask_base >= p_thresh

            if show_mask:
                vis_roi = processed_frame[y1_arr:y2_arr, x1:x2]
                color = [0, 0, 255]
                if vis_roi.ndim == 3 and vis_roi.shape[2] == 4:
                    color = [0, 0, 255, 255]
                vis_roi[mask] = color

        return processed_frame

    def process_lowdepth(self, frame: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Processes image before rendering and adds overlay if mask is provided."""
        frame = self.remap_and_annotate(frame, min_val, max_val)
        if mask is not None:
            frame[mask] = [0, 0, 255]
        return frame

    def _emit_cropped_frame(self, frame: np.ndarray, force_emit: bool = False) -> None:
        if not self.connections.get("Cropped Frame"):
            return

        if not force_emit and not dpg.get_value(self.roi_live_crop_tag):
            return

        img_h, img_w = frame.shape[:2]
        active_rect = None
        last_uuid = getattr(self, "last_selected_roi_uuid", None)

        if last_uuid and last_uuid in self.rois and self.rois[last_uuid].get("selected", False):
            active_rect = self.rois[last_uuid]["rect"]
        else:
            for r_uuid, r_data in reversed(list(self.rois.items())):
                if r_data.get("selected", False):
                    active_rect = r_data["rect"]
                    self.last_selected_roi_uuid = r_uuid
                    break

        if active_rect is None and dpg.get_value(self.roi_enable_tag) and self.is_hovering:
            w = dpg.get_value(self.roi_w_tag)
            h = dpg.get_value(self.roi_h_tag)
            x = self.mouse_x - w // 2
            y = self.mouse_y - h // 2
            active_rect = [x, y, w, h]

        if active_rect:
            x, y, w, h = active_rect
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            y1_arr = max(0, min(img_h - y2, img_h))
            y2_arr = max(0, min(img_h - y1, img_h))

            crop = frame[y1_arr:y2_arr, x1:x2]
            if crop.size > 0:
                for module in self.connections.get("Cropped Frame", []):
                    try:
                        module.input_cb(data=crop, data_type=IOTypes.FRAME)
                    except Exception as e:
                        logger.debug(f"[{self.label}] Cropped frame emission failed: {e}")

    def _emit_lowdepth_frame(self, frame: np.ndarray, force_emit: bool = False) -> None:
        if not self.connections.get("Lowdepth Frame"):
            return

        minval, maxval = self._get_current_scale_limits()
        lowdepth = self.convert_to_lowdepth(frame)
        image = self.remap_and_annotate(lowdepth, minval, maxval)

        for module in self.connections.get("Lowdepth Frame", []):
            try:
                module.input_cb(data=image, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.debug(f"[{self.label}] Lowdepth frame emission failed: {e}")

    def update_image(self, frame: np.ndarray) -> None:
        """Receives a new frame, emits dependent frames, and signals render thread (non-blocking)."""
        if frame is None:
            return

        self.calc_fps()
        self._emit_cropped_frame(frame)
        self._emit_lowdepth_frame(frame)

        self.last_image = frame
        with self.frame_lock:
            self.latest_frame = frame
        self.update_event.set()

    def _render_loop(self) -> None:
        """Internal render loop that displays the latest available frame without blocking input."""
        while self.running:
            self.update_event.wait(timeout=0.1)
            self.update_event.clear()

            frame_to_render = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_render = self.latest_frame
                    self.latest_frame = None

            if frame_to_render is not None:
                try:
                    self._render_frame(frame_to_render)
                except Exception as e:
                    logger.warning(f"[{self.label}] Error in render loop: {e}")

    def update_image_callback(self, *args: Any, **kwargs: Any) -> None:
        """Callback to trigger update from UI."""
        if self.last_image is not None:
            with self.frame_lock:
                self.latest_frame = self.last_image
            self.update_event.set()

        self._update_roi_overlay()

    def _prepare_processed_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute the processed visualization frame along with its (minval, maxval) scale limits."""
        show_scale = bool(dpg.get_value(self.show_scale_tag))
        need_roi_mask = bool(dpg.get_value(self.roi_enable_tag) and dpg.get_value(self.roi_show_mask_tag))

        if frame.ndim == 3 and frame.shape[2] in (3, 4) and frame.dtype == np.uint8:
            # Copy only if an in-place operation (ROI mask or scale bar) will mutate the image
            processed = frame.copy() if (need_roi_mask or show_scale) else frame
            is_owned = need_roi_mask or show_scale
            if show_scale:
                minval, maxval = self.get_minmax_values(frame)
            else:
                minval, maxval = 0.0, 255.0
            if need_roi_mask:
                processed = self.process_roi(frame, processed)
        else:
            lowdepth = self.convert_to_lowdepth(frame)
            if lowdepth is frame and dpg.get_value(self.auto_remap_tag):
                sample = lowdepth[::5, ::5]
                min_8 = int(np.percentile(sample, 1))
                max_8 = int(np.max(sample))
                dpg.set_value(self.sup_tag, max_8)
                dpg.set_value(self.inf_tag, min_8)
                minval, maxval = float(min_8), float(max_8)
                if lowdepth.ndim == 2 or (lowdepth.ndim == 3 and lowdepth.shape[2] == 1):
                    remapped = self.remap_pixel_intensity(lowdepth, min_8, max_8)
                else:
                    remapped = lowdepth
                processed = self.colorize_frame(remapped)
            else:
                minval, maxval = self._get_current_scale_limits()
                processed = self.remap_and_annotate(lowdepth, minval, maxval)
            is_owned = processed is not frame
            if need_roi_mask:
                if not is_owned:
                    processed = processed.copy()
                    is_owned = True
                processed = self.process_roi(frame, processed)

        if self.is_zero_as_alpha_enabled():
            next_proc = self.apply_zero_as_alpha(frame, processed)
            if next_proc is not processed:
                is_owned = True
            processed = next_proc

        if self.is_crop_limit_enabled():
            next_proc = self.crop_to_alpha_limit(processed)
            if next_proc is not processed:
                is_owned = True
            processed = next_proc

        if show_scale:
            processed = self.add_scale_and_annotations(processed, minval, maxval, in_place=is_owned)

        return processed, float(minval), float(maxval)

    def _render_frame(self, frame: np.ndarray) -> None:
        """Processes the frame and updates the GPU texture."""
        if frame is None:
            return
        with self.lock:
            try:
                processed, _, _ = self._prepare_processed_frame(frame)

                h, w = processed.shape[:2]
                texture_data = self.convert_to_texture(processed)

                if self.texture_mode == "static":
                    dim_changed = (h != self.texture_height or w != self.texture_width or not self._texture_initialized)
                    self.texture_width = w
                    self.texture_height = h

                    def _recreate_static() -> None:
                        with self._thread_lock:
                            if not self.running:
                                return
                            if dpg.does_item_exist(self.series_tag):
                                try:
                                    dpg.delete_item(self.series_tag)
                                except Exception:
                                    pass
                            if dpg.does_item_exist(self.texture_tag):
                                try:
                                    dpg.delete_item(self.texture_tag)
                                except Exception:
                                    pass
                            dpg.add_static_texture(
                                width=w,
                                height=h,
                                default_value=texture_data,
                                tag=self.texture_tag,
                                parent=self.texture_reg_tag,
                            )
                            dpg.add_image_series(
                                self.texture_tag,
                                [0, 0],
                                [w, h],
                                tag=self.series_tag,
                                parent=self.yaxis_tag,
                            )
                            if dim_changed and dpg.does_item_exist(self.xaxis_tag) and dpg.does_item_exist(self.yaxis_tag):
                                dpg.set_axis_limits_auto(self.xaxis_tag)
                                dpg.set_axis_limits_auto(self.yaxis_tag)
                                dpg.fit_axis_data(self.xaxis_tag)
                                dpg.fit_axis_data(self.yaxis_tag)
                            self._texture_initialized = True

                    if threading.current_thread() is threading.main_thread():
                        _recreate_static()
                    else:
                        automation_manager.post_to_main_thread(_recreate_static, wait=True)

                else:
                    if not self._texture_initialized or h != self.texture_height or w != self.texture_width:
                        self.init_viewer(w, h)

                    with self._thread_lock:
                        if self._texture_initialized and dpg.does_item_exist(self.texture_tag):
                            dpg.set_value(self.texture_tag, texture_data)

                # FPS Display Update
                if self._frame_count % self._fps_update_interval == 0:
                    if dpg.does_item_exist(self.fps_tag):
                        dpg.set_value(self.fps_tag, str(int(self.fps)))
                self._frame_count += 1

            except Exception as e:
                logger.error(f"[{self.label}] Error in render frame: {e}")

    def set_texture_mode(self, mode: str) -> None:
        """Switches texture rendering mode ('raw' or 'static')."""
        mode_clean = mode.lower()
        if mode_clean in ("raw", "static") and mode_clean != self.texture_mode:
            with self._thread_lock:
                self.texture_mode = mode_clean
                self._texture_initialized = False
            self.update_image_callback()

    def set_preserve_aspect_ratio(self, preserve: bool) -> None:
        """Dynamically toggle aspect ratio preservation / stretching."""
        self.preserve_aspect_ratio = bool(preserve)

        def _apply() -> None:
            if dpg.does_item_exist(self.plot_tag):
                dpg.configure_item(self.plot_tag, equal_aspects=self.preserve_aspect_ratio)
                if dpg.does_item_exist(self.xaxis_tag) and dpg.does_item_exist(self.yaxis_tag):
                    dpg.set_axis_limits_auto(self.xaxis_tag)
                    dpg.set_axis_limits_auto(self.yaxis_tag)
                    dpg.fit_axis_data(self.xaxis_tag)
                    dpg.fit_axis_data(self.yaxis_tag)

        if threading.current_thread() is threading.main_thread():
            _apply()
        else:
            automation_manager.post_to_main_thread(_apply)

    def set_mode(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handles UI mode changes (negative toggle or palette switch)."""
        if app_data in ["Normal", "Negative"]:
            self.negative = (app_data == "Negative")
        else:
            self.palette = str(app_data)

        if self.last_image is not None:
            self._render_frame(self.last_image)

    def set_depth(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handles image depth changes."""
        self.image_depth = str(app_data)
        if self.last_image is not None:
            self._render_frame(self.last_image)
        self._update_roi_overlay()

    def copy_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Copies the processed image to clipboard using the custom injector."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning(f"[{self.label}] No image available to copy to clipboard.")
                    return

                processed, _, _ = self._prepare_processed_frame(self.last_image)
                clipboardinjector.send_image(processed)
            except Exception as e:
                logger.error(f"[{self.label}] Error copying image to clipboard: {e}")

    def save_image(self, *args: Any, **kwargs: Any) -> None:
        """Saves both raw and processed images to PNG with indexed names."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning(f"[{self.label}] No image available to save.")
                    return

                raw = self.last_image.copy()

                if raw.ndim == 3 and raw.shape[2] in [3, 4] and raw.dtype == np.uint8:
                    processed = raw.copy()
                    minval, maxval = self.get_minmax_values(raw)
                    processed = self.process_roi(raw, processed)
                else:
                    minval, maxval = self._get_current_scale_limits()
                    lowdepth = self.convert_to_lowdepth(raw)
                    processed = self.remap_and_annotate(lowdepth, minval, maxval)
                    processed = self.process_roi(raw, processed)

                if self.is_zero_as_alpha_enabled():
                    processed = self.apply_zero_as_alpha(raw, processed)

                if self.is_crop_limit_enabled():
                    if processed.ndim == 3 and processed.shape[2] == 4:
                        alpha = processed[:, :, 3]
                        x_c, y_c, w_c, h_c = cv2.boundingRect(alpha)
                        if w_c > 0 and h_c > 0:
                            processed = np.ascontiguousarray(processed[y_c:y_c + h_c, x_c:x_c + w_c])
                            if raw.shape[:2] == alpha.shape[:2]:
                                raw = np.ascontiguousarray(raw[y_c:y_c + h_c, x_c:x_c + w_c])
                    else:
                        processed = self.crop_to_alpha_limit(processed)

                if dpg.get_value(self.show_scale_tag):
                    processed = self.add_scale_and_annotations(processed, minval, maxval, in_place=True)

                name = dpg.get_value(self.name_tag)

                if self.export_path:
                    os.makedirs(self.export_path, exist_ok=True)
                    raw_path = os.path.join(self.export_path, f"{name}_RAW_{self.imgcount}.png")
                    processed_path = os.path.join(self.export_path, f"{name}_Processed_{self.imgcount}.png")
                else:
                    raw_path = f"{name}_RAW_{self.imgcount}.png"
                    processed_path = f"{name}_Processed_{self.imgcount}.png"

                cv2.imwrite(raw_path, raw)
                cv2.imwrite(processed_path, processed)

                dpg.set_value(self.count_tag, f"count : {self.imgcount}")
                self.imgcount += 1
            except Exception as e:
                logger.error(f"[{self.label}] Error saving image: {e}")

    def update_save_tooltip(self) -> None:
        """Updates the tooltip for the Save button based on the current export path."""
        if hasattr(self, "save_tooltip_text_tag") and dpg.does_item_exist(self.save_tooltip_text_tag):
            tip_text = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
            dpg.set_value(self.save_tooltip_text_tag, tip_text)

    def reset_counter(self, *args: Any, **kwargs: Any) -> None:
        """Resets the saved image counter to zero."""
        self.imgcount = 0
        dpg.set_value(self.count_tag, f"count: {self.imgcount}")

    def _update_categories_count(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self.categories_count = int(app_data)

    def send_palette_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Computes and sends the current category palette to the clipboard in palette.json format."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning(f"[{self.label}] No image loaded to compute palette.")
                    return

                num_categories = dpg.get_value(self.cat_count_tag)
                if num_categories < 1:
                    logger.warning(f"[{self.label}] Number of categories must be at least 1.")
                    return

                minval, maxval = self._get_current_scale_limits()
                range_diff = maxval - minval
                if range_diff <= 0:
                    range_diff = 1.0

                palette_dict = {}
                for i in range(num_categories):
                    bin_low = minval + i * range_diff / num_categories
                    bin_high = minval + (i + 1) * range_diff / num_categories
                    cat_name = f"{bin_low/1000.0:.3f}-{bin_high/1000.0:.3f}"

                    val_center = int((i + 0.5) * 255 / num_categories)
                    pixel_img = np.array([[val_center]], dtype=np.uint8)
                    colorized = self.colorize_frame(pixel_img)
                    if colorized.ndim == 3 and colorized.shape[2] >= 3:
                        bgr = colorized[0, 0]
                        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
                    else:
                        val = int(colorized[0, 0] if colorized.ndim >= 2 else val_center)
                        rgb = [val, val, val]
                    palette_dict[cat_name] = rgb

                import json
                palette_json = json.dumps(palette_dict, indent=4)
                clipboardinjector.send_text(palette_json)
                logger.success(f"[{self.label}] Palette with {num_categories} categories copied to clipboard.")
            except Exception as e:
                logger.error(f"[{self.label}] Error sending palette to clipboard: {e}")

    def send_result_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Computes the category distribution and sends it to the clipboard in TSV format."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning(f"[{self.label}] No image loaded to compute proportions.")
                    return

                num_categories = dpg.get_value(self.cat_count_tag)
                if num_categories < 1:
                    logger.warning(f"[{self.label}] Number of categories must be at least 1.")
                    return

                target_img = self.last_image.copy()
                if self.is_crop_limit_enabled():
                    target_img = self.crop_to_alpha_limit(target_img)

                if target_img.ndim == 3 and target_img.shape[2] >= 3:
                    raw_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY if target_img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
                else:
                    raw_gray = target_img

                minval, maxval = self._get_current_scale_limits()

                img_h, img_w = raw_gray.shape[:2]
                active_rect = None
                last_uuid = getattr(self, "last_selected_roi_uuid", None)
                thresh_pct = 100

                if last_uuid and last_uuid in self.rois and self.rois[last_uuid].get("selected", False):
                    active_rect = self.rois[last_uuid]["rect"]
                    thresh_pct = self.rois[last_uuid].get("threshold", 100)
                else:
                    for r_uuid, r_data in reversed(list(self.rois.items())):
                        if r_data.get("selected", False):
                            active_rect = r_data["rect"]
                            self.last_selected_roi_uuid = r_uuid
                            thresh_pct = r_data.get("threshold", 100)
                            break

                if active_rect:
                    x, y, w, h = active_rect
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(img_w, x + w)
                    y2 = min(img_h, y + h)

                    y1_arr = max(0, min(img_h - y2, img_h))
                    y2_arr = max(0, min(img_h - y1, img_h))

                    roi_raw = raw_gray[y1_arr:y2_arr, x1:x2]

                    if roi_raw.size > 0:
                        p_thresh = np.percentile(roi_raw, 100 - thresh_pct)
                        mask = (roi_raw >= p_thresh) & (roi_raw > 0)
                        target_raw = roi_raw[mask]
                    else:
                        target_raw = np.array([], dtype=raw_gray.dtype)
                else:
                    mask = raw_gray > 0
                    target_raw = raw_gray[mask]

                range_diff = maxval - minval
                if range_diff <= 0:
                    range_diff = 1.0

                if target_raw.size > 0:
                    normalized = (target_raw.astype(np.float32) - minval) / range_diff
                    np.clip(normalized, 0.0, 1.0, out=normalized)

                    indices = (normalized * num_categories).astype(np.int32)
                    indices = np.clip(indices, 0, num_categories - 1)

                    bin_counts = np.bincount(indices, minlength=num_categories)
                else:
                    bin_counts = np.zeros(num_categories, dtype=np.int32)

                total_pixels = np.sum(bin_counts)
                lines = ["Label\tCount\tPercentage (%)"]
                for i in range(num_categories):
                    bin_low = minval + i * range_diff / num_categories
                    bin_high = minval + (i + 1) * range_diff / num_categories
                    cat_name = f"{bin_low/1000.0:.3f}-{bin_high/1000.0:.3f}"
                    count = int(bin_counts[i])
                    percent = (100.0 * count / total_pixels) if total_pixels > 0 else 0.0
                    lines.append(f"{cat_name}\t{count}\t{percent:.2f}")

                tsv_data = "\n".join(lines)
                clipboardinjector.send_text(tsv_data)
                logger.success(f"[{self.label}] Category distribution copied to clipboard in TSV format.")
            except Exception as e:
                logger.error(f"[{self.label}] Error sending result to clipboard: {e}")

    def is_ready(self) -> bool:
        """Indicate whether the image viewer is ready to accept and render new frames."""
        return self.running

    def on_close(self) -> None:
        """Release resources, background render loop, and GPU texture buffers."""
        self.running = False
        self.update_event.set()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive() and threading.current_thread() != self.worker_thread:
            self.worker_thread.join(timeout=0.2)
        if hasattr(self, "handler_reg_tag") and dpg.does_item_exist(self.handler_reg_tag):
            try:
                dpg.delete_item(self.handler_reg_tag)
            except Exception:
                pass
        if hasattr(self, "_plot_transparent_theme") and dpg.does_item_exist(self._plot_transparent_theme):
            try:
                dpg.delete_item(self._plot_transparent_theme)
            except Exception:
                pass
        with self._thread_lock:
            self._texture_initialized = False
            if dpg.does_item_exist(self.series_tag):
                try:
                    dpg.delete_item(self.series_tag)
                except Exception:
                    pass
            if dpg.does_item_exist(self.texture_tag):
                try:
                    dpg.delete_item(self.texture_tag)
                except Exception:
                    pass
            if dpg.does_item_exist(self.texture_reg_tag):
                try:
                    dpg.delete_item(self.texture_reg_tag)
                except Exception:
                    pass
            self._texture_buffer = None
        super().on_close()

    def close(self) -> None:
        """Close viewer window and unregister."""
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()


ImageViewerWin = Image_viewer_win
EXPORTED_CLASS = Image_viewer_win
EXPORTED_NAME = "Image Viewer"
