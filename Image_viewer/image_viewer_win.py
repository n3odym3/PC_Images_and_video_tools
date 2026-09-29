"""
High-Performance Image Viewer Window Module.

Provides a DearPyGui-based modular image display canvas supporting GPU texture
streaming, interactive ROI analysis, colormapping, intensity rescaling, transparent
rendering, and image export pipelines.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
import numpy as np
import threading
import time
import cv2
import os
from loguru import logger

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from modules.image_viewer.clipboard_injector import clipboardinjector
from core.input_output_types import IOTypes
from core.config_manager import config

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

    def __init__(self,
                label="Image Viewer",
                win_width: int = display_scaling.scale(1280),
                win_height: int = display_scaling.scale(900),
                pos=(100, 100),
                uuid=None,
                outputs=None,
                visible=True,
                colorize=True,
                intensity_rescaling=True,
                fps_counter=True,
                image_export=True,
                image_depth="12bit",
                show_scale=True,
                categories_count=5,
                use_rgba=True,
                zero_as_alpha=False,
                export_path=None,
                **kwargs):

        super().__init__(label=label, pos=pos, win_width=win_width, win_height=win_height,
            uuid=uuid, outputs=outputs or [], visible=visible, **kwargs)

        self.use_rgba = True

        # FPS tracking - separate input and display
        # FPS tracking
        self.fps = 0
        self.fps_now = time.time()
        self.fps_last = self.fps_now

        # Lock for threaded updates
        self.lock = threading.Lock()

        # Last image reference
        self.last_image = None

        # Display options
        self.negative = False
        self.palette = "Inferno"
        self.imgcount = 0
        
        # Performance optimization: frame counter for FPS display
        self._frame_count = 0
        self._fps_update_interval = 10  # Update FPS display every N frames

        # Texture mode configuration from config.json (fallback: "raw")
        viewer_cfg = config.get("Viewers", {}).get("image_viewer", {})
        default_mode = viewer_cfg if isinstance(viewer_cfg, str) else viewer_cfg.get("texture_mode", "raw")
        self.texture_mode: str = kwargs.get("texture_mode", default_mode).lower()
        if self.texture_mode not in ("raw", "static"):
            self.texture_mode = "raw"

        # Texture state
        self.texture_width = 0
        self.texture_height = 0
        self._texture_initialized = False
        self._texture_buffer = None
        self._thread_lock = threading.Lock()

        # DPG element tags
        self.texture_tag = f"viewer_texture_{self.UUID}"
        self.series_tag = f"viewer_series_{self.UUID}"
        self.plot_tag = f"viewer_plot_{self.UUID}"
        self.xaxis_tag = f"viewer_xaxis_{self.UUID}"
        self.yaxis_tag = f"viewer_yaxis_{self.UUID}"
        self.fps_tag = f"viewer_fps_{self.UUID}"
        self.name_tag = f"viewer_name_{self.UUID}"
        self.count_tag = f"viewer_count_{self.UUID}"
        self.save_btn = f"viewer_save_{self.UUID}"
        self.auto_remap_tag = f"viewer_autoremap_{self.UUID}"
        self.show_scale_tag = f"viewer_showscale_{self.UUID}"
        self.inf_tag = f"viewer_inf_{self.UUID}"
        self.sup_tag = f"viewer_sup_{self.UUID}"
        self.texture_reg_tag = f"viewer_tex_reg_{self.UUID}"
        self.crop_limit_tag = f"viewer_crop_limit_{self.UUID}"
        self.zero_as_alpha_tag = f"viewer_zero_alpha_{self.UUID}"
        self.zero_as_alpha = kwargs.get("zero_as_alpha", zero_as_alpha)
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
        self.fps_group_tag = f"viewer_fps_group_{self.UUID}"
        self.roi_thresh_group_tag = f"viewer_roi_thresh_group_{self.UUID}"
        self.palette_combo_tag = f"viewer_palette_combo_{self.UUID}"
        self.depth_combo_tag = f"viewer_depth_combo_{self.UUID}"
        
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

        # Config flags
        self.colorize = colorize
        self.intensity_rescaling = intensity_rescaling
        self.fps_counter = fps_counter
        self.image_export = image_export
        self.image_depth = image_depth
        self.show_scale = show_scale
        self.export_path = export_path
        self.categories_count = 5

        # ROI Storage
        self.rois = {} # {uuid: {rect: [x,y,w,h], name: str, selected: bool, ...}}
        self.last_selected_roi_uuid = None
        self.roi_index_counter = 1

        # Persistence and IO setup
        self._persistent_fields = [
            "label", "colorize", "intensity_rescaling",
            "fps_counter", "image_export", "image_depth", "show_scale", "export_path", "categories_count", "zero_as_alpha", "texture_mode"
        ]
        self.accepted_input_types = [
            IOTypes.FRAME, IOTypes.FILE_PATH, IOTypes.FRAME12, IOTypes.FRAME16,
            IOTypes.ROI_SAMPLE, IOTypes.CMD_DICT, IOTypes.FOLDER_PATH,
            IOTypes.FRAME_MASK_PAIR, IOTypes.MASK
        ]
        self.output_types = [IOTypes.FRAME, IOTypes.ROI_SAMPLE, IOTypes.FRAME, IOTypes.FRAME]

        self.outputs = {
            "Frame": IOTypes.FRAME,
            "ROI Data": IOTypes.ROI_SAMPLE,
            "Cropped Frame": IOTypes.FRAME,
            "Lowdepth Frame": IOTypes.FRAME
        }
        self.connections = {k: [] for k in self.outputs}
        # Rendering State
        self.running = True

        # Threading & Latest Frame approach (no blocking, 600+ FPS)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.update_event = threading.Event()

        self._build_interface(win_width, win_height, pos)
        self.update_permission()

        # Start Render Thread
        self.worker_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.worker_thread.start()

    def update_permission(self):
        """
        Adapt UI based on app_state.mode.
        In 'user' mode: hide normal/negative combo, show selection, live crop, clear, threshold, export group.
        """
        from core.app_state import app_state
        mode = app_state.mode
        
        # Elements to toggle
        elements = [
            self.mode_combo_tag,
            self.roi_show_mask_tag,
            self.roi_live_crop_tag,
            self.btn_clear_rois_tag,
            self.roi_thresh_group_tag,
            self.export_group_tag,
            self.show_scale_tag,
            self.fps_group_tag,
            self.palette_combo_tag,
            self.depth_combo_tag,
            self.roi_enable_tag,
            self.cat_group_tag
        ]
        
        show = (mode != "user")
        
        for tag in elements:
            if dpg.does_item_exist(tag):
                if show:
                    dpg.show_item(tag)
                else:
                    dpg.hide_item(tag)

        is_user = (mode == "user")
        dpg.configure_item(self.winID, no_close=is_user)

    def _build_interface(self, win_width, win_height, pos):
        """Creates the DearPyGui layout."""
        with dpg.window(label=self.label, width=win_width, height=win_height,
            pos=pos, tag=self.winID, show=self.visible):

            with dpg.group(horizontal=True):
                if self.colorize:
                    dpg.add_combo(("Normal", "Negative"), tag=self.mode_combo_tag, width=display_scaling.scale(120), default_value="Normal", callback=self.set_mode)
                    dpg.add_combo(("B&W", "Inferno", "Jet", "HSV"), tag=self.palette_combo_tag, width=display_scaling.scale(120), default_value=self.palette, callback=self.set_mode)
                    dpg.add_combo(("8bit", "12bit", "16bit_norm", "16bit_abs"), tag=self.depth_combo_tag, width=display_scaling.scale(120), default_value=self.image_depth, callback=self.set_depth)

                if self.intensity_rescaling:
                    dpg.add_text("Auto:")
                    dpg.add_checkbox(default_value=True, tag=self.auto_remap_tag, callback=self.update_image_callback)
                    dpg.add_text("Scale:")
                    dpg.add_checkbox(default_value=self.show_scale, tag=self.show_scale_tag, callback=self.update_image_callback)
                    dpg.add_text("Inf:")
                    dpg.add_drag_int(default_value=0, min_value=0, max_value=255, width=display_scaling.scale(70), tag=self.inf_tag, callback=self.update_image_callback)
                    dpg.add_text("Sup:")
                    dpg.add_drag_int(default_value=255, min_value=0, max_value=255, width=display_scaling.scale(70), tag=self.sup_tag, callback=self.update_image_callback)

                if self.fps_counter:
                    with dpg.group(horizontal=True, tag=self.fps_group_tag):
                        dpg.add_text("| FPS:")
                        dpg.add_text("0", tag=self.fps_tag)

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Enable ROI", default_value=True, tag=self.roi_enable_tag, callback=self.update_image_callback)
                dpg.add_checkbox(label="Show Selection", tag=self.roi_show_mask_tag, callback=self.update_image_callback)
                dpg.add_checkbox(label="Live Crop", tag=self.roi_live_crop_tag, default_value=False)
                dpg.add_button(label="Clear", tag=self.btn_clear_rois_tag, callback=self.clear_rois)
            
                dpg.add_text("W:")
                dpg.add_drag_int(width=display_scaling.scale(60), tag=self.roi_w_tag, default_value=25, min_value=1, callback=self.update_image_callback)
                dpg.add_text("H:")
                dpg.add_drag_int(width=display_scaling.scale(60), tag=self.roi_h_tag, default_value=25, min_value=1, callback=self.update_image_callback)

                with dpg.group(horizontal=True, tag=self.roi_thresh_group_tag):
                    dpg.add_text("Threshold %:")
                    dpg.add_drag_int(width=display_scaling.scale(100), tag=self.roi_thresh_tag, default_value=100, min_value=1, max_value=100, callback=self.update_image_callback)
                dpg.add_text("Mean Value:", color=(255, 100, 100))
                dpg.add_text("0.0", tag=self.roi_result_tag)

            if self.image_export:
                with dpg.group(horizontal=True, tag=self.export_group_tag):
                    dpg.add_button(label="Clipboard", callback=self.copy_to_clipboard)
                    dpg.add_input_text(default_value="Image", width=display_scaling.scale(150), tag=self.name_tag, callback=self.reset_counter)
                    dpg.add_text("count: 0", tag=self.count_tag)
                    dpg.add_button(label="Save", tag=self.save_btn, callback=self.save_image)
                    
                    self.save_tooltip_text_tag = f"viewer_save_tooltip_text_{self.UUID}"
                    with dpg.tooltip(parent=self.save_btn):
                        initial_tip = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
                        dpg.add_text(initial_tip, tag=self.save_tooltip_text_tag)

                    dpg.add_checkbox(label="Crop limit", default_value=False, tag=self.crop_limit_tag, callback=self.update_image_callback)
                    with dpg.tooltip(parent=self.crop_limit_tag):
                        dpg.add_text("Crops viewer and exports to the minimal bounding box where alpha > 0")

                    dpg.add_checkbox(label="Zero as alpha", default_value=self.zero_as_alpha, tag=self.zero_as_alpha_tag, callback=self.update_image_callback)
                    with dpg.tooltip(parent=self.zero_as_alpha_tag):
                        dpg.add_text("Sets alpha to 0 for all zero-value pixels to generate an RGBA image with transparent background")

            # Category mapping group
            with dpg.group(horizontal=True, tag=self.cat_group_tag):
                dpg.add_text("Categories:")
                dpg.add_input_int(default_value=self.categories_count, min_value=2, max_value=20, step=1, width=display_scaling.scale(150), tag=self.cat_count_tag, callback=self._update_categories_count)
                dpg.add_button(label="Send Palette", tag=self.btn_send_palette_tag, callback=self.send_palette_to_clipboard)
                dpg.add_button(label="Send Result", tag=self.btn_send_result_tag, callback=self.send_result_to_clipboard)
            
            # Add mouse handler for ROI interaction
            self.handler_reg_tag = f"viewer_handler_reg_{self.UUID}"
            with dpg.handler_registry(tag=self.handler_reg_tag):
                dpg.add_mouse_move_handler(callback=self.on_mouse_move)
                dpg.add_mouse_click_handler(callback=self.on_mouse_click)

            self.init_viewer(1280, 720)
            self._apply_transparency_theme()

    def on_mouse_move(self, sender, data, *args, **kwargs):
        """Tracks mouse position for ROI preview."""
        try:
            if dpg.does_item_exist(self.plot_tag) and dpg.is_item_hovered(self.plot_tag):
                self.is_hovering = True
                # This returns (x, y) in plot coords
                mpos = dpg.get_plot_mouse_pos()
                self.mouse_x = int(mpos[0])
                self.mouse_y = int(mpos[1])
                
                if dpg.get_value(self.roi_show_mask_tag) and self.last_image is not None:
                    self.update_image(self.last_image)
                
                self._update_hover_roi()
            else:
                if self.is_hovering:
                    self.is_hovering = False
                    # Hide hover ROI when leaving plot
                    self._update_hover_roi()
        except Exception:
            # DPG might throw SystemError if item state is invalid during resize/reinit
            self.is_hovering = False

    def on_mouse_click(self, sender, data, *args, **kwargs):
        """Handles mouse clicks, specifically CTRL+Click or Middle Click for ROI creation."""
        btn = data[0] if isinstance(data, (list, tuple)) else data
        is_ctrl_click = (btn == 0 and dpg.is_key_down(dpg.mvKey_LControl))
        is_middle_click = (btn == 2)
        
        # Strictly require Ctrl+LeftClick or MiddleClick
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
            "threshold": dpg.get_value(self.roi_thresh_tag)
        }
        
        self.last_selected_roi_uuid = roi_uuid
        # Immediately draw persistent ROI on screen before triggering downstream modules
        self._update_single_persistent_roi(roi_uuid)
        # Emit
        self._emit_roi(self.rois[roi_uuid], action="create")
        if self.last_image is not None:
            self._emit_cropped_frame(self.last_image, force_emit=True)

    def _emit_roi(self, roi_data, action="select"):
        """Emits ROI_SAMPLE to outputs."""
        if action == "clear":
            sample = {"action": "clear", "clear": True}
        else:
            sample = {
                "uuid": roi_data["uuid"],
                "name": roi_data["name"],
                "rect": roi_data["rect"],
                "threshold": roi_data.get("threshold", 100),
                "action": action
            }
        
        # Send to all connected outputs
        for module in self.connections.get("ROI Data", []):
            module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)

    def clear_rois(self, *args, **kwargs):
        """Clears all ROIs from the viewer and notifies connected modules."""
        self._handle_roi_input({"clear": True})
        self._emit_roi(None, action="clear")
        self.roi_index_counter = 1
        self.last_selected_roi_uuid = None
        self.update_image_callback()

    def input_cb(self, *args, **kwargs):
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
                logger.info(f"Image Viewer export path set to: {self.export_path}")
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
            raw_frame, mask = data[0], data[1]
            h, w = raw_frame.shape[:2]

            # Convert mask to 8-bit alpha channel (0 to 255)
            if mask.dtype == bool:
                alpha = np.where(mask, 255, 0).astype(np.uint8)
            elif mask.max() <= 1 and np.issubdtype(mask.dtype, np.number):
                alpha = (mask * 255).astype(np.uint8)
            else:
                alpha = np.clip(mask, 0, 255).astype(np.uint8)

            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)

            # Ensure base frame is 8-bit color
            if raw_frame.dtype != np.uint8:
                base = self.convert_to_lowdepth(raw_frame)
            else:
                base = raw_frame

            if base.ndim == 2 or (base.ndim == 3 and base.shape[2] == 1):
                bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            elif base.shape[2] == 4:
                bgr = base[:, :, :3]
            else:
                bgr = base

            frame = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
            self.update_image(frame)
            return

        # 5. Dict with "frame" and optional "mask"
        if isinstance(data, dict) and "frame" in data:
            raw_frame = data["frame"]
            if "mask" in data and data["mask"] is not None and isinstance(data["mask"], np.ndarray):
                mask = data["mask"]
                h, w = raw_frame.shape[:2]
                if mask.dtype == bool:
                    alpha = np.where(mask, 255, 0).astype(np.uint8)
                elif mask.max() <= 1 and np.issubdtype(mask.dtype, np.number):
                    alpha = (mask * 255).astype(np.uint8)
                else:
                    alpha = np.clip(mask, 0, 255).astype(np.uint8)

                if alpha.shape[:2] != (h, w):
                    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)

                if raw_frame.dtype != np.uint8:
                    base = self.convert_to_lowdepth(raw_frame)
                else:
                    base = raw_frame

                if base.ndim == 2 or (base.ndim == 3 and base.shape[2] == 1):
                    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                elif base.shape[2] == 4:
                    bgr = base[:, :, :3]
                else:
                    bgr = base

                frame = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
                self.update_image(frame)
                return
            else:
                data = raw_frame

        # 6. Single mask input
        if data_type == IOTypes.MASK and isinstance(data, np.ndarray):
            mask = data
            if mask.dtype == bool:
                alpha = np.where(mask, 255, 0).astype(np.uint8)
            elif mask.max() <= 1 and np.issubdtype(mask.dtype, np.number):
                alpha = (mask * 255).astype(np.uint8)
            else:
                alpha = np.clip(mask, 0, 255).astype(np.uint8)
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
                return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            case "16bit_abs":
                return (frame >> 8).astype(np.uint8)

    def get_minmax_values(self, frame: np.ndarray) -> tuple[int, int]:
        """Returns min and max intensity values. Optimized with OpenCV."""
        if frame is None or frame.size == 0:
            return 0, 255

        if frame.ndim == 2:
            min_val, max_val, _, _ = cv2.minMaxLoc(frame)
        else:
            # For multi-channel images, compute global min/max
            min_val = np.min(frame)
            max_val = np.max(frame)
        
        return int(min_val), int(max_val)

    def _lowdepth_to_original(self, val_8bit, frame):
        """Converts an 8-bit value (from sliders) back to the original image depth range."""
        if frame is None:
            return val_8bit
            
        match self.image_depth:
            case "8bit":
                return val_8bit
            case "12bit":
                return val_8bit << 4
            case "16bit_abs":
                return val_8bit << 8
            case "16bit_norm":
                fmin, fmax = self.get_minmax_values(frame)
                return fmin + (val_8bit / 255.0) * (fmax - fmin)
        return val_8bit

    def _get_current_scale_limits(self):
        """Returns (minval, maxval) in original precision based on UI state (Auto vs Manual)."""
        if self.last_image is None:
            return 0, 255
            
        if dpg.get_value(self.auto_remap_tag):
            sample = self.last_image[::5, ::5]
            minval = int(np.percentile(sample, 1))
            maxval = int(np.max(sample))
            return minval, maxval
        else:
            ui_inf = dpg.get_value(self.inf_tag)
            ui_sup = dpg.get_value(self.sup_tag)
            minval = self._lowdepth_to_original(ui_inf, self.last_image)
            maxval = self._lowdepth_to_original(ui_sup, self.last_image)
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

            lut = np.zeros(256, dtype=np.uint8)
            if high_i > low_i:
                lut[high_i:] = 255
                ramp = np.linspace(0, 255, high_i - low_i + 1, endpoint=True)
                lut[low_i:high_i + 1] = np.clip(np.round(ramp), 0, 255).astype(np.uint8)
            else:
                lut[high_i:] = 255

            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            return cv2.LUT(frame, lut)
        else:
            diff = high_f - low_f
            if diff <= 0:
                diff = 1.0
            clipped = np.clip(frame, low_f, high_f)
            scaled = (clipped.astype(np.float32) - low_f) * (255.0 / diff)
            return np.clip(scaled, 0, 255).astype(np.uint8)

    def add_intensity_scale(self, img: np.ndarray, scale_width=30, position='left') -> np.ndarray:
        """Adds vertical or horizontal intensity scale bar to image."""
        h = img.shape[0]
        scale_w = min(display_scaling.scale(scale_width), img.shape[1])
        if scale_w <= 0:
            return img

        gradient = np.linspace(255, 0, h).reshape((h, 1)).astype(np.uint8)
        scale = np.tile(gradient, (1, scale_w))
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                scale_color = self.colorize_frame(scale)
                if scale_color.ndim == 2:
                    scale_color = cv2.cvtColor(scale_color, cv2.COLOR_GRAY2BGR)
                alpha_chan = np.full((h, scale_w, 1), 255, dtype=np.uint8)
                scale = np.concatenate([scale_color[:, :, :3], alpha_chan], axis=-1)
            else:
                scale_color = self.colorize_frame(scale)
                if scale_color.ndim == 2 and img.shape[2] == 3:
                    scale_color = cv2.cvtColor(scale_color, cv2.COLOR_GRAY2BGR)
                scale = scale_color
        else:
            scale = scale

        overlay = img.copy()

        match position:
            case 'right': overlay[:, -scale_w:] = scale
            case 'left': overlay[:, :scale_w] = scale
            case 'top': overlay[:scale_w, :] = scale.T
            case 'bottom': overlay[-scale_w:, :] = scale.T

        return overlay

    def add_scale_and_annotations(self, img: np.ndarray, min_val=None, max_val=None) -> np.ndarray:
        """Adds scale bar and min/max text to image (supports 2D, BGR, and RGBA)."""
        if img is None or img.size == 0:
            return img

        if min_val is None or max_val is None:
            min_val, max_val = self.get_minmax_values(img)

        img = self.add_intensity_scale(img)

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
            self.fps = 1 / delta
            self.fps_last = self.fps_now
        return self.fps

    def is_crop_limit_enabled(self) -> bool:
        """Returns True if 'Crop limit' checkbox is checked."""
        if hasattr(self, 'crop_limit_tag') and dpg.does_item_exist(self.crop_limit_tag):
            return bool(dpg.get_value(self.crop_limit_tag))
        return False

    def is_zero_as_alpha_enabled(self) -> bool:
        """Returns True if 'Zero as alpha' checkbox is checked."""
        if hasattr(self, 'zero_as_alpha_tag') and dpg.does_item_exist(self.zero_as_alpha_tag):
            return bool(dpg.get_value(self.zero_as_alpha_tag))
        return getattr(self, "zero_as_alpha", False)

    def apply_zero_as_alpha(self, raw_frame: np.ndarray, processed_frame: np.ndarray) -> np.ndarray:
        """
        If 'Zero as alpha' is enabled, creates an RGBA/BGRA image where any pixel that was 0 in raw_frame
        has alpha set to 0 (fully transparent), and all other pixels have alpha set to 255.
        """
        if not self.is_zero_as_alpha_enabled() or raw_frame is None or processed_frame is None:
            return processed_frame

        # Compute mask of zero-value pixels from raw_frame
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

        # Ensure zero_mask matches spatial dimensions of processed_frame
        h, w = processed_frame.shape[:2]
        if zero_mask.shape[:2] != (h, w):
            zero_mask = cv2.resize(zero_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        alpha = np.where(zero_mask, 0, 255).astype(np.uint8)

        if processed_frame.ndim == 2 or (processed_frame.ndim == 3 and processed_frame.shape[2] == 1):
            gray = processed_frame if processed_frame.ndim == 2 else processed_frame[:, :, 0]
            return cv2.merge([gray, gray, gray, alpha])
        elif processed_frame.shape[2] == 3:
            b, g, r = cv2.split(processed_frame)
            return cv2.merge([b, g, r, alpha])
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
        pts = np.argwhere(alpha > 0)
        if pts.size == 0:
            return frame

        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)

        return np.ascontiguousarray(frame[y_min:y_max + 1, x_min:x_max + 1])

    def _apply_transparency_theme(self):
        """Applies transparent plot background theme so transparency directly reveals the window background."""
        if not hasattr(self, "_plot_transparent_theme") or not dpg.does_item_exist(self._plot_transparent_theme):
            with dpg.theme() as self._plot_transparent_theme:
                with dpg.theme_component(dpg.mvPlot):
                    dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
        
        if dpg.does_item_exist(self.plot_tag):
            dpg.bind_item_theme(self.plot_tag, self._plot_transparent_theme)

    def _on_toggle_rgba(self, sender=None, app_data=None, user_data=None):
        """Retained for backward compatibility. Viewer operates permanently in RGBA."""
        self.use_rgba = True

    def convert_to_texture(self, frame: np.ndarray) -> np.ndarray:
        """Converts image to normalized float32 RGBA array for DearPyGui raw texture."""
        h, w = frame.shape[:2]
        expected_size = h * w * 4

        if self._texture_buffer is None or self._texture_buffer.size != expected_size:
            self._texture_buffer = np.zeros(expected_size, dtype=np.float32)

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            # Grayscale -> RGBA (R=G=B=Gray, A=255)
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        elif frame.shape[2] == 4:
            # BGRA -> RGBA (preserves existing Alpha)
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        elif frame.shape[2] == 3:
            # BGR -> RGBA (adds Alpha=255)
            conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            conv_frame = cv2.cvtColor(frame[:, :, :1], cv2.COLOR_GRAY2RGBA)

        if conv_frame.dtype != np.uint8:
            conv_frame = conv_frame.astype(np.uint8)

        flattened = conv_frame.ravel()
        
        if len(self._texture_buffer) == len(flattened):
            np.multiply(flattened, 1.0 / 255.0, out=self._texture_buffer, casting='unsafe')
        else:
            self._texture_buffer = flattened.astype(np.float32) * (1.0 / 255.0)

        return self._texture_buffer

    def init_viewer(self, width: int, height: int):
        """Initializes DearPyGui image display area (main thread safe)."""
        from core.automation_manager import automation_manager
        if threading.current_thread() is threading.main_thread():
            self._init_viewer_internal(width, height)
        else:
            automation_manager.post_to_main_thread(self._init_viewer_internal, width, height, wait=True)

    def _init_viewer_internal(self, width: int, height: int):
        with self._thread_lock:
            self._texture_initialized = False
            self.texture_width = width
            self.texture_height = height

            if dpg.does_item_exist(self.series_tag):
                try:
                    dpg.delete_item(self.series_tag)
                except Exception as e:
                    logger.warning(f"Error deleting series {self.series_tag}: {e}")

            if dpg.does_item_exist(self.texture_tag):
                try:
                    dpg.delete_item(self.texture_tag)
                except Exception as e:
                    logger.warning(f"Error deleting texture {self.texture_tag}: {e}")

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
                # Pre-init Raw RGBA texture
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
                    equal_aspects=True,
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
            # Skip colormapping for multi-channel images
            return frame

        match self.palette:
            case 'B&W': return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            case 'Inferno': return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
            case 'Jet': return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            case 'HSV': return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
        return frame

    def remap_and_annotate(self, frame, min_val=None, max_val=None):
        """Remaps intensities and colorizes frame."""
        # Calculate 8-bit limits for the VIEW (sliders)
        if dpg.get_value(self.auto_remap_tag):
            sample = frame[::5, ::5]
            min_val_8bit = int(np.percentile(sample, 1))
            max_val_8bit = int(np.max(sample))
            dpg.set_value(self.sup_tag, max_val_8bit)
            dpg.set_value(self.inf_tag, min_val_8bit)
            
            if min_val is None: min_val = min_val_8bit
            if max_val is None: max_val = max_val_8bit

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = self.remap_pixel_intensity(frame, dpg.get_value(self.inf_tag), dpg.get_value(self.sup_tag))
        
        frame = self.colorize_frame(frame)
        return frame

    def _update_single_persistent_roi(self, r_uuid):
        """Updates or creates a single persistent ROI on the plot (O(1))."""
        if r_uuid not in self.rois:
            return
        r_data = self.rois[r_uuid]
        r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
        t_tag = f"roi_text_{self.UUID}_{r_uuid}"
        
        x, y, w, h = r_data["rect"]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Create items if they don't exist
        if not dpg.does_item_exist(r_tag):
            dpg.draw_rectangle(pmin=(x1, y1), pmax=(x2, y2), color=(0, 255, 0, 255), thickness=2, parent=self.plot_tag, tag=r_tag)
            dpg.add_plot_annotation(label=r_data["name"], default_value=(x2, y2), color=(0, 255, 0, 255), offset=(10, -10), parent=self.plot_tag, tag=t_tag)
        
        # Update Appearance
        color = (0, 255, 0, 255) if r_data.get("selected", False) else (0, 100, 0, 255)
        fill_color = (0, 255, 0, 70) if r_data.get("highlighted", False) else (0, 0, 0, 0)
        show = dpg.get_value(self.roi_enable_tag)
        
        dpg.configure_item(r_tag, pmin=(x1, y1), pmax=(x2, y2), color=color, fill=fill_color, show=show)
        dpg.configure_item(t_tag, default_value=(x2, y2), label=r_data["name"], color=color, show=show)

    def _update_persistent_rois(self):
        """Updates all persistent ROIs (called on clear/toggle/bulk changes)."""
        for r_uuid in list(self.rois.keys()):
            self._update_single_persistent_roi(r_uuid)

    def _update_hover_roi(self):
        """Updates ONLY preview/hover ROI rectangle and stats on mouse move (O(1))."""
        rect_tag = f"roi_rect_{self.UUID}"
        text_tag = f"roi_text_{self.UUID}"
        
        if not dpg.does_item_exist(rect_tag) or not dpg.does_item_exist(text_tag):
            return

        if not dpg.get_value(self.roi_enable_tag) or not self.is_hovering:
            dpg.configure_item(rect_tag, show=False)
            dpg.configure_item(text_tag, show=False)
            return

        # Get current ROI params
        w = dpg.get_value(self.roi_w_tag)
        h = dpg.get_value(self.roi_h_tag)
        
        # Calculate coords (Center logic)
        x = self.mouse_x - w // 2
        y = self.mouse_y - h // 2
        
        # Clip to image bounds
        img_w, img_h = self.texture_width, self.texture_height
        if img_w == 0 or img_h == 0: return

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

        # Fast ROI stats preview
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

    def _update_roi_overlay(self):
        """Updates both persistent ROIs and hover preview (called on UI callbacks or mode changes)."""
        self._update_persistent_rois()
        self._update_hover_roi()

    def _handle_roi_input(self, sample):
        """Handles updates from ROI Manager."""
        if sample.get("clear"):
            for r_uuid in list(self.rois.keys()):
                r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
                t_tag = f"roi_text_{self.UUID}_{r_uuid}"
                if dpg.does_item_exist(r_tag): dpg.delete_item(r_tag)
                if dpg.does_item_exist(t_tag): dpg.delete_item(t_tag)
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
                if dpg.does_item_exist(r_tag): dpg.delete_item(r_tag)
                if dpg.does_item_exist(t_tag): dpg.delete_item(t_tag)
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
                    "threshold": sample.get("threshold", 100)
                }
                self.last_selected_roi_uuid = uuid
                self._update_single_persistent_roi(uuid)

    def process_roi(self, frame, processed_frame):
        """Calculates ROI stats using raw frame and annotates processed_frame."""
        if not dpg.get_value(self.roi_enable_tag):
            return processed_frame

        # We still calculate mask for visualization if requested
        show_mask = dpg.get_value(self.roi_show_mask_tag)
        if not show_mask:
            return processed_frame

        # Get ROI parameters (We need to recalculate coords here for the thread)
        # NOTE: This runs in a thread, self.mouse_x/y might change during execution.
        # Ideally we snapshot them, but for visual mask it's okay.
        x = self.mouse_x
        y = self.mouse_y
        w = dpg.get_value(self.roi_w_tag)
        h = dpg.get_value(self.roi_h_tag)
        thresh_pct = dpg.get_value(self.roi_thresh_tag)

        # Center ROI on mouse
        x = x - w // 2
        y = y - h // 2

        # Ensure ROI is within bounds
        img_h, img_w = frame.shape[:2]
        
        # Clip coordinates to image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        # DPG Visualization is now handled in _update_roi_overlay (Main Thread)
        # We only handle mask stamping here.

        # Calculation (on Raw data)
        # Correct for coordinate system mismatch:
        # DPG Plot (0,0) is Bottom-Left. Image Array (0,0) is Top-Left.
        
        # Invert Y for array slicing
        y1_arr = img_h - y2
        y2_arr = img_h - y1
        
        # Ensure valid slice (handle negative indices from OOB)
        y1_arr = max(0, min(y1_arr, img_h))
        y2_arr = max(0, min(y2_arr, img_h))
        
        roi = frame[y1_arr:y2_arr, x1:x2]
        
        if roi.size > 0:
            # Calculate Threshold
            if roi.ndim == 3 and roi.shape[2] >= 3:
                mask_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            else:
                mask_base = roi
                
            p_thresh = np.percentile(mask_base, 100 - thresh_pct)
            
            # Create mask
            mask = mask_base >= p_thresh
            
            # Show Selection Mask (Red overlay) - Still needs OpenCV modification of texture
            if show_mask:
                vis_roi = processed_frame[y1_arr:y2_arr, x1:x2]
                color = [0, 0, 255]
                if vis_roi.ndim == 3 and vis_roi.shape[2] == 4:
                    color = [0, 0, 255, 255]
                vis_roi[mask] = color
                
        return processed_frame

    def process_lowdepth(self, frame, min_val=None, max_val=None, mask=None):
        """Processes image before rendering and adds overlay if mask is provided."""
        frame = self.remap_and_annotate(frame, min_val, max_val)
        if mask is not None:
            frame[mask] = [0, 0, 255]
        return frame

    def _emit_cropped_frame(self, frame, force_emit=False):
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

            y1_arr = img_h - y2
            y2_arr = img_h - y1
            y1_arr = max(0, min(y1_arr, img_h))
            y2_arr = max(0, min(y2_arr, img_h))
            
            crop = frame[y1_arr:y2_arr, x1:x2]
            if crop.size > 0:
                for module in self.connections.get("Cropped Frame", []):
                    module.input_cb(data=crop, data_type=IOTypes.FRAME)

    def _emit_lowdepth_frame(self, frame, force_emit=False):
        if not self.connections.get("Lowdepth Frame"):
            return

        minval, maxval = self._get_current_scale_limits()
        lowdepth = self.convert_to_lowdepth(frame)
        image = self.remap_and_annotate(lowdepth, minval, maxval)
        
        for module in self.connections.get("Lowdepth Frame", []):
            module.input_cb(data=image, data_type=IOTypes.FRAME)

    def update_image(self, frame):
        """
        Receives a new frame, emits dependent frames, and signals render thread (non-blocking).
        """
        if frame is None:
            return
        
        # Calculate incoming FPS
        self.calc_fps()
        
        self._emit_cropped_frame(frame)
        self._emit_lowdepth_frame(frame)
        
        self.last_image = frame
        with self.frame_lock:
            self.latest_frame = frame
        self.update_event.set()

    def _render_loop(self):
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
                    logger.warning(f"Error in Image Viewer render loop: {e}")

    def update_image_callback(self, *args, **kwargs):
        """Callback to trigger update from UI."""
        if self.last_image is not None:
            with self.frame_lock:
                self.latest_frame = self.last_image
            self.update_event.set()

        # Also update overlay immediately for responsiveness (lightweight)
        self._update_roi_overlay()

    def _render_frame(self, frame):
        """Processes the frame and updates the GPU texture."""
        if frame is None:
            return
        with self.lock:
            try:
                # 1. Process color / grayscale image
                if frame.ndim == 3 and frame.shape[2] in [3, 4] and frame.dtype == np.uint8:
                    processed = frame.copy()
                    minval, maxval = self.get_minmax_values(frame)
                    processed = self.process_roi(frame, processed)
                else:
                    lowdepth = self.convert_to_lowdepth(frame)
                    minval, maxval = self._get_current_scale_limits()
                    processed = self.remap_and_annotate(lowdepth, minval, maxval)
                    processed = self.process_roi(frame, processed)

                # 2. Apply Zero as Alpha (converts to RGBA with alpha=0 on zeros)
                if self.is_zero_as_alpha_enabled():
                    processed = self.apply_zero_as_alpha(frame, processed)

                # 3. Crop to alpha limit (minimal bounding box where alpha > 0)
                if self.is_crop_limit_enabled():
                    processed = self.crop_to_alpha_limit(processed)

                # 4. Add scale annotations on top (always fully visible with alpha=255)
                if dpg.get_value(self.show_scale_tag):
                    processed = self.add_scale_and_annotations(processed, minval, maxval)

                # 5. Handle texture update based on mode
                h, w = processed.shape[:2]
                texture_data = self.convert_to_texture(processed)

                if self.texture_mode == "static":
                    dim_changed = (h != self.texture_height or w != self.texture_width or not self._texture_initialized)
                    self.texture_width = w
                    self.texture_height = h

                    def _recreate_static():
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

                    from core.automation_manager import automation_manager
                    if threading.current_thread() is threading.main_thread():
                        _recreate_static()
                    else:
                        automation_manager.post_to_main_thread(_recreate_static, wait=True)

                else:
                    # Raw mode: Pre-init raw texture + set_value
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
                logger.error(f"Error in Image Viewer render: {e}")

    def set_texture_mode(self, mode: str) -> None:
        """Switches texture rendering mode ('raw' or 'static')."""
        mode_clean = mode.lower()
        if mode_clean in ("raw", "static") and mode_clean != self.texture_mode:
            with self._thread_lock:
                self.texture_mode = mode_clean
                self._texture_initialized = False
            self.update_image_callback()

    def set_mode(self, sender, app_data, user_data=None, *args):
        """Handles UI mode changes (negative toggle or palette switch)."""
        if app_data in ["Normal", "Negative"]:
            self.negative = (app_data == "Negative")
        else:
            self.palette = app_data
        
        if self.last_image is not None:
            self._render_frame(self.last_image)

    def set_depth(self, sender, app_data, user_data=None, *args):
        """Handles image depth changes."""
        self.image_depth = app_data
        if self.last_image is not None:
            self._render_frame(self.last_image)
        self._update_roi_overlay()

    def copy_to_clipboard(self, *args, **kwargs):
        """Copies the processed image to clipboard using the custom injector."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning("No image available to copy to clipboard.")
                    return

                # Defensive copy to prevent race condition with render/input threads
                target_img = self.last_image.copy()

                if target_img.ndim == 3 and target_img.shape[2] in [3, 4] and target_img.dtype == np.uint8:
                    processed = target_img.copy()
                    minval, maxval = self.get_minmax_values(target_img)
                    processed = self.process_roi(target_img, processed)
                else:
                    minval, maxval = self._get_current_scale_limits()
                    lowdepth = self.convert_to_lowdepth(target_img)
                    processed = self.remap_and_annotate(lowdepth, minval, maxval)
                    processed = self.process_roi(target_img, processed)

                if self.is_zero_as_alpha_enabled():
                    processed = self.apply_zero_as_alpha(target_img, processed)

                if self.is_crop_limit_enabled():
                    processed = self.crop_to_alpha_limit(processed)

                if dpg.get_value(self.show_scale_tag):
                    processed = self.add_scale_and_annotations(processed, minval, maxval)

                clipboardinjector.send_image(processed)
            except Exception as e:
                logger.error(f"Error copying image to clipboard: {e}")

    def save_image(self, *args, **kwargs):
        """Saves both raw and processed images to PNG with indexed names."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning("No image available to save.")
                    return

                # Defensive copy to prevent race condition with render/input threads
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
                        pts = np.argwhere(alpha > 0)
                        if pts.size > 0:
                            y_min, x_min = pts.min(axis=0)
                            y_max, x_max = pts.max(axis=0)
                            processed = np.ascontiguousarray(processed[y_min:y_max + 1, x_min:x_max + 1])
                            if raw.shape[:2] == alpha.shape[:2]:
                                raw = np.ascontiguousarray(raw[y_min:y_max + 1, x_min:x_max + 1])
                    else:
                        processed = self.crop_to_alpha_limit(processed)

                if dpg.get_value(self.show_scale_tag):
                    processed = self.add_scale_and_annotations(processed, minval, maxval)

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
                logger.error(f"Error saving image: {e}")

    def update_save_tooltip(self):
        """Updates the tooltip for the Save button based on the current export path."""
        if hasattr(self, 'save_tooltip_text_tag') and dpg.does_item_exist(self.save_tooltip_text_tag):
            tip_text = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
            dpg.set_value(self.save_tooltip_text_tag, tip_text)

    def reset_counter(self, *args, **kwargs):
        """Resets the saved image counter to zero."""
        self.imgcount = 0
        dpg.set_value(self.count_tag, f"count: {self.imgcount}")

    def _update_categories_count(self, sender, app_data, user_data=None, *args, **kwargs):
        self.categories_count = app_data

    def send_palette_to_clipboard(self, *args, **kwargs):
        """Computes and sends the current category palette to the clipboard in palette.json format."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning("No image loaded to compute palette.")
                    return
                
                # Get number of categories from UI
                num_categories = dpg.get_value(self.cat_count_tag)
                if num_categories < 1:
                    logger.warning("Number of categories must be at least 1.")
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
                logger.success(f"Palette with {num_categories} categories copied to clipboard in palette.json format.")
            except Exception as e:
                logger.error(f"Error sending palette to clipboard: {e}")

    def send_result_to_clipboard(self, *args, **kwargs):
        """Computes the category distribution and sends it to the clipboard in TSV format."""
        with self.lock:
            try:
                if self.last_image is None:
                    logger.warning("No image loaded to compute proportions.")
                    return

                num_categories = dpg.get_value(self.cat_count_tag)
                if num_categories < 1:
                    logger.warning("Number of categories must be at least 1.")
                    return

                # Defensive copy to prevent race condition with render/input threads
                target_img = self.last_image.copy()
                if self.is_crop_limit_enabled():
                    target_img = self.crop_to_alpha_limit(target_img)

                if target_img.ndim == 3 and target_img.shape[2] >= 3:
                    raw_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY if target_img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
                else:
                    raw_gray = target_img

                # Get current scale limits in raw values
                minval, maxval = self._get_current_scale_limits()

                # Determine the region of interest
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

                    # Invert Y coordinate for numpy array slicing (coordinate system mismatch)
                    y1_arr = img_h - y2
                    y2_arr = img_h - y1
                    y1_arr = max(0, min(y1_arr, img_h))
                    y2_arr = max(0, min(y2_arr, img_h))

                    roi_raw = raw_gray[y1_arr:y2_arr, x1:x2]
                    
                    if roi_raw.size > 0:
                        p_thresh = np.percentile(roi_raw, 100 - thresh_pct)
                        # Exclude zero values
                        mask = (roi_raw >= p_thresh) & (roi_raw > 0)
                        target_raw = roi_raw[mask]
                    else:
                        target_raw = np.array([], dtype=raw_gray.dtype)
                else:
                    # Exclude zero values from the entire image
                    mask = raw_gray > 0
                    target_raw = raw_gray[mask]

                # Compute range and binning
                range_diff = maxval - minval
                if range_diff <= 0:
                    range_diff = 1.0

                if target_raw.size > 0:
                    # Normalize raw intensities to [0.0, 1.0] range
                    normalized = (target_raw.astype(np.float32) - minval) / range_diff
                    np.clip(normalized, 0.0, 1.0, out=normalized)
                    
                    # Map to [0, num_categories - 1] bin indices
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
                logger.success(f"Category distribution copied to clipboard in TSV format.")
            except Exception as e:
                logger.error(f"Error sending result to clipboard: {e}")

    def is_ready(self) -> bool:
        """Indicate whether the image viewer is ready to accept and render new frames."""
        return self.running

    def on_close(self) -> None:
        """Release resources, background render loop, and GPU texture buffers."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self.running = False
        self.update_event.set()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.2)
        if hasattr(self, "handler_reg_tag") and dpg.does_item_exist(self.handler_reg_tag):
            try:
                dpg.delete_item(self.handler_reg_tag)
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

    def close(self) -> None:
        """Close viewer window and unregister."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()
        super().destroy()


ImageViewerWin = Image_viewer_win
EXPORTED_CLASS = Image_viewer_win
EXPORTED_NAME = "Image Viewer"
