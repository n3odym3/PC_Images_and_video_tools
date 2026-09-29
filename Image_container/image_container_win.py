from __future__ import annotations

import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.file_explorer import file_explorer
from loguru import logger
import os
import sys
import cv2
import threading
import time
import numpy as np
import re
import zipfile
import json
import queue
import pickle
from config.theme_manager import highlight, theme_manager
from config.display_scaling import display_scaling
from typing import Any

class ImageContainerWin(WindowBase):
    """
    ImageContainerWin:
    Loads images from a folder path (String input) or receives frames (Frame input) into memory.
    Allows sequential playback and filtering.
    """
    folder_path: str
    image_cache: dict[str, np.ndarray]
    metadata_cache: dict[str, dict[str, Any]]
    current_list: list[str]
    playback_active: bool
    stop_playback_flag: bool
    playback_thread: threading.Thread | None
    groups: list[str]
    frame_counter: int
    active_rois: dict[str, dict[str, Any]]
    ts_reference: float
    manual: bool
    pending_group: str | None
    listbox_tag: str
    group_combo_tag: str
    filter_input_tag: str
    fps_drag_tag: str
    count_text_tag: str
    progress_modal_tag: str
    progress_bar_tag: str
    progress_text_tag: str
    x_axis_combo_tag: str
    clear_list_btn_tag: str
    btn_export_tag: str
    btn_import_tag: str
    record_check_tag: str
    save_check_tag: str
    timestamp_check_tag: str
    browse_btn_tag: str
    handler_registry_tag: str
    _persistent_fields: list[str]
    accepted_input_types: list[IOTypes]
    name_queue: list[str]
    roi_queue: queue.Queue
    play_btn: int | str
    stop_btn: int | str
    outputs: dict[str, Any]
    connections: dict[str, list[Any]]

    def __init__(
        self,
        label: str = "Image Container",
        win_width: int = display_scaling.scale(300),
        win_height: int = display_scaling.scale(500),
        pos: tuple[int, int] = (50, 50),
        uuid: str | None = None,
        outputs: dict[str, Any] | None = None,
        visible: bool = True,
        folder_path: str | None = None,
        **kwargs: Any,
    ) -> None:

        if outputs is None:
            outputs = {
                "Frame": IOTypes.FRAME,
                "Result": IOTypes.SAMPLE,
                "Folder": IOTypes.FOLDER_PATH,
                "Group": IOTypes.TEXT,
                "CMD": IOTypes.CMD_DICT,
                "ROI": IOTypes.ROI_SAMPLE,
                "Image Name": IOTypes.TEXT,
                "Cropped Images": IOTypes.FRAME_LIST,
                "Frame List": IOTypes.FRAME_LIST,
            }

        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
            **kwargs,
        )

        self.outputs = outputs or {}
        self.connections = {k: [] for k in self.outputs}
        self._is_closing = False

        # State
        self.folder_path = folder_path if folder_path else os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "experiments")
        self.image_cache = {}  # {filename: image_data}
        self.metadata_cache = {} # {filename: {id, group}}
        self.current_list = [] # List of filenames currently displayed/filtered
        self.playback_active = False
        self.stop_playback_flag = False
        self.playback_thread = None
        self.groups = ["All"]
        self.frame_counter = 0 # For naming incoming frames
        self.active_rois = {} # {uuid: {sample, x, y, func}}
        self._roi_cache = {} # {uuid: {sample, prepared, x, y, labels, list_len}}
        self.ts_reference = time.time()
        self.manual = False # Manual mode flag to disable auto-scan
        self.pending_group = None # Group to select after next scan
        
        # UI Tags
        self.browse_btn_tag = f"ic_browse_{self.UUID}"
        self.listbox_tag = f"ic_listbox_{self.UUID}"
        self.group_combo_tag = f"ic_group_combo_{self.UUID}"
        self.filter_input_tag = f"ic_filter_input_{self.UUID}"
        self.fps_drag_tag = f"ic_fps_drag_{self.UUID}"
        self.count_text_tag = f"ic_count_text_{self.UUID}"
        self.progress_modal_tag = f"ic_progress_modal_{self.UUID}"
        self.progress_bar_tag = f"ic_progress_bar_{self.UUID}"
        self.progress_text_tag = f"ic_progress_text_{self.UUID}"
        self.x_axis_combo_tag = f"ic_xaxis_combo_{self.UUID}"
        self.send_framelist_btn_tag = f"ic_send_framelist_{self.UUID}"
        self.clear_list_btn_tag = f"ic_clear_list_btn_{self.UUID}"

        # Tags for permission management
        self.btn_export_tag = f"ic_export_{self.UUID}"
        self.btn_import_tag = f"ic_import_{self.UUID}"
        self.record_check_tag = f"ic_record_check_{self.UUID}"
        self.save_check_tag = f"ic_save_check_{self.UUID}"
        self.timestamp_check_tag = f"ic_timestamp_check_{self.UUID}"
        
        # Module Configuration
        self._persistent_fields = ["label", "folder_path"]
        self.accepted_input_types = [IOTypes.FOLDER_PATH, IOTypes.FRAME, IOTypes.FRAME12, IOTypes.FRAME16, IOTypes.ROI_SAMPLE, IOTypes.CMD_DICT] 
        
        # Load config.json
        self.config = {
            "emit_clear_on_group_selection": True,
            "emit_fo_fm_on_group_selection": False
        }
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    for k in ["emit_clear_on_group_selection", "emit_clear_on groupe_selection"]:
                        if k in user_config:
                            self.config["emit_clear_on_group_selection"] = bool(user_config[k])
                    if "emit_fo_fm_on_group_selection" in user_config:
                        self.config["emit_fo_fm_on_group_selection"] = bool(user_config["emit_fo_fm_on_group_selection"])
            except Exception as e:
                logger.error(f"Error loading config.json for ImageContainer: {e}")

        self.name_queue = [] # Queue for incoming image names
        
        self.roi_queue = queue.Queue()
        threading.Thread(target=self._roi_worker, daemon=True).start()

        self._build_interface(win_width, win_height)
        self.update_permission()

    def update_permission(self) -> None:
        """
        Adapt UI based on app_state.mode.
        In 'user' mode: hide export, import, play, stop, fps, record, save, timestamp, and disable all inputs.
        """
        from core.app_state import app_state

        mode = app_state.mode
        is_user = (mode == "user")

        # Elements to disable and hide
        hidden_elements = [
            self.btn_export_tag,
            self.btn_import_tag,
            self.play_btn,
            self.stop_btn,
            self.fps_drag_tag,
            self.record_check_tag,
            self.save_check_tag,
            self.timestamp_check_tag,
            self.filter_input_tag,
            self.x_axis_combo_tag,
            self.clear_list_btn_tag,
            self.send_framelist_btn_tag,
        ]

        for tag in hidden_elements:
            if dpg.does_item_exist(tag):
                if is_user:
                    dpg.hide_item(tag)
                    dpg.configure_item(tag, enabled=False)
                else:
                    dpg.show_item(tag)
                    dpg.configure_item(tag, enabled=True)

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _build_interface(self, width: int, height: int) -> None:
        with dpg.window(label=self.label, width=width, height=height, pos=self.pos, tag=self.winID, show=self.visible):

            with dpg.group(horizontal=True):
                dpg.add_button(label="Browse", tag=self.browse_btn_tag, width=display_scaling.scale(70), callback=self._on_browse_clicked)
                dpg.add_button(label="Export", tag=self.btn_export_tag, width=display_scaling.scale(70), callback=self._export_experiment_to_file)
                dpg.add_button(label="Import", tag=self.btn_import_tag, width=display_scaling.scale(70), callback=self._import_experiment_from_file)

            dpg.add_separator()

            # Filters (Highlighted block)
            filter_block_tag = f"filter_block_{self.UUID}"
            with dpg.child_window(tag=filter_block_tag, border=True, auto_resize_y=True, width=-1):
                with dpg.group(horizontal=True):
                    dpg.add_text("Group:")
                    dpg.add_combo(self.groups, default_value="All", tag=self.group_combo_tag, callback=self._apply_filters, width=display_scaling.scale(120))

                with dpg.group(horizontal=True):
                    dpg.add_text("Filter:")
                    dpg.add_input_text(hint="Filter filenames...", tag=self.filter_input_tag, callback=self._apply_filters, width=display_scaling.scale(120))
            highlight(filter_block_tag)

            dpg.add_separator()

            # Playback Controls
            with dpg.group(horizontal=True):
                self.play_btn = dpg.add_button(label="\uf04b", width=display_scaling.scale(30), callback=self._start_playback)
                self.stop_btn = dpg.add_button(label="\uf04d", width=display_scaling.scale(30), callback=self._stop_playback)
                theme_manager.apply_icon_font(self.play_btn)
                theme_manager.apply_icon_font(self.stop_btn)

                dpg.add_drag_int(label="FPS", default_value=10, min_value=1, max_value=60, tag=self.fps_drag_tag, width=display_scaling.scale(80))

            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Record", tag=self.record_check_tag, default_value=False)
                dpg.add_checkbox(label="Save", tag=self.save_check_tag, default_value=False)
                dpg.add_checkbox(label="Timestamp", tag=self.timestamp_check_tag, default_value=False)

            dpg.add_combo(["Index", "Time", "Intensity"], default_value="Time", label="X-Axis", tag=self.x_axis_combo_tag, width=display_scaling.scale(120))

            with dpg.group(horizontal=True):
                dpg.add_text("0 images", tag=self.count_text_tag)
                dpg.add_button(
                    label="Send",
                    tag=self.send_framelist_btn_tag,
                    width=display_scaling.scale(60),
                    callback=self._send_current_framelist,
                )

            dpg.add_button(label="Clear List", tag=self.clear_list_btn_tag, callback=self.clear_list, width=-1)

            # Image List
            dpg.add_listbox([], tag=self.listbox_tag, callback=self._on_list_select, width=-1, num_items=15)

        # Create hidden modal for progress
        with dpg.window(
            label="Processing...",
            modal=True,
            show=False,
            tag=self.progress_modal_tag,
            no_title_bar=True,
            width=display_scaling.scale(300),
            height=display_scaling.scale(120),
        ):
            with dpg.group(horizontal=True):
                dpg.add_loading_indicator(radius=2.0)
                dpg.add_text("Working...", tag=self.progress_text_tag)
            dpg.add_spacer(height=display_scaling.scale(10))
            dpg.add_progress_bar(tag=self.progress_bar_tag, width=-1)

        self.handler_registry_tag = f"ic_handler_reg_{self.UUID}"
        with dpg.item_handler_registry(tag=self.handler_registry_tag):
            dpg.add_item_resize_handler(callback=self._on_resize)
        dpg.bind_item_handler_registry(self.winID, self.handler_registry_tag)

    def _on_resize(self, *args: Any) -> None:
        if not dpg.does_item_exist(self.winID) or not dpg.does_item_exist(self.listbox_tag):
            return
        win_h = dpg.get_item_height(self.winID)
        if win_h and win_h > 0:
            # Convert window height from scaled physical pixels to base (unscaled) coordinates
            scale_factor = max(0.1, display_scaling.scale_factor)
            win_h_base = win_h / scale_factor
            
            new_num_items = max(3, int((win_h_base - 300) / 30.0))
            dpg.configure_item(self.listbox_tag, num_items=new_num_items)

    def _on_browse_clicked(self, *args: Any, **kwargs: Any) -> None:
        file_explorer.select_folder(default_path=self.folder_path, callback=self._on_folder_selected)

    def _on_folder_selected(self, selected_dir: str | None, *args: Any, **kwargs: Any) -> None:
        if selected_dir:
            self.input_cb(data=selected_dir)

    def _get_x_value(self, filename: str, index: int) -> float:
        """
        Determines the X val based on selected mode.
        If 'Time (from name)', parses last number after '_'.
        Else returns index.
        """
        mode = dpg.get_value(f"ic_xaxis_combo_{self.UUID}")
        if mode == "Time":
            try:
                name_no_ext = os.path.splitext(filename)[0]
                parts = name_no_ext.split('_')
                if parts and parts[-1].isdigit():
                    return float(parts[-1])
            except Exception:
                pass # Fallback to index
        elif mode == "Intensity":
            try:
                # Check metadata for 'A' value, fallback to 0
                meta = self.metadata_cache.get(filename, {})
                if 'A' in meta:
                    return float(meta['A'])
                # Also try to extract it from the path just in case metadata isn't ready
                name_no_ext = os.path.splitext(filename)[0]
                a_match = re.search(r'[_\-]A(\d+)(?:[_\-]|$)', f"_{name_no_ext}_")
                if a_match:
                    return float(a_match.group(1))
                return 0.0 # Default to 0 instead of index for intensity to keep scales consistent
            except Exception:
                pass
                
        return float(index)

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle input data.
        - String: Treat as folder path, load images (Append mode).
        - Frame (ndarray) or (ndarray, str_name): Append to list and output immediately.
        - ROI_SAMPLE: Trigger batch processing on current sequence.
        """
        # Handle ROI Batch Processing
        if kwargs.get("data_type") == IOTypes.ROI_SAMPLE:
            sample = kwargs["sample"]
            
            if sample.get("clear"):
                self.active_rois = {}
                return

            action = sample.get("action")
            
            # Only process on 'select' or 'create'. Ignore 'highlight', 'unhighlight'
            if action in ["select", "create", "delete", "unselect", "rename"]:
                self.roi_queue.put(sample)
            return

        # Handle CMD Input
        cmd = None
        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")
            
        if cmd and isinstance(cmd, dict):
            if cmd.get("browse", False):
                self._on_browse_clicked()
            
            if "record" in cmd:
                dpg.set_value(f"ic_record_check_{self.UUID}", bool(cmd["record"]))

            if "save" in cmd:
                dpg.set_value(f"ic_save_check_{self.UUID}", bool(cmd["save"]))
            
            if "add_timestamp" in cmd:
                dpg.set_value(f"ic_timestamp_check_{self.UUID}", bool(cmd["add_timestamp"]))
            if cmd.get("reset_ts", False):
                self.ts_reference = time.time()
            
            if "manual" in cmd:
                self.manual = bool(cmd["manual"])

            if "names" in cmd and isinstance(cmd["names"], list):
                self.name_queue = list(cmd["names"])
            
            if cmd.get("clear", False):
                visited = cmd.get("visited", [])
                if self.UUID in visited:
                    return
                cmd["visited"] = visited + [self.UUID]
                self.clear_list(cmd=cmd)
            
            if cmd.get("scan", False) and not self.manual:
                self.clear_list()
                threading.Thread(target=self._load_images, daemon=True).start()
                
            if "group" in cmd:
                group_name = str(cmd["group"])
                self.pending_group = group_name
                # We need to ensure the group exists in the list to select it effectively
                if group_name not in self.groups:
                    self.groups.append(group_name)
                    dpg.configure_item(self.group_combo_tag, items=self.groups)
                
                dpg.set_value(self.group_combo_tag, group_name)
                self._apply_filters()
            
            # if cmd.get("export", False):
            #     self._export_experiment_to_file()
            
            if "import_experiment" in cmd:
                self._import_experiment(cmd["import_experiment"])

            if cmd.get("auto_select_first_two", False):
                # Automatically select first two images to initialize fluo_selector
                if len(self.current_list) >= 2:
                    # Select first image
                    self._on_list_select(None, self.current_list[0])
                    # Small delay then select second image
                    threading.Timer(0.1, lambda: self._on_list_select(None, self.current_list[1])).start()
                    logger.info(f"Auto-selected first two images: {self.current_list[0]} and {self.current_list[1]}")

            return

        path_input = data
        if path_input is None:
            path_input = kwargs.get("path")
            if path_input is None:
                path_input = kwargs.get("folder_path")
            if path_input is None:
                path_input = kwargs.get("data")

        if isinstance(path_input, str):
            if str(path_input).lower().endswith(".zip"):
                self._emit_folder(self.folder_path)
                threading.Thread(target=self._import_experiment_from_file, args=(None, path_input), daemon=True).start()
            else:
                try:
                    if not os.path.exists(path_input):
                        os.makedirs(path_input, exist_ok=True)
                        logger.info(f"Created new directory: {path_input}")
                    if os.path.isdir(path_input):
                        self.folder_path = path_input
                        if not self.manual:
                            self._show_progress("Indexing files...")
                            threading.Thread(target=self._load_images, daemon=True).start()
                        self._emit_folder(path_input)
                    else:
                        logger.warning(f"ImageContainer received invalid path (not a directory): {path_input}")
                except Exception as e:
                    logger.warning(f"ImageContainer received invalid path: {path_input} ({e})")

        else:
            # Assume it's a frame or (frame, name)
            img = None
            base_name = None
            
            if isinstance(data, np.ndarray):
                img = data
                base_name = f"frame_{self.frame_counter}"
                self.frame_counter += 1
            elif isinstance(data, (tuple, list)) and len(data) >= 2:
                if isinstance(data[0], np.ndarray) and isinstance(data[1], str):
                    img = data[0]
                    base_name = data[1]
            elif isinstance(data, dict) and 'frame' in data:
                img = data.get('frame')
                base_name = data.get('name')

            if img is not None:
                # Determine Record Indent
                should_record = dpg.get_value(f"ic_record_check_{self.UUID}")
                forced_name = None
                
                # Check Name Queue (Overrides Record state)
                if self.name_queue:
                    forced_name = self.name_queue.pop(0)
                    should_record = True # Force record for queued names
                
                # Check Record state
                if should_record:
                    # Generate unique name
                    if forced_name:
                        name = forced_name
                        # Try to extract group from forced name if it has an underscore (e.g. 0_GTEST_Photo)
                        parts = forced_name.split('_')
                        if len(parts) >= 2:
                            # Typically parts[1] is the group name if format is count_group_capture
                            group_name = parts[1]
                        else:
                            group_name = forced_name
                    else:
                        name = base_name or "Unknown"
                        group_name = base_name if base_name else "Stream" # Use base name as group
                    
                    if dpg.get_value(f"ic_timestamp_check_{self.UUID}"):
                        ts_ms = int((time.time() - self.ts_reference) * 1000)
                        name = f"{name}_{ts_ms}"
                    
                    if name in self.image_cache:
                        i = 1
                        while f"{name}_{i}" in self.image_cache:
                            i += 1
                        name = f"{name}_{i}"
                    
                    self.image_cache[name] = img.copy()
                    
                    # Save to disk if Save checkbox is checked
                    should_save = dpg.get_value(f"ic_save_check_{self.UUID}")
                    if should_save and self.folder_path:
                        try:
                            os.makedirs(self.folder_path, exist_ok=True)
                            save_name = name if name.lower().endswith('.png') else f"{name}.png"
                            save_path = os.path.join(self.folder_path, save_name)
                            cv2.imwrite(save_path, img)
                            logger.debug(f"Saved image to {save_path}")
                        except Exception as e:
                            logger.error(f"Failed to save image: {e}")
                    
                    # Metadata for frames
                    self._parse_metadata(name)
                    
                    self._update_list_ui()
                    self._on_list_select(None, name)
                    if self.active_rois:
                        self._process_active_rois_single_frame(name)
                else:
                    for module in self.connections.get("Frame", []):
                        if hasattr(module, 'input_cb'):
                            try:
                                module.input_cb(img)
                            except Exception as e:
                                logger.error(f"Error passing through frame: {e}")

    # ------------------------------------------------------------------
    # ROI computation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_roi_mean(img: np.ndarray, prepared: dict[str, Any]) -> float:
        """
        Compute the mean pixel value on img using a pre-prepared ROI descriptor.

        ``prepared`` is built once before a batch loop by
        ``_prepare_roi_mask`` or ``_prepare_roi_rect`` so that no
        per-image allocations are needed.

        Returns a float (0.0 if no valid pixels).
        """
        kind = prepared["kind"]
        if kind == "mask":
            mask_crop = prepared["mask_crop"]   # bool array cropped to bbox
            bx, by, bw, bh = prepared["bbox"]   # (x1, y1_arr, w, h) in array coords
            ref_shape = prepared["ref_shape"]    # (H, W) the mask was built for

            # Resize mask only when image dimensions differ from the reference
            ih, iw = img.shape[:2]
            if (ih, iw) != ref_shape:
                full_mask = cv2.resize(
                    prepared["mask_full"].astype(np.uint8),
                    (iw, ih), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                # Recompute bbox rows for this size
                rows = np.any(full_mask, axis=1)
                cols = np.any(full_mask, axis=0)
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                mask_crop = full_mask[r0:r1+1, c0:c1+1]
                img_crop  = img[r0:r1+1, c0:c1+1]
            else:
                r0, c0 = by, bx
                r1, c1 = by + bh - 1, bx + bw - 1
                img_crop = img[r0:r1+1, c0:c1+1]

            if img_crop.size == 0:
                return 0.0
            # Two-pass version: faster than fancy-index mean for large arrays
            valid = mask_crop & (img_crop > 0)
            n = int(np.count_nonzero(valid))
            if n == 0:
                return 0.0
            return float(np.sum(img_crop[valid])) / n

        else:  # kind == "rect"
            h, w = img.shape[:2]
            x1, x2 = prepared["x1"], min(w, prepared["x2"])
            y1_arr   = h - prepared["y_top"]
            y2_arr   = h - prepared["y_bot"]
            y1_arr   = max(0, min(y1_arr, h))
            y2_arr   = max(0, min(y2_arr, h))
            if x2 <= x1 or y2_arr <= y1_arr:
                return 0.0
            roi_img = img[y1_arr:y2_arr, x1:x2]
            if roi_img.size == 0:
                return 0.0
            threshold = prepared["threshold"]
            if threshold >= 100:
                return float(np.mean(roi_img))
            p_thresh = np.percentile(roi_img, 100 - threshold)
            mask = roi_img >= p_thresh
            n = int(np.count_nonzero(mask))
            return float(np.sum(roi_img[mask])) / n if n > 0 else 0.0

    @staticmethod
    def _crop_roi(img: np.ndarray, prepared: dict[str, Any]) -> np.ndarray:
        """
        Return the image cropped to the ROI bounding box.
        For mask ROIs, returns the bbox crop (mask info is preserved
        in ``prepared`` for downstream use if needed).
        """
        kind = prepared["kind"]
        if kind == "mask":
            bx, by, bw, bh = prepared["bbox"]
            ref_shape = prepared["ref_shape"]
            ih, iw = img.shape[:2]
            if (ih, iw) != ref_shape:
                full_mask = cv2.resize(
                    prepared["mask_full"].astype(np.uint8),
                    (iw, ih), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                rows = np.any(full_mask, axis=1)
                cols = np.any(full_mask, axis=0)
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                img_crop = img[r0:r1+1, c0:c1+1]
            else:
                r0, c0 = by, bx
                r1, c1 = by + bh - 1, bx + bw - 1
                img_crop = img[r0:r1+1, c0:c1+1]
            return img_crop if img_crop.size else np.zeros((1, 1), dtype=img.dtype)
        else:  # rect
            h, w = img.shape[:2]
            x1, x2 = prepared["x1"], min(w, prepared["x2"])
            y1_arr = h - prepared["y_top"]
            y2_arr = h - prepared["y_bot"]
            y1_arr = max(0, min(y1_arr, h))
            y2_arr = max(0, min(y2_arr, h))
            if x2 <= x1 or y2_arr <= y1_arr:
                return np.zeros((1, 1), dtype=img.dtype)
            roi_img = img[y1_arr:y2_arr, x1:x2]
            return roi_img if roi_img.size else np.zeros((1, 1), dtype=img.dtype)

    def _emit_cropped_images(self, roi_sample: dict[str, Any], prepared: dict[str, Any], x_data: list[float], labels: list[str]) -> None:
        """
        Emit a batch of cropped ROI images to the 'Cropped Images' output.
        Only runs if at least one module is connected to that output.
        """
        conns = self.connections.get("Cropped Images", [])
        if not conns:
            return

        cropped_images = []
        for i, filename in enumerate(self.current_list):
            if filename not in self.image_cache:
                continue
            img = self.image_cache[filename]
            cropped = self._crop_roi(img, prepared)
            cropped_images.append({
                "image": cropped,
                "filename": filename,
                "x": x_data[i] if i < len(x_data) else float(i),
            })

        payload = {
            "roi_uuid": roi_sample.get("uuid"),
            "roi_name": roi_sample.get("name"),
            "images": cropped_images,
            "labels": labels,
        }

        for module in conns:
            if hasattr(module, 'input_cb'):
                try:
                    module.input_cb(data=payload, data_type=IOTypes.FRAME_LIST)
                except Exception as e:
                    logger.error(f"Error emitting cropped images: {e}")

    @staticmethod
    def _prepare_roi_mask(roi_mask: np.ndarray) -> dict[str, Any]:
        """
        Build a reusable descriptor for a mask-based ROI.
        The expensive bounding-box and crop are computed here *once*
        instead of inside the per-image loop.
        """
        # Bounding box of the nonzero region
        rows = np.any(roi_mask, axis=1)
        cols = np.any(roi_mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return {
            "kind":       "mask",
            "mask_full":  roi_mask,
            "mask_crop":  roi_mask[r0:r1+1, c0:c1+1],
            "bbox":       (c0, r0, c1-c0+1, r1-r0+1),  # (x1, y1_arr, w, h)
            "ref_shape":  roi_mask.shape,                # (H, W)
        }

    @staticmethod
    def _prepare_roi_rect(roi_sample: dict[str, Any]) -> dict[str, Any]:
        """Build a reusable descriptor for a rect-based ROI."""
        rx, ry, rw, rh = roi_sample["rect"]
        rx_i, ry_i = int(rx), int(ry)
        rw_i, rh_i = int(rw), int(rh)
        return {
            "kind":      "rect",
            "x1":        max(0, rx_i),
            "x2":        rx_i + rw_i,
            "y_bot":     ry_i,
            "y_top":     ry_i + rh_i,
            "threshold": roi_sample.get("threshold", 100),
        }

    def _roi_worker(self) -> None:
        """Single background worker to process ROIs sequentially, preventing race conditions and order inversion."""
        while True:
            try:
                sample = self.roi_queue.get()
                if sample is None:
                    break
                self._process_roi_batch(sample)
                self.roi_queue.task_done()
            except Exception as e:
                logger.error(f"Error in ROI worker: {e}")

    def _process_roi_batch(self, roi_sample: dict[str, Any]) -> None:
        """
        Process ROI on all images in the current filtered list.
        Calculates mean intensity and outputs a plotting SAMPLE.
        Supports two modes:
          - rect-based: use rect + threshold (legacy)
          - mask-based: use a boolean mask (from FluoSelector contours)
        """
        roi_uuid = roi_sample["uuid"]
        action = roi_sample.get("action")
        
        if action in ["unselect", "delete"]:
            # Remove from active tracking
            roi_data = self.active_rois.pop(roi_uuid, None)
            if action == "delete":
                self._roi_cache.pop(roi_uuid, None)
            
            sample = {
                "uuid": roi_uuid,
                "action": "unselect" # Lineplot uses unselect to remove
            }
            if roi_data and "sample" in roi_data and "name" in roi_data["sample"]:
                sample["name"] = roi_data["sample"]["name"]
            elif "name" in roi_sample:
                sample["name"] = roi_sample["name"]

            self._emit_sample(sample)
            return

        if action == "rename":
            if roi_uuid in self.active_rois:
                self.active_rois[roi_uuid]["sample"]["name"] = roi_sample["name"]
            if roi_uuid in self._roi_cache:
                self._roi_cache[roi_uuid]["sample"]["name"] = roi_sample["name"]
                
            out_sample = {
                "uuid": roi_uuid,
                "name": roi_sample["name"],
                "action": "rename" 
            }
            self._emit_sample(out_sample)
            return

        if not self.current_list:
            logger.warning("No images to process.")
            return

        # Check if already computed and cached for the exact same list
        if action == "select" and roi_uuid in self._roi_cache:
            cached = self._roi_cache[roi_uuid]
            cached_rect = cached["sample"].get("rect")
            req_rect = roi_sample.get("rect")
            if cached.get("list_len") == len(self.current_list) and cached_rect == req_rect:
                self.active_rois[roi_uuid] = cached
                out_sample = {
                    "uuid":   roi_uuid,
                    "name":   roi_sample.get("name", cached["sample"].get("name")),
                    "x":      cached["x"],
                    "y":      cached["y"],
                    "labels": cached["labels"],
                    "action": "select" 
                }
                self._emit_sample(out_sample)
                return

        # ── Pre-compute ROI descriptor ONCE (outside the image loop) ──
        roi_mask = roi_sample.get("mask")  # bool np.ndarray or None
        is_mask_roi = roi_mask is not None and isinstance(roi_mask, np.ndarray)

        if is_mask_roi:
            prepared = self._prepare_roi_mask(roi_mask)
        else:
            prepared = self._prepare_roi_rect(roi_sample)

        x_data = []
        y_data = []

        for i, filename in enumerate(self.current_list):
            if filename not in self.image_cache:
                continue
            img = self.image_cache[filename]
            x_data.append(self._get_x_value(filename, i))
            y_data.append(self._compute_roi_mean(img, prepared))

        # Build labels list
        labels = [f for f in self.current_list if f in self.image_cache]

        # ── Emit cropped ROI images if a downstream module is connected ──
        self._emit_cropped_images(roi_sample, prepared, x_data, labels)
        
        # Store as active ROI for future frames and in cache
        roi_entry = {
            "sample":   roi_sample,
            "prepared": prepared,  # keep descriptor for single-frame updates
            "x":        x_data,
            "y":        y_data,
            "labels":   labels,
            "list_len": len(self.current_list),
        }
        self.active_rois[roi_uuid] = roi_entry
        self._roi_cache[roi_uuid] = roi_entry
        
        out_sample = {
            "uuid":   roi_uuid,
            "name":   roi_sample["name"],
            "x":      x_data,
            "y":      y_data,
            "labels": labels,
            "action": "select" 
        }
        
        self._emit_sample(out_sample)
    
    def _process_active_rois_single_frame(self, filename: str) -> None:
        """Processes a newly added frame against all active ROIs."""
        if filename not in self.image_cache:
            return

        img = self.image_cache[filename]
        idx = len(self.current_list) - 1  # Index of the new frame
        
        for uuid, data in self.active_rois.items():
            # Re-use pre-computed descriptor from _process_roi_batch.
            # Fall back to building it on the fly for ROIs that arrived
            # before this optimisation (e.g. rect ROIs from image_viewer).
            prepared = data.get("prepared")
            if prepared is None:
                sample = data["sample"]
                roi_mask = sample.get("mask")
                if roi_mask is not None and isinstance(roi_mask, np.ndarray):
                    prepared = self._prepare_roi_mask(roi_mask)
                else:
                    prepared = self._prepare_roi_rect(sample)
                data["prepared"] = prepared  # cache for subsequent frames

            mean_val = self._compute_roi_mean(img, prepared)

            x_val = self._get_x_value(filename, idx)
            data["x"].append(x_val)
            data["y"].append(float(mean_val))
            data["list_len"] = len(self.current_list)
            
            if "labels" not in data:
                data["labels"] = []
            data["labels"].append(filename)
            
            sample = data["sample"]
            out_sample = {
                "uuid":   uuid,
                "name":   sample["name"],
                "x":      data["x"],
                "y":      data["y"],
                "labels": data["labels"],
                "action": "select"
            }
            self._emit_sample(out_sample)

    def _emit_folder(self, folder_path: str) -> None:
        for module in self.connections.get("Folder", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(data=folder_path, data_type=IOTypes.FOLDER_PATH)
        
        # Also setup camera to Trigger mode when a folder is selected
        self._emit_cmd({"trigger_mode": "Trigger"})
        
    def _emit_cmd(self, cmd: dict[str, Any]) -> None:
        for module in self.connections.get("CMD", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)

    def _emit_roi(self, sample: dict[str, Any]) -> None:
        for module in self.connections.get("ROI", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)

    def _emit_sample(self, sample: dict[str, Any]) -> None:
        for module in self.connections.get("Result", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(sample=sample)

    def _send_current_framelist(self, *args: Any, **kwargs: Any) -> None:
        """
        Emit the currently displayed image list as a FRAME_LIST to 'Frame List' output.

        Respects the active group filter and filename filter — only images in
        ``self.current_list`` that are present in the cache are included.

        Each element is a dict::

            {
                'image':    np.ndarray,   # raw pixel data from cache
                'filename': str,          # original filename key
                'x':        float,        # X value (index / time / intensity)
            }
        """
        conns = self.connections.get("Frame List", [])
        if not conns:
            logger.debug("ImageContainer: no module connected to 'Frame List' output.")
            return

        if not self.current_list:
            logger.warning("ImageContainer: current list is empty, nothing to send.")
            return

        frame_list = []
        for i, filename in enumerate(self.current_list):
            if filename not in self.image_cache:
                continue
            frame_list.append({
                "image":    self.image_cache[filename],
                "filename": filename,
                "x":        self._get_x_value(filename, i),
            })

        if not frame_list:
            logger.warning("ImageContainer: no cached images in current list.")
            return

        logger.info(f"ImageContainer: sending {len(frame_list)} frame(s) via 'Frame List' output.")
        for module in conns:
            if hasattr(module, 'input_cb'):
                try:
                    module.input_cb(data=frame_list, data_type=IOTypes.FRAME_LIST)
                except Exception as e:
                    logger.error(f"ImageContainer: error sending Frame List to '{getattr(module, 'label', module)}': {e}")


    def clear_list(self, sender: Any = None, app_data: Any = None, user_data: Any = None, cmd: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        """Clears the current image list, cache, and ROIs (notifies ROI Manager)."""
        visited = cmd.get("visited", []) if cmd else []
        if self.UUID not in visited:
            visited = visited + [self.UUID]
            
        # Global clear commands to downstream and upstream
        self._emit_roi({"action": "clear", "clear": True, "visited": visited})
        self._emit_cmd({"clear": True, "visited": visited})
        self._emit_sample({"action": "clear", "clear": True})
        
        # Also remove the drag line on Lineplot if connected
        pos_payload = {"action": "position", "value": None}
        for module in self.connections.get("Result", []):
            if hasattr(module, 'input_cb'):
                try:
                    module.input_cb(data=pos_payload, data_type=IOTypes.CMD_DICT)
                except Exception:
                    pass
            
        self.image_cache = {}
        self.metadata_cache = {}
        self.active_rois = {}  # Fully clear ROIs
        self._roi_cache = {}
        self.frame_counter = 0
        self.current_list = []
        self.groups = ["All"]
        
        # Reset UI
        dpg.configure_item(self.listbox_tag, items=[])
        dpg.set_value(self.count_text_tag, "0 images")
        dpg.configure_item(self.group_combo_tag, items=self.groups)
        dpg.set_value(self.group_combo_tag, "All")
        dpg.set_value(self.filter_input_tag, "")
            
        logger.info("ImageContainer list and ROIs cleared.")

    def _update_list_ui(self) -> None:
        """Refreshes the listbox based on current cache and filters."""
        # Capture current selection or pending group
        current_selection = self.pending_group if self.pending_group else dpg.get_value(self.group_combo_tag)
        
        # Update Groups
        unique_groups = sorted(list(set(m['group'] for m in self.metadata_cache.values() if m['group'] != "Unknown")))
        if "All" not in unique_groups:
            unique_groups.insert(0, "All")
        
        # Only update combo if list changed substantially to avoid flickering reset
        if self.groups != unique_groups:
            self.groups = unique_groups
            dpg.configure_item(self.group_combo_tag, items=self.groups)
            
        # Restore or apply selection
        if current_selection in self.groups:
            dpg.set_value(self.group_combo_tag, current_selection)
        else:
            dpg.set_value(self.group_combo_tag, "All")
            
        self.pending_group = None # Reset pending group

        self._apply_filters()

    def _show_progress(self, text: str) -> None:
        # Center the modal based on viewport size 
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        # Window size is 300x120
        dpg.set_item_pos(self.progress_modal_tag, [(vw - 300) // 2, (vh - 120) // 2])
        
        dpg.configure_item(self.progress_modal_tag, show=True)
        dpg.set_value(self.progress_text_tag, text)
        dpg.set_value(self.progress_bar_tag, 0)
        
    def _update_progress(self, value: float, text: str | None = None) -> None:
        dpg.set_value(self.progress_bar_tag, value)
        if text:
            dpg.set_value(self.progress_text_tag, text)

    def _hide_progress(self) -> None:
        dpg.configure_item(self.progress_modal_tag, show=False)

    def _load_images(self) -> None:
        if not self.folder_path or not os.path.exists(self.folder_path):
            return

        self._show_progress("Indexing files...")
        
        # Clear everything before loading new folder
        self.clear_list()
        
        try:
            files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.png')]
            files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            self._hide_progress()
            return
        
        if not files:
            logger.warning("No PNG images found in folder")
            self._hide_progress()
            return
        
        total = len(files)
        logger.info(f"Loading {total} images from {self.folder_path}...")
        
        for i, f in enumerate(files):
            path = os.path.join(self.folder_path, f)
            try:
                # Load image
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.image_cache[f] = img
                    # Parse metadata
                    self._parse_metadata(f)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                
            if i % 5 == 0:
                progress = i / total
                self._update_progress(progress, f"Loading {i}/{total}")

        self._update_list_ui()
        logger.success(f"Added {len(files)} images. Total: {len(self.image_cache)}")
        self._hide_progress()

    def _parse_metadata(self, filename: str) -> None:
        """
        Parses filename in format: Index_Group_Type_Intensity_Time
        Example: 0_GTEST_F0_A0_2150.png
        Only extracts ID and Group.
        """
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        meta = {
            'id': 0,
            'group': 'Unknown',
            'A': 0
        }
        
        try:
            if len(parts) >= 1:
                # ID
                if parts[0].isdigit():
                    meta['id'] = int(parts[0])
                
                # Group: Remove 'G' prefix if present
                if len(parts) > 1:
                    raw_group = parts[1]
                    if raw_group.startswith('G') and len(raw_group) > 1:
                        meta['group'] = raw_group[1:]
                    else:
                        meta['group'] = raw_group
                        
                # Extract A value via Regex: looks for _A123_ or -A123. or 0_A123
                a_match = re.search(r'[_\-]A(\d+)(?:[_\-]|$)', f"_{name_no_ext}_")
                if a_match:
                    meta['A'] = int(a_match.group(1))
                    
        except Exception:
            pass
            
        self.metadata_cache[filename] = meta

    def _export_experiment_to_file(self, *args: Any, **kwargs: Any) -> None:
        """Ask for a path and save current selection as a pickle bundle."""
        if not self.current_list:
            logger.warning("No images selected for export.")
            return
        
        compression_options = [
            ("Pickle export (single file)", "*.pkl"),
            ("Raw export (ZIP archive)", "*.zip"),
        ]

        folder_name = os.path.basename(self.folder_path) if self.folder_path else "experiment"
        default_name = f"{folder_name}.pkl"

        file_explorer.save_file(
            default_path=self.folder_path, 
            default_name=default_name, 
            extensions=compression_options,
            callback=self._on_export_path_selected
        )

    def _on_export_path_selected(self, path: str | None, selected_ext: str = "", *args: Any, **kwargs: Any) -> None:
        if not path:
            return
            
        if os.path.isdir(path):
            path = os.path.join(path, "experiment")

        path_str = str(path)
        path_lower = path_str.lower()

        if selected_ext == ".zip" or path_lower.endswith(".zip"):
            mode = "raw_zip"
        else:
            mode = "pickle"

        stripped, _ = os.path.splitext(path_str)
        if mode == "raw_zip":
            path = stripped + ".zip"
        else:
            path = stripped + ".pkl"

        mode_labels = {
            "raw_zip": "Raw export (ZIP archive)",
            "pickle": "Pickle export (single file)",
        }
        logger.info(f"Using export mode: {mode_labels.get(mode, mode)}")

        threading.Thread(target=self._export_worker, args=(path,), daemon=True).start()

    def _export_worker(self, path: str) -> None:
        """Heavy lifting for export in a background thread."""
        try:
            total = len(self.current_list)
            logger.debug(f"Starting export of {total} images to {path}")
            self._show_progress(f"Compressing {total} images...")
            
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            mode = 'raw_zip' if path.lower().endswith('.zip') else 'pickle'

            method_map = {
                "raw_zip": "Raw export (ZIP archive)",
                "pickle": "Pickle export (single file)",
            }
            method_name = method_map.get(mode, "Unknown archive")

            cache_size = sum(img.nbytes for img in self.image_cache.values() if isinstance(img, np.ndarray))
            logger.info(f"EXPORT START: {total} images | Method: {method_name} | Raw cache size: {cache_size/1024/1024:.1f} MB -> {path}")

            metadata_snapshot = {f: self.metadata_cache.get(f, {}) for f in self.current_list}
            rois_payload = [data['sample'] for data in self.active_rois.values()]

            sequences_payload = {}
            if self.folder_path and os.path.exists(self.folder_path):
                unique_groups = set(m.get('group', 'Unknown') for m in metadata_snapshot.values() if m.get('group', 'Unknown') != 'Unknown')
                for group_name in unique_groups:
                    json_path = os.path.join(self.folder_path, f"{group_name}.json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                payload_sq = json.load(f)
                                if "sequence" in payload_sq:
                                    sequences_payload[group_name] = payload_sq["sequence"]
                        except Exception as e:
                            logger.warning(f"Failed to load sequence for {group_name}: {e}")

            if mode == 'pickle':
                export_data = {
                    'format': 'pickle_bundle_v1',
                    'current_list': list(self.current_list),
                    'image_cache': {},
                    'metadata_cache': metadata_snapshot,
                    'rois': rois_payload,
                    'sequences': sequences_payload,
                }

                for i, name in enumerate(self.current_list):
                    if name in self.image_cache:
                        export_data['image_cache'][name] = self.image_cache[name]
                    if i % 10 == 0:
                        self._update_progress(i / max(1, total), f"Serialising {i}/{total}")

                self._show_progress("Writing pickle archive...")
                pickled = pickle.dumps(export_data, protocol=pickle.HIGHEST_PROTOCOL)
                with open(path, 'wb') as f:
                    f.write(pickled)
                logger.info(f"Pickle saved ({len(pickled)/1024/1024:.1f} MB)")

            else:
                self._show_progress("Building ZIP archive...")
                archive_map = {}
                missing_on_disk = []
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                    for i, name in enumerate(self.current_list):
                        archive_name = f"images/{os.path.basename(name)}"
                        data_bytes = None

                        source_path = os.path.join(self.folder_path or '', name)
                        if os.path.isfile(source_path):
                            try:
                                with open(source_path, 'rb') as img_file:
                                    data_bytes = img_file.read()
                            except Exception as e:
                                logger.warning(f"Failed to read {source_path}: {e}")

                        if data_bytes is None:
                            img = self.image_cache.get(name)
                            if img is None:
                                missing_on_disk.append(name)
                                continue
                            success, buffer = cv2.imencode('.png', img)
                            if not success:
                                raise RuntimeError(f"Failed to encode image {name} as PNG")
                            archive_name = f"images/{os.path.splitext(os.path.basename(name))[0]}.png"
                            data_bytes = buffer.tobytes()

                        z.writestr(archive_name.replace('\\', '/'), data_bytes)
                        archive_map[name] = archive_name.replace('\\', '/')

                        if i % 10 == 0:
                            self._update_progress(i / max(1, total), f"Archiving {i}/{total}")

                    metadata_payload = {
                        "format": "raw_zip_v1",
                        "exported_at": time.time(),
                        "current_list": list(self.current_list),
                        "metadata": metadata_snapshot,
                        "sequences": sequences_payload,
                        "archive_map": archive_map,
                        "rois_pickle": "rois.pkl" if rois_payload else None,
                    }

                    z.writestr("metadata.json", json.dumps(metadata_payload, ensure_ascii=False, indent=2))

                    if rois_payload:
                        z.writestr("rois.pkl", pickle.dumps(rois_payload, protocol=pickle.HIGHEST_PROTOCOL))

                if missing_on_disk:
                    logger.warning(f"{len(missing_on_disk)} images were encoded from cache because they were missing on disk.")

                zip_size = os.path.getsize(path)
                logger.info(f"ZIP archive created ({zip_size/1024/1024:.1f} MB)")

            logger.success(f"Export finished: {path}")
        except Exception as e:
            logger.error(f"Failed to export experiment: {e}")
        finally:
            self._hide_progress()

    def _import_experiment_from_file(self, sender: Any = None, path: str | None = None, *args: Any, **kwargs: Any) -> None:
        """Ask for an export archive and import its content."""
        if path is None:
            import_options = [
                ("Raw export (ZIP archive)", "*.zip"),
                ("Pickle export (single file)", "*.pkl"),
            ]

            file_explorer.select_file(
                default_path=self.folder_path, 
                extensions=import_options,
                callback=self._on_import_file_selected
            )
        else:
            self._on_import_file_selected(path)

    def _on_import_file_selected(self, path: str | None, *args: Any, **kwargs: Any) -> None:
        if not path or not os.path.exists(path):
            return
        
        path_lower = path.lower()
        is_pickle = path_lower.endswith('.pkl')
        is_raw_zip = path_lower.endswith('.zip')

        if not (is_pickle or is_raw_zip):
            logger.error(f"{path} is not a supported export file (expected .zip or .pkl).")
            return
            
        self._emit_folder(self.folder_path)
        threading.Thread(target=self._import_worker, args=(path,), daemon=True).start()

    def _import_worker(self, path: str) -> None:
        """Heavy lifting for import in a background thread."""
        try:
            self._show_progress("Extracting experiment data...")
            images = {}
            metadata = {}
            rois = []
            sequences = {}

            path_lower = path.lower()
            if path_lower.endswith('.pkl'):
                logger.info("Importing from Pickle archive")
                with open(path, 'rb') as f:
                    pickled_data = f.read()
                export_data = pickle.loads(pickled_data)
                images = export_data.get('image_cache', {})
                metadata = export_data.get('metadata_cache', {})
                rois = export_data.get('rois', [])
                sequences = export_data.get('sequences', {})
                self._update_progress(1.0, "Pickle extracted")
            elif path_lower.endswith('.zip'):
                logger.info("Importing from raw ZIP archive")
                with zipfile.ZipFile(path, 'r') as z:
                    names = z.namelist()
                    if "metadata.json" not in names:
                        raise ValueError("Missing metadata.json in ZIP archive")

                    metadata_payload = json.loads(z.read("metadata.json").decode('utf-8'))
                    metadata = metadata_payload.get("metadata", {})
                    sequences = metadata_payload.get("sequences", {})
                    archive_map = metadata_payload.get("archive_map") or {
                        os.path.basename(n): n for n in names if n.startswith("images/") and not n.endswith("/")
                    }

                    rois_pickle_name = metadata_payload.get("rois_pickle")
                    if rois_pickle_name and rois_pickle_name in names:
                        try:
                            rois = pickle.loads(z.read(rois_pickle_name))
                        except Exception as e:
                            logger.warning(f"Failed to restore ROIs from {rois_pickle_name}: {e}")
                            rois = []
                    else:
                        rois = metadata_payload.get("rois", [])

                    for original_name, archive_name in archive_map.items():
                        normalized = archive_name.replace('\\', '/')
                        if normalized not in names:
                            logger.warning(f"Archive entry {archive_name} missing for image {original_name}")
                            continue
                        data = z.read(normalized)
                        nparr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            logger.warning(f"Failed to decode image {original_name} from archive entry {normalized}")
                            continue
                        images[original_name] = img

                    if not images:
                        raise ValueError("Archive contains no images")

                    self._update_progress(1.0, "ZIP archive extracted")
                    logger.debug(f"ZIP metadata restored with {len(images)} images")
            else:
                raise ValueError("Unsupported archive format")

            if images:
                self._import_experiment({"images": images, "metadata": metadata, "rois": rois, "sequences": sequences})
                logger.success(f"Imported {len(images)} images and {len(rois)} ROIs from {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Failed to import archive: {e}")
        finally:
            self._hide_progress()

    def _import_experiment(self, experiment_data: dict[str, Any]) -> None:
        """Receives images, metadata and ROIs and adds them to the cache."""
        images = experiment_data.get("images", {})
        metadata = experiment_data.get("metadata", {})
        rois = experiment_data.get("rois", [])
        sequences = experiment_data.get("sequences", {})
        
        if sequences and self.folder_path:
            os.makedirs(self.folder_path, exist_ok=True)
            for group, seq in sequences.items():
                json_path = os.path.join(self.folder_path, f"{group}.json")
                try:
                    with open(json_path, 'w') as f:
                        json.dump({"sequence": seq}, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to restore sequence JSON for {group}: {e}")
                    
        if not images:
            logger.warning("Import received no images.")
            return

        # Clear everything before importing new experiment
        self.clear_list()

        for name, img in images.items():
            self.image_cache[name] = img
            if name in metadata:
                self.metadata_cache[name] = metadata[name]
        
        # Restore ROIs
        for roi_sample in rois:
            # Force 'create' action so ROI Manager adds it to its list
            sample_copy = roi_sample.copy()
            sample_copy["action"] = "create"
            self._emit_roi(sample_copy)
            
            # Internal re-processing
            self._process_roi_batch(roi_sample)
        
        self._update_list_ui()
        logger.success(f"Imported {len(images)} images.")

    def _apply_filters(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        selected_group = dpg.get_value(self.group_combo_tag)
        filter_text = dpg.get_value(self.filter_input_tag) # General Filter Text
        
        should_emit_fo_fm = False
        # If the combo box triggered this, emit the group change/clear command
        if sender == self.group_combo_tag and selected_group:
            # Reload config dynamically
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        user_config = json.load(f)
                        for k in ["emit_clear_on_group_selection", "emit_clear_on groupe_selection"]:
                            if k in user_config:
                                self.config["emit_clear_on_group_selection"] = bool(user_config[k])
                        if "emit_fo_fm_on_group_selection" in user_config:
                            self.config["emit_fo_fm_on_group_selection"] = bool(user_config["emit_fo_fm_on_group_selection"])
                except Exception as e:
                    logger.error(f"Error reloading config.json for ImageContainer: {e}")

            # Emit CMD for clear/reset downstream modules
            if self.config.get("emit_clear_on_group_selection", True):
                self._emit_cmd({"action": "clear", "group": selected_group})
            
            if selected_group != "All":
                targets = self.connections.get("Group", [])
                logger.info(f"Group filter changed to '{selected_group}', emitting to {len(targets)} targets")
                for module in targets:
                    if hasattr(module, 'input_cb'):
                        try:
                            module.input_cb(data=selected_group, data_type=IOTypes.TEXT)
                        except Exception as e:
                            logger.error(f"Error emitting group from filter: {e}")
                            
            if self.config.get("emit_fo_fm_on_group_selection", False):
                should_emit_fo_fm = True

        
        filtered = []
        
        # Filter based on the sorted keys of image_cache (already natsorted or insertion order)
        for f in self.image_cache.keys():
            meta = self.metadata_cache.get(f, {})
            
            # Group Filter
            if selected_group != "All" and selected_group is not None:
                if meta.get('group', 'Unknown') != selected_group:
                    continue
            
            # General Filename Filter (Contains)
            if filter_text:
                # Case insensitive substring match against FILENAME
                if filter_text.lower() not in f.lower():
                    continue
            
            filtered.append(f)
            
        self.current_list = filtered
        dpg.configure_item(self.listbox_tag, items=self.current_list)
        dpg.set_value(self.count_text_tag, f"{len(filtered)} images")
        
        if should_emit_fo_fm and selected_group != "All":
            if len(self.current_list) >= 2:
                # Select first image
                dpg.set_value(self.listbox_tag, self.current_list[0])
                self._on_list_select(None, self.current_list[0])
                
                # Small delay then select second image
                def select_second():
                    dpg.set_value(self.listbox_tag, self.current_list[1])
                    self._on_list_select(None, self.current_list[1])
                threading.Timer(0.1, select_second).start()
                logger.info(f"Auto-selected first two images of group '{selected_group}': {self.current_list[0]} and {self.current_list[1]}")

    def _on_list_select(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if not app_data: 
            logger.warning("No app_data received in _on_list_select")
            return
        
        if app_data not in self.image_cache:
            logger.warning(f"'{app_data}' not found in image_cache")

        if app_data in self.image_cache:
            img = self.image_cache[app_data]
            # Send to frame output
            for module in self.connections.get("Frame", []):
                try:
                    if hasattr(module, 'input_cb'):
                        module.input_cb(img)
                except Exception as e:
                    logger.error(f"Error sending frame to {module.label}: {e}")
            
            # Send to image name output
            for module in self.connections.get("Image Name", []):
                try:
                    if hasattr(module, 'input_cb'):
                        module.input_cb(data=app_data, data_type=IOTypes.TEXT)
                except Exception as e:
                    logger.error(f"Error sending image name to {module.label}: {e}")
            
            # Emit position (index in current list) via CMD output
            if app_data in self.current_list:
                idx = self.current_list.index(app_data)
                x_val = self._get_x_value(app_data, idx)
                self._emit_cmd({"action": "position", "value": x_val})
                
                # Also emit to Result connection to update the drag line on Lineplot if connected
                pos_payload = {"action": "position", "value": x_val}
                for module in self.connections.get("Result", []):
                    if hasattr(module, 'input_cb'):
                        try:
                            module.input_cb(data=pos_payload, data_type=IOTypes.CMD_DICT)
                        except Exception as e:
                            logger.error(f"Error sending position to Result connection: {e}")
            else:
                logger.warning(f"'{app_data}' not in current_list? (List has {len(self.current_list)} items)")

    def _start_playback(self, *args: Any, **kwargs: Any) -> None:
        if self.playback_active:
            return
            
        if not self.current_list:
            return
            
        self.playback_active = True
        self.stop_playback_flag = False
        
        def playback_task() -> None:
            fps = dpg.get_value(self.fps_drag_tag)
            if fps <= 0: fps = 1
            delay = 1.0 / fps
            
            # Start from ZERO (beginning)
            start_index = 0
                
            for i in range(start_index, len(self.current_list)):
                if self.stop_playback_flag:
                    break
                
                filename = self.current_list[i]
                
                # Update UI selection
                dpg.set_value(self.listbox_tag, filename)
                
                # Send Data
                self._on_list_select(None, filename)
                
                # Dynamic FPS update
                fps = dpg.get_value(self.fps_drag_tag)
                delay = 1.0 / max(1, fps)
                
                time.sleep(delay)
            
            self.playback_active = False

        self.playback_thread = threading.Thread(target=playback_task, daemon=True)
        self.playback_thread.start()

    def _stop_playback(self, *args: Any, **kwargs: Any) -> None:
        self.stop_playback_flag = True

    def is_ready(self) -> bool:
        """Indicate whether the module is initialized and ready to operate."""
        return not getattr(self, "_is_closing", False)

    def on_close(self) -> None:
        """Release resources, terminate playback thread, and signal background workers on closure."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self._stop_playback()
        if hasattr(self, "roi_queue"):
            try:
                self.roi_queue.put(None)
            except Exception:
                pass
        if hasattr(self, "playback_thread") and self.playback_thread and self.playback_thread.is_alive():
            try:
                self.playback_thread.join(timeout=0.5)
            except Exception:
                pass

    def close(self) -> None:
        """Close window and unregister."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()
        super().destroy()


ImageContainer_win = ImageContainerWin
EXPORTED_CLASS = ImageContainerWin
EXPORTED_NAME = "Image Container"
