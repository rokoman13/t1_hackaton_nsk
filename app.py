import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        _gpu_test = torch.randn(10, 10).cuda()
        torch.cuda.synchronize()
        del _gpu_test
        torch.cuda.empty_cache()
        print("CUDA контекст активирован!")
    except Exception as e:
        print(f"Ошибка активации CUDA: {e}")

import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
from PIL import Image, ImageDraw, ImageFont
import json
import queue
import time
import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from torch.hub import load

st.set_page_config(
    page_title="Video Segmentation with Overlay", 
    page_icon="🎥",
    layout="wide"
)

os.makedirs("backgrounds", exist_ok=True)
os.makedirs("data", exist_ok=True)

global_employee_data = None
global_overlay_settings = None

class HighQualitySegmentator:
    def __init__(self, model_type="modnet"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.transform = None
        
        if model_type == "modnet":
            self._init_modnet()
        elif model_type == "u2net":
            self._init_u2net()
        elif model_type == "briaai":
            self._init_briaai()
        else:
            self._init_modnet()
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_mask = None
        self.smooth_alpha = 0.2

    def _init_modnet(self):
        try:
            self.model = torch.hub.load('ZHKKKe/MODNet', 'modnet_photographic', pretrained=True, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Ошибка загрузки MODNet: {e}")
            self._init_mediapipe_fallback()

    def _init_u2net(self):
        try:
            self.model = torch.hub.load('xuebinqin/U-2-Net', 'u2net', pretrained=True, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Ошибка загрузки U2-Net: {e}")
            self._init_modnet()

    def _init_briaai(self):
        try:
            self.model = torch.hub.load('briaai/RMBG-1.4', 'RMBG-1.4', pretrained=True, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        except Exception as e:
            print(f"Ошибка загрузки RMBG-1.4: {e}")
            self._init_modnet()

    def _init_mediapipe_fallback(self):
        self.model = None
        self.model_type = "mediapipe"

    def segment_frame(self, frame):
        try:
            if self.model is not None and self.model_type != "mediapipe":
                return self._neural_segmentation(frame)
            else:
                return self._mediapipe_segmentation(frame)
                
        except Exception as e:
            print(f"Ошибка сегментации: {e}")
            return self._quick_fallback(frame)

    def _neural_segmentation(self, frame):
        with torch.no_grad():
            input_tensor, original_size = self._preprocess_frame(frame)
            
            if self.model_type == "modnet":
                _, _, matte = self.model(input_tensor, True)
                mask = self._postprocess_modnet(matte, original_size)
            elif self.model_type == "u2net":
                matte = self.model(input_tensor)
                mask = self._postprocess_u2net(matte[0], original_size)
            elif self.model_type == "briaai":
                matte = self.model(input_tensor)
                mask = self._postprocess_briaai(matte, original_size)
            else:
                mask = self._mediapipe_segmentation(frame)
            
            refined_mask = self._refine_mask(mask)
            smoothed_mask = self._temporal_smooth(refined_mask)
            
            return smoothed_mask

    def _preprocess_frame(self, frame):
        h, w = frame.shape[:2]
        original_size = (w, h)
        
        if self.model_type == "modnet":
            target_size = (512, 512)
        elif self.model_type == "u2net":
            target_size = (320, 320)
        else:
            target_size = (1024, 1024)
        
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h
        scale = min(scale_x, scale_y)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_frame = cv2.resize(frame, (new_w, new_h))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        
        return tensor, original_size

    def _postprocess_modnet(self, matte, original_size):
        matte = matte.squeeze().cpu().numpy()
        w, h = original_size
        mask = (cv2.resize(matte, (w, h)) * 255).astype(np.uint8)
        return mask

    def _postprocess_u2net(self, matte, original_size):
        matte = matte.squeeze().cpu().numpy()
        w, h = original_size
        matte = (matte - matte.min()) / (matte.max() - matte.min() + 1e-8)
        mask = (cv2.resize(matte, (w, h)) * 255).astype(np.uint8)
        return mask

    def _postprocess_briaai(self, matte, original_size):
        matte = matte.squeeze().cpu().numpy()
        w, h = original_size
        mask = (cv2.resize(matte, (w, h)) * 255).astype(np.uint8)
        return mask

    def _refine_mask(self, mask):
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 1)
        kernel = np.ones((3, 3), np.uint8)
        mask_closed = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) > 1000:
                refined_mask = np.zeros_like(mask_opened)
                cv2.fillPoly(refined_mask, [main_contour], 255)
                return refined_mask
        
        return mask_opened

    def _mediapipe_segmentation(self, frame):
        try:
            h, w = frame.shape[:2]
            
            if w > 320:
                scale = 320 / w
                new_w, new_h = 320, int(h * scale)
                frame_small = cv2.resize(frame, (new_w, new_h))
            else:
                frame_small = frame
            
            rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            results_seg = self.selfie_segmentation.process(rgb_frame)
            if results_seg.segmentation_mask is not None:
                mask_small = (results_seg.segmentation_mask * 255).astype(np.uint8)
                
                if w > 320:
                    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask = mask_small
                    
                return mask
                
        except Exception as e:
            print(f"MediaPipe segmentation error: {e}")
        
        return self._quick_fallback(frame)

    def _quick_fallback(self, frame):
        h, w = frame.shape[:2]
        
        if w > 320:
            small_frame = cv2.resize(frame, (320, int(h * 320/w)))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            for (x, y, fw, fh) in faces:
                if w > 320:
                    scale = w / 320
                    x, y, fw, fh = int(x*scale), int(y*scale), int(fw*scale), int(fh*scale)
                center_x, center_y = x + fw//2, y + fh//2
                cv2.ellipse(mask, (center_x, center_y), (fw, fh), 0, 0, 360, 255, -1)
            
            mask = cv2.GaussianBlur(mask, (9, 9), 2)
            return mask
        
        return self._create_fallback_mask(frame.shape)

    def _create_fallback_mask(self, shape):
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        cv2.ellipse(mask, (center_x, center_y), (w//6, h//4), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 3)
        return mask

    def _temporal_smooth(self, current_mask):
        if self.last_mask is None:
            self.last_mask = current_mask
            return current_mask
        
        current_float = current_mask.astype(np.float32) / 255.0
        last_float = self.last_mask.astype(np.float32) / 255.0
        smoothed = (self.smooth_alpha * last_float + (1 - self.smooth_alpha) * current_float)
        result = (smoothed * 255).astype(np.uint8)
        self.last_mask = result
        return result

    def cleanup(self):
        if hasattr(self, 'selfie_segmentation'):
            self.selfie_segmentation.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TextOverlayGenerator:
    def __init__(self):
        self.available_fonts = self._get_available_fonts()
    
    def _get_available_fonts(self):
        fonts = []
        font_names = [
            "arial.ttf", "arialbd.ttf", 
            "times.ttf", "timesbd.ttf",
            "verdana.ttf", "verdanab.ttf",
        ]
        
        for font_name in font_names:
            try:
                font = ImageFont.truetype(font_name, 12)
                fonts.append(font_name)
            except:
                continue
        
        return fonts if fonts else ["arial.ttf"]

    def add_text_overlay(self, frame, employee_data, overlay_settings):
        try:
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            
            employee = employee_data.get('employee', {})
            contact = employee_data.get('contact', {})
            branding = employee_data.get('branding', {})
            
            font_name = overlay_settings.get('font', 'arialbd.ttf')
            text_color = overlay_settings.get('color', '#FFFFFF')
            position = overlay_settings.get('position', 'bottom_right')
            show_background = overlay_settings.get('show_background', True)
            bg_opacity = overlay_settings.get('bg_opacity', 80)
            
            text_size_scale = overlay_settings.get('text_size_scale', 100) / 100.0
            
            h, w = frame.shape[:2]
            base_size = max(24, min(w, h) // 20) * text_size_scale
            
            font_sizes = {
                'name': int(base_size * 1.6),
                'position': int(base_size * 1.3),
                'company': int(base_size * 1.1),
                'department': int(base_size * 1.0),
                'phone': int(base_size * 0.9),
                'email': int(base_size * 0.9),
                'slogan': int(base_size * 0.8)
            }
            
            fonts = {}
            for key, size in font_sizes.items():
                try:
                    fonts[key] = ImageFont.truetype(font_name, int(size))
                except:
                    try:
                        fonts[key] = ImageFont.truetype("arial.ttf", int(size))
                    except:
                        fonts[key] = ImageFont.load_default()
            
            # Тексты для отображения
            texts_to_display = []
            if overlay_settings.get('show_name', True):
                texts_to_display.append((employee.get('full_name', 'Иван Иванов'), fonts['name']))
            if overlay_settings.get('show_position', True):
                texts_to_display.append((employee.get('position', 'Должность'), fonts['position']))
            if overlay_settings.get('show_company', True):
                texts_to_display.append((employee.get('company', 'Компания'), fonts['company']))
            if overlay_settings.get('show_department', False):
                texts_to_display.append((employee.get('department', 'Отдел'), fonts['department']))
            if overlay_settings.get('show_phone', True):
                texts_to_display.append((contact.get('phone', '+7 (999) 123-45-67'), fonts['phone']))
            if overlay_settings.get('show_email', False):
                texts_to_display.append((contact.get('email', 'email@example.com'), fonts['email']))
            if overlay_settings.get('show_slogan', True):
                texts_to_display.append((branding.get('slogan', 'Слоган компании'), fonts['slogan']))
            
            if not texts_to_display:
                return frame
            
            max_text_width = 0
            total_height = 0
            text_heights = []
            
            for text, font in texts_to_display:
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    text_width = len(text) * font_sizes['name'] // 2
                    text_height = font_sizes['name'] + 10
                
                max_text_width = max(max_text_width, text_width)
                text_heights.append(text_height)
                total_height += text_height + 15
            
            # Отступы
            padding_scale = overlay_settings.get('padding_scale', 100) / 100.0
            padding = int(25 * padding_scale)
            bg_width = max_text_width + padding * 2
            bg_height = total_height + padding
            
            # Позишн
            if position == 'top_left':
                bg_x, bg_y = padding, padding
            elif position == 'top_right':
                bg_x, bg_y = max(padding, w - bg_width - padding), padding
            elif position == 'bottom_left':
                bg_x, bg_y = padding, max(padding, h - bg_height - padding)
            elif position == 'bottom_center':
                bg_x, bg_y = max(padding, (w - bg_width) // 2), max(padding, h - bg_height - padding)
            else:  # bottom_right
                bg_x, bg_y = max(padding, w - bg_width - padding), max(padding, h - bg_height - padding)
            
            # текст бля фон
            if show_background:
                try:
                    bg_color = (0, 0, 0, int(255 * bg_opacity / 100))
                    overlay_bg = Image.new('RGBA', (bg_width, bg_height), bg_color)
                    pil_img_rgba = pil_img.convert('RGBA')
                    pil_img_rgba.paste(overlay_bg, (bg_x, bg_y), overlay_bg)
                    pil_img = pil_img_rgba.convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                except Exception as e:
                    print(f"Ошибка создания фона: {e}")
            
            rgb_color = self.hex_to_rgb(text_color)
            current_y = bg_y + padding // 2
            
            for i, (text, font) in enumerate(texts_to_display):
                text_x = bg_x + padding
                
                if overlay_settings.get('text_shadow', True):
                    shadow_color = (0, 0, 0)
                    for dx in [-2, 2]:
                        for dy in [-2, 2]:
                            try:
                                draw.text((text_x+dx, current_y+dy), text, font=font, fill=shadow_color)
                            except:
                                pass
                
                # Основной текст
                try:
                    draw.text((text_x, current_y), text, font=font, fill=rgb_color)
                except:
                    try:
                        draw.text((text_x, current_y), text, fill=rgb_color)
                    except:
                        pass
                
                current_y += text_heights[i] + 15
            
            return np.array(pil_img)
            
        except Exception as e:
            print(f"Ошибка добавления текста: {e}")
            return frame
    
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class SegmentationVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.segmentator = HighQualitySegmentator(model_type="modnet")
        self.text_overlay = TextOverlayGenerator()
        self.current_background = self._load_background()
        
        global global_employee_data, global_overlay_settings
        self.employee_data = global_employee_data
        self.overlay_settings = global_overlay_settings

    def _load_background(self):
        custom_bg_path = os.path.join("backgrounds", "custom_background.png")
        default_bg_path = os.path.join("backgrounds", "current_background.png")
        
        if os.path.exists(custom_bg_path):
            return Image.open(custom_bg_path)
        elif os.path.exists(default_bg_path):
            return Image.open(default_bg_path)
        else:
            return Image.new('RGB', (640, 480), color=(0, 82, 204))

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        # СЕГМЕНТАЦИЯ
        mask = self.segmentator.segment_frame(img)
        
        # Смешивание с фоном
        if self.current_background:
            bg_resized = self.current_background.resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS)
            bg_array = np.array(bg_resized)
            
            mask_normalized = mask.astype(np.float32) / 255.0
            mask_3d = mask_normalized[:, :, np.newaxis]
            
            result = img.astype(np.float32) * mask_3d + bg_array.astype(np.float32) * (1 - mask_3d)
            result = result.astype(np.uint8)
        else:
            result = img
        
        # Добавляем текст
        if self.employee_data is not None and self.overlay_settings is not None:
            result = self.text_overlay.add_text_overlay(result, self.employee_data, self.overlay_settings)
        
        return av.VideoFrame.from_ndarray(result, format="rgb24")

    def __del__(self):
        self.segmentator.cleanup()

class NormalVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.text_overlay = TextOverlayGenerator()
        
        global global_employee_data, global_overlay_settings
        self.employee_data = global_employee_data
        self.overlay_settings = global_overlay_settings
        
    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        if self.employee_data is not None and self.overlay_settings is not None:
            img = self.text_overlay.add_text_overlay(img, self.employee_data, self.overlay_settings)
        
        return av.VideoFrame.from_ndarray(img, format="rgb24")

def segmentation_video_processor_factory():
    return SegmentationVideoProcessor()

def normal_video_processor_factory():
    return NormalVideoProcessor()

def save_employee_data(data):
    try:
        filepath = os.path.join("data", "employee_data.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

def load_employee_data():
    try:
        filepath = os.path.join("data", "employee_data.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    
    return {
        "employee": {
            "full_name": "Иван Иванов",
            "position": "Ведущий инженер", 
            "company": "ООО «Рога и Копыта»",
            "department": "Департамент разработки",
            "location": "Москва",
            "contact": {
                "email": "ivan@company.ru",
                "phone": "+7 (999) 123-45-67",
                "telegram": "@ivanov"
            }
        },
        "branding": {
            "logo_url": "https://example.com/logo.png",
            "corporate_colors": {
                "primary": "#0052CC",
                "secondary": "#00B8D9"
            },
            "slogan": "Инновации в каждый кадр"
        },
        "privacy_level": "medium"
    }

def get_default_overlay_settings():
    return {
        'font': 'arialbd.ttf',
        'color': '#FFFFFF',
        'position': 'bottom_right',
        'show_background': True,
        'bg_opacity': 80,
        'text_shadow': True,
        'text_size_scale': 100,
        'padding_scale': 100,
        'show_name': True,
        'show_position': True,
        'show_company': True,
        'show_department': False,
        'show_phone': True,
        'show_email': False,
        'show_slogan': True
    }

def main():
    st.title("🎥 Видеосегментация")
    
    # Инициализация session_state
    if 'employee_data' not in st.session_state:
        st.session_state.employee_data = load_employee_data()
    if 'use_segmentation' not in st.session_state:
        st.session_state.use_segmentation = True
    if 'video_key' not in st.session_state:
        st.session_state.video_key = 0
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "modnet"
    if 'overlay_settings' not in st.session_state:
        st.session_state.overlay_settings = get_default_overlay_settings()

    global global_employee_data, global_overlay_settings
    global_employee_data = st.session_state.employee_data
    global_overlay_settings = st.session_state.overlay_settings

    st.header("📹 Видеопоток")
    
    if st.session_state.use_segmentation:
        video_processor_factory = segmentation_video_processor_factory
        status_text = f"Сегментация включена ({st.session_state.selected_model.upper()})"
        status_color = "success"
    else:
        video_processor_factory = normal_video_processor_factory
        status_text = "Оригинальное видео"
        status_color = "info"
    
    st.write(f":{status_color}[{status_text}]")

    # Создаем контекст WebRTC
    webrtc_ctx = webrtc_streamer(
        key=f"video-{st.session_state.video_key}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=video_processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    with st.sidebar:
        st.header("⚙️ Управление")
        
        st.subheader("Режим видео")
        new_segmentation_state = st.toggle(
            "Включить сегментацию", 
            value=st.session_state.use_segmentation,
            key="segmentation_toggle"
        )
        
        if new_segmentation_state != st.session_state.use_segmentation:
            st.session_state.use_segmentation = new_segmentation_state
            st.session_state.video_key += 1
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("🧠 Модель сегментации")
        
        model_options = {
            "MODNet": "modnet",
            "U²-Net": "u2net", 
            "RMBG-1.4": "briaai"
        }
        
        selected_model_name = st.selectbox(
            "Выберите модель:",
            options=list(model_options.keys()),
            index=list(model_options.values()).index(st.session_state.selected_model)
        )
        
        selected_model = model_options[selected_model_name]
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.video_key += 1
            st.success(f"Модель изменена на {selected_model_name}!")
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("🖼️ Фон")
        
        uploaded_file = st.file_uploader("Загрузить свой фон", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                bg_path = os.path.join("backgrounds", "custom_background.png")
                image.save(bg_path)
                st.session_state.video_key += 1
                st.success("Фон загружен! Видео обновляется...")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
        
        custom_bg_path = os.path.join("backgrounds", "custom_background.png")
        if os.path.exists(custom_bg_path):
            if st.button("Удалить загруженный фон"):
                os.remove(custom_bg_path)
                st.session_state.video_key += 1
                st.success("Фон удален! Видео обновляется...")
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📝 Текстовый оверлей")
        
        with st.form("overlay_settings_form"):
            overlay_settings = st.session_state.overlay_settings.copy()
            employee_data = st.session_state.employee_data.get('employee', {})
            contact = employee_data.get('contact', {})
            branding = st.session_state.employee_data.get('branding', {})
            
            st.write("**Отображаемые поля:**")
            col1, col2 = st.columns(2)
            with col1:
                show_name = st.checkbox("ФИО", value=overlay_settings['show_name'], key="show_name")
                show_position = st.checkbox("Должность", value=overlay_settings['show_position'], key="show_position")
                show_company = st.checkbox("Компания", value=overlay_settings['show_company'], key="show_company")
                show_department = st.checkbox("Отдел", value=overlay_settings['show_department'], key="show_department")
            with col2:
                show_phone = st.checkbox("Телефон", value=overlay_settings['show_phone'], key="show_phone")
                show_email = st.checkbox("Email", value=overlay_settings['show_email'], key="show_email")
                show_slogan = st.checkbox("Слоган", value=overlay_settings['show_slogan'], key="show_slogan")
            
            st.write("**Внешний вид:**")
            text_overlay_generator = TextOverlayGenerator()
            available_fonts = text_overlay_generator.available_fonts
            
            selected_font = st.selectbox(
                "Шрифт", 
                available_fonts,
                index=available_fonts.index(overlay_settings['font']) if overlay_settings['font'] in available_fonts else 0,
                key="font_select"
            )
            
            text_color = st.color_picker("Цвет текста", overlay_settings['color'], key="text_color")
            position = st.selectbox(
                "Позиция оверлея",
                ["top_left", "top_right", "bottom_left", "bottom_center", "bottom_right"],
                index=["top_left", "top_right", "bottom_left", "bottom_center", "bottom_right"].index(overlay_settings['position']),
                key="position_select"
            )
            
            show_background = st.checkbox("Показывать фон оверлея", value=overlay_settings['show_background'], key="show_bg")
            bg_opacity = st.slider("Прозрачность фона (%)", 0, 100, overlay_settings['bg_opacity'], key="bg_opacity")
            text_shadow = st.checkbox("Тень текста", value=overlay_settings['text_shadow'], key="text_shadow")
            
            st.write("**Размеры:**")
            text_size_scale = st.slider(
                "Размер текста (%)", 
                50, 200, 
                overlay_settings['text_size_scale'], 
                key="text_size_scale"
            )
            padding_scale = st.slider(
                "Размер отступов (%)", 
                50, 200, 
                overlay_settings['padding_scale'], 
                key="padding_scale"
            )
            
            st.write("**Данные для отображения:**")
            full_name = st.text_input("ФИО", value=employee_data.get('full_name', ''), key="full_name_input")
            position_text = st.text_input("Должность", value=employee_data.get('position', ''), key="position_input")
            company = st.text_input("Компания", value=employee_data.get('company', ''), key="company_input")
            department = st.text_input("Отдел", value=employee_data.get('department', ''), key="department_input")
            phone = st.text_input("Телефон", value=contact.get('phone', ''), key="phone_input")
            email = st.text_input("Email", value=contact.get('email', ''), key="email_input")
            slogan = st.text_input("Слоган", value=branding.get('slogan', ''), key="slogan_input")
            
            submitted = st.form_submit_button("Применить настройки оверлея")
            
            if submitted:
                new_overlay_settings = {
                    'font': selected_font,
                    'color': text_color,
                    'position': position,
                    'show_background': show_background,
                    'bg_opacity': bg_opacity,
                    'text_shadow': text_shadow,
                    'text_size_scale': text_size_scale,
                    'padding_scale': padding_scale,
                    'show_name': show_name,
                    'show_position': show_position,
                    'show_company': show_company,
                    'show_department': show_department,
                    'show_phone': show_phone,
                    'show_email': show_email,
                    'show_slogan': show_slogan
                }
                
                st.session_state.overlay_settings = new_overlay_settings
                
                new_data = {
                    "employee": {
                        "full_name": full_name,
                        "position": position_text,
                        "company": company,
                        "department": department,
                        "location": employee_data.get('location', ''),
                        "contact": {
                            "email": email,
                            "phone": phone,
                            "telegram": contact.get('telegram', '')
                        }
                    },
                    "branding": {
                        "corporate_colors": branding.get('corporate_colors', {}),
                        "slogan": slogan
                    }
                }
                
                st.session_state.employee_data = new_data
                save_employee_data(new_data)
                
                # Обновляем глобальные переменные
                global_employee_data = st.session_state.employee_data
                global_overlay_settings = st.session_state.overlay_settings
                
                st.session_state.video_key += 1
                st.success("Настройки оверлея обновлены!")
                st.rerun()

        st.markdown("---")
        st.subheader("📊 Система")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.success(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            st.warning("GPU не доступен, используется CPU")

if __name__ == "__main__":
    main()