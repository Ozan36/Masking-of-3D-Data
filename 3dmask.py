# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy import ndimage
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSlider, QComboBox, QLabel, QPushButton, QCheckBox
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle

class NDVIAnalyzer:
    def __init__(self, data):
        self.data = data
        self.red = data[:, :, 97].astype(np.float32)
        self.nir = data[:, :, 165].astype(np.float32)
        self.green = data[:, :, 56].astype(np.float32)  # Green band ekle
        self.ndvi = (self.nir - self.red) / (self.nir + self.red + 1e-6)

    def create_vegetation_mask(self):
        ndvi_threshold = 0.5
        nir_threshold = 0.15
        red_threshold = 0.3
        vegetation_mask = (
            (self.ndvi > ndvi_threshold) &
            (self.nir > nir_threshold) &
            (self.red < red_threshold)
        )
        vegetation_mask = ndimage.binary_opening(vegetation_mask, structure=np.ones((3,3)))
        vegetation_mask = ndimage.binary_closing(vegetation_mask, structure=np.ones((3,3)))
        return vegetation_mask

    def create_false_color_composite(self):
        """NIR + RED + GREEN birleşimi oluştur (false color composite)"""
        # Bantları normalize et
        nir_norm = (self.nir - self.nir.min()) / (self.nir.max() - self.nir.min() + 1e-6)
        red_norm = (self.red - self.red.min()) / (self.red.max() - self.red.min() + 1e-6)
        green_norm = (self.green - self.green.min()) / (self.green.max() - self.green.min() + 1e-6)
        
        # False color composite: NIR -> Red, Red -> Green, Green -> Blue
        false_color = np.stack([nir_norm, red_norm, green_norm], axis=2)
        
        # Kontrastı artır
        false_color = np.clip(false_color * 1.2, 0, 1)
        
        return false_color

class BrushMaskLabeler(QtWidgets.QWidget):
    def __init__(self, dataset_paths):
        super().__init__()
        self.setWindowTitle("Fırça ile Maskeleme (Otomatik Doldurma)")
        self.setGeometry(50,50,1700,900)
        self.dataset_paths = dataset_paths
        self.image_data = []
        self.current_index = 0
        self.current_mask = None
        self.drawing = False
        self.brush_size = 10
        self.last_point = None
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Üst menü
        top_layout = QtWidgets.QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(list(self.dataset_paths.keys()))
        self.load_btn = QPushButton("Görüntüleri Yükle")
        self.prev_btn = QPushButton("Geri")
        self.next_btn = QPushButton("İleri")
        top_layout.addWidget(QLabel("Bitki Türü:"))
        top_layout.addWidget(self.combo)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.prev_btn)
        top_layout.addWidget(self.next_btn)
        layout.addLayout(top_layout)

        # Canvas - NDVI ve False Color görüntüleri yan yana
        self.fig, (self.ax_ndvi, self.ax_false_color) = plt.subplots(1, 2, figsize=(12,6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Alt menü: kaydet, temizle, fırça boyutu, doldur
        bottom_layout = QtWidgets.QHBoxLayout()
        self.clear_btn = QPushButton("Maskeyi Temizle")
        self.save_btn = QPushButton("Maskeyi Kaydet")
        self.brush_slider = QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(self.brush_size)
        self.fill_btn = QPushButton("İçini Doldur")
        self.smooth_checkbox = QCheckBox("Yumuşak Fırça")
        self.smooth_checkbox.setChecked(True)
        bottom_layout.addWidget(self.clear_btn)
        bottom_layout.addWidget(self.save_btn)
        bottom_layout.addWidget(QLabel("Fırça Boyutu:"))
        bottom_layout.addWidget(self.brush_slider)
        bottom_layout.addWidget(self.fill_btn)
        bottom_layout.addWidget(self.smooth_checkbox)
        layout.addLayout(bottom_layout)

        # Event bağlantıları
        self.load_btn.clicked.connect(self.load_images)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_mask)
        self.clear_btn.clicked.connect(self.clear_mask)
        self.fill_btn.clicked.connect(self.fill_mask)
        self.brush_slider.valueChanged.connect(self.update_brush)

        # Her iki eksen için de event bağlantıları
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Optimize overlay
        self.mask_im_ndvi = None
        self.mask_im_false_color = None
        self.background_im_ndvi = None
        self.background_im_false_color = None

    def update_brush(self,val):
        self.brush_size = val

    def load_images(self):
        self.image_data.clear()
        plant = self.combo.currentText()
        path = self.dataset_paths[plant]
        files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
        self.image_data = [(plant,f) for f in files]
        self.current_index = 0
        self.show_image()

    def show_image(self):
        if not self.image_data:
            return
        plant, filename = self.image_data[self.current_index]
        filepath = os.path.join(self.dataset_paths[plant], filename)
        try:
            data = np.load(filepath)
        except Exception as e:
            print(f"Hata: {e}")
            return
        self.current_data = data
        self.current_filename = os.path.splitext(filename)[0]

        analyzer = NDVIAnalyzer(data)
        ndvi = analyzer.ndvi
        false_color = analyzer.create_false_color_composite()
        vegetation_mask = analyzer.create_vegetation_mask()
        masked_ndvi = np.where(vegetation_mask, ndvi, np.nan)

        cmap = LinearSegmentedColormap.from_list('health',['#8B0000','#FF4500','#FFD700','#9ACD32','#006400'],N=100)
        
        # NDVI görüntüsü
        self.ax_ndvi.clear()
        self.background_im_ndvi = self.ax_ndvi.imshow(masked_ndvi, cmap=cmap, vmin=0, vmax=1)
        self.ax_ndvi.set_title(f"{plant} - {filename} - NDVI")
        self.ax_ndvi.axis("off")
        
        # False Color görüntüsü (NIR + RED + GREEN)
        self.ax_false_color.clear()
        self.background_im_false_color = self.ax_false_color.imshow(false_color)
        self.ax_false_color.set_title(f"{plant} - {filename} - False Color (NIR-R-G)")
        self.ax_false_color.axis("off")
        
        # Yeni maske oluştur (her zaman sıfırdan başla)
        self.current_mask = np.zeros(masked_ndvi.shape[:2], dtype=np.uint8)
        
        # Mavi renkli maskeleme için - ÖZEL MAVİ RENK HARİTASI
        blue_cmap = LinearSegmentedColormap.from_list('blue_mask', ['#0000FF', '#0000FF'], N=2)
        
        # Maskeyi her iki görüntüde de çiz
        self.mask_im_ndvi = self.ax_ndvi.imshow(
            np.ma.masked_where(self.current_mask == 0, self.current_mask), 
            cmap=blue_cmap, 
            alpha=0.6,
            vmin=0,
            vmax=1
        )
        
        self.mask_im_false_color = self.ax_false_color.imshow(
            np.ma.masked_where(self.current_mask == 0, self.current_mask), 
            cmap=blue_cmap, 
            alpha=0.6,
            vmin=0,
            vmax=1
        )
        
        self.canvas.draw()

    def get_active_axis(self, event):
        """Tıklanan ekseni belirle"""
        if event.inaxes == self.ax_ndvi:
            return self.ax_ndvi
        elif event.inaxes == self.ax_false_color:
            return self.ax_false_color
        return None

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
            
        active_axis = self.get_active_axis(event)
        if active_axis is None:
            return
            
        if event.button == 1:  # Sol tıklama
            x, y = int(event.xdata), int(event.ydata)
            self.drawing = True
            self.last_point = (x, y)
            self.paint(x, y)

    def on_motion(self, event):
        if self.drawing and event.button == 1:  # Sol tıklama sürüklüyorsa
            if event.xdata is None or event.ydata is None:
                return
                
            active_axis = self.get_active_axis(event)
            if active_axis is None:
                return
                
            x, y = int(event.xdata), int(event.ydata)
            
            if self.smooth_checkbox.isChecked() and self.last_point is not None:
                # Yumuşak çizim - noktalar arasını doldur
                self.draw_line(self.last_point[0], self.last_point[1], x, y)
            else:
                # Normal çizim
                self.paint(x, y)
                
            self.last_point = (x, y)

    def on_release(self, event):
        if event.button == 1:  # Sol tıklama bırakıldı
            self.drawing = False
            self.last_point = None

    def draw_line(self, x0, y0, x1, y1):
        """Bresenham çizgi algoritması ile noktalar arasını doldur"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self.paint(x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self.paint(x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.paint(x, y)

    def paint(self, x, y):
        if self.current_mask is None:
            return
            
        radius = self.brush_size // 2
        
        if self.smooth_checkbox.isChecked():
            # Yumuşak kenarlı fırça
            y_indices, x_indices = np.ogrid[
                max(y-radius, 0):min(y+radius+1, self.current_mask.shape[0]),
                max(x-radius, 0):min(x+radius+1, self.current_mask.shape[1])
            ]
            
            distance = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
            brush_mask = distance <= radius
            
            # Kenar yumuşatma
            edge_mask = (distance > radius * 0.7) & (distance <= radius)
            brush_mask = brush_mask.astype(np.float32)
            brush_mask[edge_mask] = 0.5  # Yarı saydam kenar
            
            # Maskeyi güncelle
            target_area = self.current_mask[
                max(y-radius, 0):min(y+radius+1, self.current_mask.shape[0]),
                max(x-radius, 0):min(x+radius+1, self.current_mask.shape[1])
            ]
            
            # Maksimum değeri al (overlay için)
            update_mask = np.maximum(target_area, brush_mask * 255)
            self.current_mask[
                max(y-radius, 0):min(y+radius+1, self.current_mask.shape[0]),
                max(x-radius, 0):min(x+radius+1, self.current_mask.shape[1])
            ] = update_mask.astype(np.uint8)
        else:
            # Normal dairesel fırça
            x_min = max(x-radius, 0)
            x_max = min(x+radius, self.current_mask.shape[1]-1)
            y_min = max(y-radius, 0)
            y_max = min(y+radius, self.current_mask.shape[0]-1)
            
            yy, xx = np.mgrid[y_min:y_max+1, x_min:x_max+1]
            mask_circle = (xx-x)**2 + (yy-y)**2 <= radius**2
            
            # Maskeyi güncelle
            self.current_mask[yy,xx] = np.maximum(self.current_mask[yy,xx], mask_circle.astype(np.uint8) * 255)
        
        # Maskeyi görselleştir
        if self.mask_im_ndvi is not None and self.mask_im_false_color is not None:
            self.mask_im_ndvi.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
            self.mask_im_false_color.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
            self.canvas.draw_idle()

    def fill_mask(self):
        if self.current_mask is not None:
            # Binary fill için threshold uygula
            binary_mask = (self.current_mask > 127).astype(np.uint8)
            # Kenarın içini doldur
            filled = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
            self.current_mask = filled * 255
            if self.mask_im_ndvi is not None and self.mask_im_false_color is not None:
                self.mask_im_ndvi.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
                self.mask_im_false_color.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
                self.canvas.draw_idle()

    # Navigation
    def next_image(self):
        if self.image_data:
            self.current_index = (self.current_index+1)%len(self.image_data)
            self.show_image()
            
    def prev_image(self):
        if self.image_data:
            self.current_index = (self.current_index-1)%len(self.image_data)
            self.show_image()

    # Mask operations
    def clear_mask(self):
        if self.current_mask is not None:
            self.current_mask[:] = 0
            if self.mask_im_ndvi is not None and self.mask_im_false_color is not None:
                self.mask_im_ndvi.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
                self.mask_im_false_color.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
                self.canvas.draw_idle()
                
    def save_mask(self):
        if self.current_mask is None:
            return
            
        os.makedirs("leaf_mask/npy", exist_ok=True)
        os.makedirs("leaf_mask/png", exist_ok=True)
        os.makedirs("leaf_mask/txt", exist_ok=True)

        # Binary maskeye çevir
        binary_mask = (self.current_mask > 127).astype(np.uint8)
        
        # .npy
        np.save(f"leaf_mask/npy/{self.current_filename}_mask.npy", binary_mask)
        # .png
        plt.imsave(f"leaf_mask/png/{self.current_filename}_mask.png", binary_mask, cmap="Blues")
        # .txt
        ys,xs = np.where(binary_mask==1)
        with open(f"leaf_mask/txt/{self.current_filename}_mask.txt","w") as f:
            for x,y in zip(xs,ys):
                f.write(f"{x} {y}\n")
                
        QtWidgets.QMessageBox.information(self,"Kayıt Başarılı","Maskeler kaydedildi.")

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    paths = {
        "canola": "/home/semih/Desktop/plant_health/data/canola/canola",
        "kochia": "/home/semih/Desktop/plant_health/data/kochia/kochia",
        "ragweed": "/home/semih/Desktop/plant_health/data/ragweed/ragweed",
        "redroot_pigweed": "dataset/redroot_pigweed",
        "soybean": "/home/semih/Desktop/plant_health/data/soybean/soybean",
        "sugarbeet": "/home/semih/Desktop/plant_health/data/sugarbeet/sugarbeet",
        "waterhemp": "/home/semih/Desktop/plant_health/data/waterhemp/waterhemp",
    }
    win = BrushMaskLabeler(paths)
    win.show()
    sys.exit(app.exec_())