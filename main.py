import sys
import os
import numpy as np
import pydicom
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QSlider, QLabel, QToolBar, QAction,
                             QDockWidget, QListWidget, QSplitter, QFrame, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MPRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.volume = None
        self.current_slices = [0, 0, 0]
        self.playing = [False, False, False]
        self.point = None
        self.timers = [QTimer(self) for _ in range(3)]
        for timer in self.timers:
            timer.timeout.connect(self.update_cine)
        self.brightness = [0, 0, 0]
        self.contrast = [1, 1, 1]
        self.current_folder = None
        self.file_list = []
        self.zoom_factors = [1, 1, 1]
        self.pan_offset = [[0, 0], [0, 0], [0, 0]]
        self.last_pos = [None, None, None]
        self.is_panning = [False, False, False]
        self.pointer_mode = False

    def initUI(self):
        self.setWindowTitle('Advanced MPR Viewer')
        self.setGeometry(100, 100, 1300, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add Cine controls for each viewer
        self.cine_buttons = []
        for i in range(3):
            viewer_controls = QVBoxLayout()
            label = QLabel(f"Viewer {i + 1} Controls")
            viewer_controls.addWidget(label)

            play_button = QPushButton("Play")
            play_button.clicked.connect(lambda _, idx=i: self.toggle_play(idx))
            viewer_controls.addWidget(play_button)

            stop_button = QPushButton("Stop")
            stop_button.clicked.connect(lambda _, idx=i: self.stop_cine(idx))
            viewer_controls.addWidget(stop_button)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.valueChanged.connect(lambda value, idx=i: self.update_slice_from_slider(idx, value))
            viewer_controls.addWidget(slider)

            brightness_label = QLabel("Brightness:")
            viewer_controls.addWidget(brightness_label)
            brightness_slider = QSlider(Qt.Horizontal)
            brightness_slider.setMinimum(-100)
            brightness_slider.setMaximum(100)
            brightness_slider.setValue(0)
            brightness_slider.valueChanged.connect(lambda value, idx=i: self.update_brightness(idx, value))
            viewer_controls.addWidget(brightness_slider)

            contrast_label = QLabel("Contrast:")
            viewer_controls.addWidget(contrast_label)
            contrast_slider = QSlider(Qt.Horizontal)
            contrast_slider.setMinimum(1)
            contrast_slider.setMaximum(300)
            contrast_slider.setValue(100)
            contrast_slider.valueChanged.connect(lambda value, idx=i: self.update_contrast(idx, value))
            viewer_controls.addWidget(contrast_slider)

            self.cine_buttons.append((play_button, stop_button, slider, brightness_slider, contrast_slider))
            left_layout.addLayout(viewer_controls)

        # Add rendering mode selector
        render_mode_label = QLabel("3D Rendering Mode:")
        left_layout.addWidget(render_mode_label)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Surface", "Volume"])
        self.render_mode_combo.currentIndexChanged.connect(self.change_render_mode)
        left_layout.addWidget(self.render_mode_combo)

        left_layout.addStretch()

        # Create viewers layout
        viewers_widget = QWidget()
        viewers_layout = QVBoxLayout(viewers_widget)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Vertical)
        viewers_layout.addWidget(splitter)

        # Create top row with two viewers
        top_row = QSplitter(Qt.Horizontal)
        self.viewers = []
        for i in range(2):
            viewer_frame = QFrame()
            viewer_frame.setFrameStyle(QFrame.StyledPanel)
            viewer_layout = QVBoxLayout(viewer_frame)
            fig = Figure(figsize=(5, 5), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            self.viewers.append((fig, canvas, ax))
            viewer_layout.addWidget(canvas)
            top_row.addWidget(viewer_frame)

        splitter.addWidget(top_row)

        # Create bottom row with 3D view and another 2D viewer
        bottom_row = QSplitter(Qt.Horizontal)

        # 3D viewer
        vtk_frame = QFrame()
        vtk_frame.setFrameStyle(QFrame.StyledPanel)
        vtk_layout = QVBoxLayout(vtk_frame)
        self.frame = QVTKRenderWindowInteractor()
        self.vtkWidget = self.frame
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        vtk_layout.addWidget(self.vtkWidget)
        bottom_row.addWidget(vtk_frame)

        # Third 2D viewer
        viewer_frame = QFrame()
        viewer_frame.setFrameStyle(QFrame.StyledPanel)
        viewer_layout = QVBoxLayout(viewer_frame)
        fig = Figure(figsize=(5, 5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self.viewers.append((fig, canvas, ax))
        viewer_layout.addWidget(canvas)
        bottom_row.addWidget(viewer_frame)
        splitter.addWidget(bottom_row)
       

        # Set equal sizes for all views
        splitter.setSizes([int(self.height()/2), int(self.height()/2)])
        top_row.setSizes([int(self.width()/2), int(self.width()/2)])
        bottom_row.setSizes([int(self.width()/2), int(self.width()/2)])

        # Add layouts to main layout تغيير عرض ال left tool bar
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(viewers_widget, 4)

        # Add file list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.load_selected_file)
        file_dock = QDockWidget("Files", self)
        file_dock.setWidget(self.file_list_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, file_dock)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create status bar
        self.statusBar().showMessage('Ready')

        # Initialize VTK
        self.iren.Initialize()

        # Connect events
        for i, (fig, canvas, _) in enumerate(self.viewers):
            fig.canvas.mpl_connect('scroll_event', lambda event, i=i: self.on_scroll(event, i))
            fig.canvas.mpl_connect('button_press_event', lambda event, i=i: self.on_press(event, i))
            fig.canvas.mpl_connect('motion_notify_event', lambda event, i=i: self.on_motion(event, i))
            fig.canvas.mpl_connect('button_release_event', lambda event, i=i: self.on_release(event, i))

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load Folder', self)
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)

    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        load_action = QAction(QIcon('icon_load.png'), 'Load Folder', self)
        load_action.triggered.connect(self.load_data)
        toolbar.addAction(load_action)

        # Add pointer tool button
        self.pointer_action = QAction(QIcon('icon_pointer.png'), 'Pointer Tool', self)
        self.pointer_action.setCheckable(True)
        self.pointer_action.triggered.connect(self.toggle_pointer_tool)
        toolbar.addAction(self.pointer_action)

    def toggle_pointer_tool(self, checked):
        self.pointer_mode = checked
        self.statusBar().showMessage('Pointer tool ' + ('activated' if checked else 'deactivated'))

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing DICOM or NIfTI files", options=options)
        if folder_path:
            self.current_folder = folder_path
            self.load_folder(folder_path)

    def load_folder(self, folder_path):
        self.file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.nii', '.nii.gz')):
                    self.file_list.append(os.path.join(root, file))
        
        if not self.file_list:
            QMessageBox.warning(self, "Error", "No DICOM or NIfTI files found in the selected folder.")
            return

        self.update_file_list_widget()
        self.load_first_file()

    def update_file_list_widget(self):
        self.file_list_widget.clear()
        for file_path in self.file_list:
            self.file_list_widget.addItem(os.path.basename(file_path))

    def load_first_file(self):
        if self.file_list:
            first_file = self.file_list[0]
            if first_file.lower().endswith('.dcm'):
                self.load_dicom(os.path.dirname(first_file))
            else:
                self.load_nifti(first_file)

    def load_selected_file(self, item):
        file_path = os.path.join(self.current_folder, item.text())
        if file_path.lower().endswith('.dcm'):
            self.load_dicom(os.path.dirname(file_path))
        else:
            self.load_nifti(file_path)

    def load_dicom(self, folder):
        slices = []
        dicom_files = [f for f in os.listdir(folder) if f.lower().endswith('.dcm')]
        for s in dicom_files:
            try:
                ds = pydicom.dcmread(os.path.join(folder, s))
                slices.append(ds)
            except:
                pass
        if not slices:
            QMessageBox.warning(self, "Error", "No valid DICOM files found in the selected folder.")
            return
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        self.volume = np.stack([s.pixel_array for s in slices])
        self.update_viewers()
        self.update_3d_view()
        self.statusBar().showMessage(f'Loaded {len(slices)} DICOM slices')

    def load_nifti(self, file_path):
        try:
            nii = nib.load(file_path)
            self.volume = nii.get_fdata()
            self.update_viewers()
            self.update_3d_view()
            self.statusBar().showMessage(f'Loaded NIfTI file: {os.path.basename(file_path)}')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load NIfTI file: {str(e)}")

    def update_viewers(self):
        if self.volume is not None:
            for i in range(3):
                self.update_slice(i, self.current_slices[i])
                self.cine_buttons[i][2].setMaximum(self.volume.shape[i] - 1)

    def update_slice(self, viewer_index, slice_index):
        if self.volume is None:
            return

        self.current_slices[viewer_index] = slice_index
        fig, canvas, ax = self.viewers[viewer_index]
        ax.clear()

        if viewer_index == 0:
            img = self.volume[slice_index, :, :]
        elif viewer_index == 1:
            img = self.volume[:, slice_index, :]
        else:
            img = self.volume[:, :, slice_index]

        # Apply brightness and contrast
        img = self.apply_brightness_contrast(img, viewer_index)

        ax.imshow(img, cmap='gray', aspect='equal', interpolation='nearest')
        ax.axis('off')

        if self.point is not None:
            x, y, z = self.point
            if viewer_index == 0 and x == slice_index:
                ax.plot(z, y, 'ro')
            elif viewer_index == 1 and y == slice_index:
                ax.plot(z, x, 'ro')
            elif viewer_index == 2 and z == slice_index:
                ax.plot(y, x, 'ro')

        self.draw_crosshairs(viewer_index)
        
        # Apply zoom and pan
        ax.set_xlim(self.pan_offset[viewer_index][0],
                    self.pan_offset[viewer_index][0] + img.shape[1] / self.zoom_factors[viewer_index])
        ax.set_ylim(self.pan_offset[viewer_index][1] + img.shape[0] / self.zoom_factors[viewer_index],
                    self.pan_offset[viewer_index][1])

        fig.tight_layout()
        canvas.draw_idle()

    def apply_brightness_contrast(self, img, viewer_index):
        return np.clip((img - img.min()) / (img.max() - img.min()) * self.contrast[viewer_index] + self.brightness[viewer_index], 0, 1)

    def draw_crosshairs(self, viewer_index):
        _, _, ax = self.viewers[viewer_index]
        shape = self.volume.shape
        if viewer_index == 0:
            ax.axhline(self.current_slices[1], color='r', alpha=0.5)
            ax.axvline(self.current_slices[2], color='g', alpha=0.5)
        elif viewer_index == 1:
            ax.axhline(self.current_slices[0], color='r', alpha=0.5)
            ax.axvline(self.current_slices[2], color='b', alpha=0.5)
        else:
            ax.axhline(self.current_slices[0], color='r', alpha=0.5)
            ax.axvline(self.current_slices[1], color='g', alpha=0.5)

    def update_3d_view(self):
        if self.volume is None:
            return

        # Create VTK data object
        dataImporter = vtk.vtkImageImport()
        data_string = self.volume.astype(np.uint8).tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, self.volume.shape[2] - 1, 0, self.volume.shape[1] - 1, 0,
                                   self.volume.shape[0] - 1)
        dataImporter.SetWholeExtent(0, self.volume.shape[2] - 1, 0, self.volume.shape[1] - 1, 0,
                                    self.volume.shape[0] - 1)

        # Create transfer mapping scalar value to opacity
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(20, 0.0)
        opacityTransferFunction.AddPoint(255, 0.2)

        # Create transfer mapping scalar value to color
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
        colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
        colorTransferFunction.AddRGBPoint(255.0, 1.0, 1.0, 1.0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        self.volume_3d = vtk.vtkVolume()
        self.volume_3d.SetMapper(volumeMapper)
        self.volume_3d.SetProperty(volumeProperty)

        # Surface rendering
        isoValue = 128
        mcubes = vtk.vtkMarchingCubes()
        mcubes.SetInputConnection(dataImporter.GetOutputPort())
        mcubes.SetValue(0, isoValue)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mcubes.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self.surface = vtk.vtkActor()
        self.surface.SetMapper(mapper)
        self.surface.GetProperty().SetColor(1.0, 1.0, 1.0)

        self.ren.RemoveAllViewProps()
        self.ren.AddVolume(self.volume_3d)
        self.ren.AddActor(self.surface)
        self.surface.SetVisibility(False)  # Start with volume rendering by default
        self.ren.ResetCamera()
        self.frame.Render()

    def on_scroll(self, event, viewer_index):
        if event.button == 'up':
            self.zoom_factors[viewer_index] *= 1.05
        elif event.button == 'down':
            self.zoom_factors[viewer_index] /= 1.05

        self.update_slice(viewer_index, self.current_slices[viewer_index])

    def on_press(self, event, viewer_index):
        if event.button == 1:  # Left click
            self.last_pos[viewer_index] = (event.xdata, event.ydata)
            if self.pointer_mode:
                if viewer_index == 0:
                    self.point = [self.current_slices[0], int(event.ydata), int(event.xdata)]
                elif viewer_index == 1:
                    self.point = [int(event.ydata), self.current_slices[1], int(event.xdata)]
                else:
                    self.point = [int(event.ydata), int(event.xdata), self.current_slices[2]]

                self.current_slices = self.point
                for i in range(3):
                    self.update_slice(i, self.current_slices[i])
                    self.cine_buttons[i][2].setValue(self.current_slices[i])
            else:
                self.is_panning[viewer_index] = True

    def on_motion(self, event, viewer_index):
        if self.is_panning[viewer_index] and event.inaxes and self.last_pos[viewer_index] is not None:
            # Pan the image
            dx = self.last_pos[viewer_index][0] - event.xdata
            dy = self.last_pos[viewer_index][1] - event.ydata
            self.pan_offset[viewer_index][0] += dx / self.zoom_factors[viewer_index]
            self.pan_offset[viewer_index][1] += dy / self.zoom_factors[viewer_index]
            self.update_slice(viewer_index, self.current_slices[viewer_index])

            self.last_pos[viewer_index] = (event.xdata, event.ydata)

    def on_release(self, event, viewer_index):
        self.last_pos[viewer_index] = None
        self.is_panning[viewer_index] = False

    def toggle_play(self, viewer_index):
        if self.playing[viewer_index]:
            self.stop_cine(viewer_index)
        else:
            self.start_cine(viewer_index)

    def start_cine(self, viewer_index):
        if not self.playing[viewer_index]:
            self.playing[viewer_index] = True
            self.cine_buttons[viewer_index][0].setText("Pause")
            self.timers[viewer_index].start(100)  # Update every 100 ms

    def stop_cine(self, viewer_index):
        if self.playing[viewer_index]:
            self.playing[viewer_index] = False
            self.cine_buttons[viewer_index][0].setText("Play")
            self.timers[viewer_index].stop()

    def update_cine(self):
        for i in range(3):
            if self.playing[i]:
                max_slice = self.volume.shape[i] - 1
                self.current_slices[i] = (self.current_slices[i] + 1) % (max_slice + 1)
                self.update_slice(i, self.current_slices[i])
                self.cine_buttons[i][2].setValue(self.current_slices[i])

    def update_slice_from_slider(self, viewer_index, value):
        self.current_slices[viewer_index] = value
        self.update_slice(viewer_index, value)

    def update_brightness(self, viewer_index, value):
        self.brightness[viewer_index] = value / 100.0
        self.update_slice(viewer_index, self.current_slices[viewer_index])

    def update_contrast(self, viewer_index, value):
        self.contrast[viewer_index] = value / 100.0
        self.update_slice(viewer_index, self.current_slices[viewer_index])

    def change_render_mode(self, index):
        if self.volume is None:
            return

        if index == 0:  # Surface rendering
            self.volume_3d.SetVisibility(False)
            self.surface.SetVisibility(True)
        else:  # Volume rendering
            self.volume_3d.SetVisibility(True)
            self.surface.SetVisibility(False)

        self.frame.Render()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())
