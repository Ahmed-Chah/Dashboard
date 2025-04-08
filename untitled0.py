import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QMessageBox, QGridLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MicrorobotDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microrobot Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        self.folder_path = os.getcwd()
        self.available_files = [f for f in os.listdir(self.folder_path) if f.endswith('.xlsx')]
        self.microrobots = [f.replace("DATA__", "").replace(".xlsx", "") for f in self.available_files]
        
        self.initUI()

    def initUI(self):
        container = QWidget()
        self.setCentralWidget(container)
        
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Dropdown for microrobot selection
        self.label = QLabel("Sélectionner un microrobot:")
        layout.addWidget(self.label)
        
        self.comboBox = QComboBox()
        self.comboBox.addItems(self.microrobots)
        self.comboBox.currentTextChanged.connect(self.update_analysis)
        layout.addWidget(self.comboBox)
        
        # Grid Layout for statistics
        stats_layout = QGridLayout()
        self.max_label = QLabel("Déflexion Max: ")
        self.min_label = QLabel("Déflexion Min: ")
        self.mean_label = QLabel("Déflexion Moyenne: ")
        
        stats_layout.addWidget(self.max_label, 0, 0)
        stats_layout.addWidget(self.min_label, 0, 1)
        stats_layout.addWidget(self.mean_label, 0, 2)
        layout.addLayout(stats_layout)
        
        # Canvas for charts
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)
        
        if self.microrobots:
            self.update_analysis(self.microrobots[0])
        else:
            QMessageBox.warning(self, "Erreur", "Aucun fichier de microrobot trouvé!")

    def update_analysis(self, selected_microrobot):
        file_path = os.path.join(self.folder_path, f"DATA__{selected_microrobot}.xlsx")
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Erreur", "Fichier non trouvé!")
            return
        
        df = pd.read_excel(file_path)
        
        # Compute statistics
        max_deflection = df['Head Deflection Angle [°]'].max()
        min_deflection = df['Head Deflection Angle [°]'].min()
        mean_deflection = df['Head Deflection Angle [°]'].mean()
        
        self.max_label.setText(f"Déflexion Max: {max_deflection:.2f}°")
        self.min_label.setText(f"Déflexion Min: {min_deflection:.2f}°")
        self.mean_label.setText(f"Déflexion Moyenne: {mean_deflection:.2f}°")
        
        # Clear and update plot
        plt.clf()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        sns.boxplot(x=df['Magnetic Distance [cm]'], y=df['Head Deflection Angle [°]'], ax=axes[0, 0])
        axes[0, 0].set_title("Déflexion vs Distance")
        
        sns.boxplot(x=df['Actuation Angle'], y=df['Head Deflection Angle [°]'], ax=axes[0, 1])
        axes[0, 1].set_title("Déflexion vs Angle d'Actionnement")
        
        sns.boxplot(x=df['Flow Profile'], y=df['Head Deflection Angle [°]'], ax=axes[1, 0])
        axes[1, 0].set_title("Déflexion vs Profil de Flux")
        
        sns.boxplot(x=df['Actuation Mode'], y=df['Head Deflection Angle [°]'], ax=axes[1, 1])
        axes[1, 1].set_title("Déflexion vs Mode d'Actionnement")
        
        plt.tight_layout()
        self.canvas.figure = fig
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MicrorobotDashboard()
    mainWin.show()
    sys.exit(app.exec_())
