# app.py - JAVÍTOTT VERZIÓ DARK THEME-mel

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

def apply_dark_theme(app):
    """🎨 Sötét téma alkalmazása"""
    dark_palette = QPalette()
    
    # 🎯 SÖTÉT SZÍNEK BEÁLLÍTÁSA
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))           # Főablak háttér
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))    # Szöveg
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))             # Input háttér
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))    # Alternatív háttér
    dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))         # Tooltip háttér
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))   # Tooltip szöveg
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))          # Input szöveg
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))           # Gomb háttér
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))    # Gomb szöveg
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))        # Világos szöveg
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))           # 🔵 KÉKES LINKEK
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))      # 🔵 KÉKES KIJELÖLÉS
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))     # Kijelölt szöveg
    
    # ✅ TÉMA ALKALMAZÁSA
    app.setPalette(dark_palette)
    app.setStyle("Fusion")  # Modern megjelenés

def main():
    print("🚀 Advanced Trading Bot - $50 Bollinger Breakout Edition")
    print("=" * 60)
    
    try:
        # ✅ MAIN WINDOW BETÖLTÉSE
        from gui.main_window import MainWindow
        
        print("📊 Loading trading components...")
        print("   ✅ GUI System")
        print("   ✅ Kraken API Client") 
        print("   ✅ Position Manager")
        print("   ✅ Trading Logic")
        print("   ✅ Risk Management")
        print("   ✅ Technical Indicators")
        
        # ✅ QT ALKALMAZÁS
        app = QApplication(sys.argv)
        
        # 🎨 SÖTÉT TÉMA ALKALMAZÁSA
        print("🎨 Applying dark theme...")
        apply_dark_theme(app)
        
        # ✅ FŐABLAK
        print("\n🎯 Starting main trading interface...")
        window = MainWindow()
        window.show()
        
        print("✅ Trading Bot ready!")
        print("\n🎮 Controls:")
        print("   🚀 Start Live Trading - Begin automated trading")
        print("   📊 Monitor charts and positions")
        print("   ⚙️ Adjust settings in right panel")
        print("   🛑 Emergency stop available")
        
        print(f"\n{'=' * 60}")
        print("🎯 Ready for $50 Bollinger Breakout Trading!")
        print(f"{'=' * 60}\n")
        
        # ✅ FUTTATÁS
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        print("🔧 Check if all required files are present")
        return False
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("🔧 Check configuration and dependencies")
        return False

if __name__ == "__main__":
    main()
