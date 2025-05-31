# start_trading_bot.py - EGYSZERŰ INDÍTÓ SZKRIPT

import sys
import os

# ✅ EGYETLEN FÁJL INDÍTÁSA
if __name__ == "__main__":
    print("🚀 Starting Advanced Trading Bot...")
    print("📊 Loading all modules automatically...")
    
    try:
        # ✅ CSAK EGY IMPORT KELL!
        from main_window import MainWindow
        from PyQt5.QtWidgets import QApplication
        
        # ✅ GUI INDÍTÁSA
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        print("✅ Trading Bot GUI started successfully!")
        print("🎯 Use the GUI buttons to:")
        print("   - Start/Stop Live Trading")
        print("   - Monitor positions")
        print("   - View charts and analysis")
        print("   - Control all bot functions")
        
        # ✅ FUTTATÁS
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Error starting trading bot: {e}")
        print("🔧 Check your dependencies and configuration")
