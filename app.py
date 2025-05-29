import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import QApplication
import sys
from gui.main_window import MainWindow
from style_dark import apply_dark_theme
from utils.logger import logger
import signal

def signal_handler(sig, frame):
    """Clean shutdown on Ctrl+C"""
    print("\n🛑 Shutting down trading bot...")
    
    # Get the main window if it exists
    app = QApplication.instance()
    if app:
        for widget in app.allWidgets():
            if isinstance(widget, MainWindow):
                # Cleanup WebSocket connections
                if hasattr(widget, 'api') and hasattr(widget.api, 'cleanup'):
                    widget.api.cleanup()
                break
    
    print("✅ Cleanup completed")
    sys.exit(0)

def main():
    """Main application entry point with volume-based trading"""
    try:
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        print("🚀 Starting Volume-Based Kraken Trading Bot...")
        print("💰 Target: $50 positions, $0.15 profit, 500K+ volume pairs")
        
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Apply dark theme
        try:
            apply_dark_theme(app)
            print("✅ Dark theme applied")
        except Exception as e:
            print(f"⚠️ Theme application failed: {e}")
        
        # Create and show main window
        print("🖥️ Creating main window with volume filtering...")
        window = MainWindow()
        
        # Initialize volume-based trading
        try:
            print("📊 Connecting to volume-filtered market data...")
            
            # Test volume filtering
            if hasattr(window, 'api') and hasattr(window.api, 'get_volume_statistics'):
                volume_stats = window.api.get_volume_statistics()
                if volume_stats:
                    print(f"📈 Volume stats: {volume_stats.get('above_500k', 0)} pairs above $500K")
            
            print("✅ Volume-based trading system ready")
            
        except Exception as e:
            print(f"⚠️ Volume system initialization warning: {e}")
        
        window.show()
        
        print("✅ Application started successfully")
        print("🎯 Focus: High-volume altcoins (BTC/ETH excluded)")
        print("💡 Minimum volume: $500,000 USD")
        logger.info("Volume-based trading application started")
        
        # Start event loop
        print("🔄 Starting event loop...")
        print("📝 Press Ctrl+C for clean shutdown")
        
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
