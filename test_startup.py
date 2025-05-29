# test_startup.py - Gyors teszt szkript

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        from data.kraken_api_client import KrakenAPIClient
        print("✅ KrakenAPIClient import OK")
    except Exception as e:
        print(f"❌ KrakenAPIClient import failed: {e}")
        return False
        
    try:
        from strategy.indicator_engine import IndicatorEngine
        print("✅ IndicatorEngine import OK")
    except Exception as e:
        print(f"❌ IndicatorEngine import failed: {e}")
        return False
        
    try:
        from core.position_manager import PositionManager
        print("✅ PositionManager import OK")
    except Exception as e:
        print(f"❌ PositionManager import failed: {e}")
        return False
        
    try:
        from gui.main_window import MainWindow
        print("✅ MainWindow import OK")
    except Exception as e:
        print(f"❌ MainWindow import failed: {e}")
        return False
        
    return True

def test_api_connection():
    """Test Kraken API connection"""
    print("\n🌐 Testing API connection...")
    
    try:
        from data.kraken_api_client import KrakenAPIClient
        api = KrakenAPIClient()
        
        # Test basic connection
        if api.test_connection():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return False
            
        # Test getting pairs
        pairs = api.get_valid_usd_pairs()
        if pairs:
            print(f"✅ Got {len(pairs)} trading pairs")
            for pair in pairs[:5]:
                print(f"   - {pair['altname']} ({pair['wsname']})")
        else:
            print("⚠️ No trading pairs received, using fallback")
            
        # Test OHLC data
        print("\n📊 Testing OHLC data...")
        test_pairs = ['XBTUSD', 'ETHUSD']
        
        for pair in test_pairs:
            ohlc = api.get_ohlc(pair, interval=1)
            if ohlc:
                data_length = len(list(ohlc.values())[0])
                print(f"✅ {pair}: {data_length} candles")
            else:
                print(f"❌ {pair}: No OHLC data")
                
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_gui_components():
    """Test GUI components without starting full app"""
    print("\n🖥️ Testing GUI components...")
    
    try:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        
        # Test dashboard
        from gui.dashboard_panel import DashboardPanel
        dashboard = DashboardPanel()
        print("✅ DashboardPanel created")
        
        # Test settings
        from gui.settings_panel import SettingsPanel
        settings = SettingsPanel()
        print("✅ SettingsPanel created")
        
        # Test position list
        from core.position_manager import PositionManager
        from gui.position_list_widget import PositionListWidget
        pos_manager = PositionManager()
        pos_list = PositionListWidget(pos_manager)
        print("✅ PositionListWidget created")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def main():
    """Run all startup tests"""
    print("🚀 Trading Bot Startup Tests\n")
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
        
    # Test API
    if not test_api_connection():
        success = False
        
    # Test GUI
    if not test_gui_components():
        success = False
        
    print(f"\n{'='*50}")
    if success:
        print("🎉 All tests passed! Bot ready to start.")
        print("\nRun: python app.py")
    else:
        print("⚠️ Some tests failed. Check errors above.")
        
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
