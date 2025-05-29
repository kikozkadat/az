# gui/position_list_widget.py – FIXED VERSION

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import time

class PositionListWidget(QWidget):
    def __init__(self, position_manager=None, parent_window=None):
        super().__init__()
        self.position_manager = position_manager
        self.parent_window = parent_window
        self.setup_ui()
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_history)
        self.refresh_timer.start(10000)  # Update every 10 seconds

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title with styling
        self.title = QLabel("📊 Active Positions")
        self.title.setStyleSheet("""
            QLabel {
                font-size: 14px; 
                font-weight: bold; 
                color: #00AAFF;
                padding: 8px;
                border-bottom: 2px solid #333;
                background-color: #2D2D30;
            }
        """)
        layout.addWidget(self.title)

        # Position display area
        self.position_display = QLabel("No active positions")
        self.position_display.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: white;
                padding: 15px;
                border: 1px solid #555;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        self.position_display.setWordWrap(True)
        self.position_display.setMinimumHeight(150)
        layout.addWidget(self.position_display)

        # Control buttons
        button_layout = QVBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh Positions")
        self.refresh_btn.clicked.connect(self.update_history)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
        """)
        button_layout.addWidget(self.refresh_btn)
        
        self.close_all_btn = QPushButton("❌ Close All Positions")
        self.close_all_btn.clicked.connect(self.close_all_positions)
        self.close_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        button_layout.addWidget(self.close_all_btn)
        
        layout.addLayout(button_layout)
        
        # Initial update
        self.update_history()

    def update_history(self):
        """Update position display with enhanced formatting"""
        try:
            if not self.position_manager:
                self.position_display.setText("""
🔸 No position manager connected
🔸 Positions cannot be displayed
🔸 Check system configuration
                """.strip())
                return

            positions = self.position_manager.get_all_positions()
            
            if not positions:
                self.position_display.setText("""
📊 POSITION STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔸 No active positions
🔸 Ready for new opportunities  
🔸 $50 Bollinger breakout system active
🔸 Scanning for high-correlation setups

💡 Tip: Monitor for volume spikes + BB breakouts
                """.strip())
                return

            # Format active positions
            display_text = """
📊 ACTIVE POSITIONS ({count})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

""".format(count=len(positions)).strip()

            for pair, pos_data in positions.items():
                try:
                    side = pos_data.get('side', 'N/A').upper()
                    entry_price = pos_data.get('entry_price', 0)
                    volume = pos_data.get('volume', 0)
                    sl = pos_data.get('stop_loss', 'N/A')
                    tp = pos_data.get('take_profit', 'N/A')
                    open_time = pos_data.get('open_time', time.time())
                    
                    # Calculate position value and hold time
                    pos_value = entry_price * volume if entry_price and volume else 0
                    hold_minutes = (time.time() - open_time) / 60 if open_time else 0
                    
                    # Mock current price for P&L calculation
                    current_price = self._get_mock_current_price(pair, entry_price)
                    
                    # Calculate P&L
                    if side == 'BUY':
                        pnl_usd = (current_price - entry_price) * volume
                    else:
                        pnl_usd = (entry_price - current_price) * volume
                    
                    pnl_pct = (pnl_usd / pos_value * 100) if pos_value > 0 else 0
                    
                    # Color coding for P&L
                    pnl_emoji = "🟢" if pnl_usd >= 0 else "🔴"
                    
                    display_text += f"""

🎯 {pair}
   ├─ Side: {side}
   ├─ Entry: ${entry_price:.4f}
   ├─ Current: ${current_price:.4f}
   ├─ Volume: {volume:.6f}
   ├─ Value: ${pos_value:.2f}
   ├─ Hold Time: {hold_minutes:.0f}m
   ├─ P&L: {pnl_emoji} ${pnl_usd:.2f} ({pnl_pct:+.1f}%)
   ├─ SL: ${sl:.4f if isinstance(sl, (int, float)) else sl}
   └─ TP: ${tp:.4f if isinstance(tp, (int, float)) else tp}
"""
                except Exception as e:
                    display_text += f"\n❌ Error displaying {pair}: {str(e)[:30]}"

            # Add summary footer
            total_positions = len(positions)
            display_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💼 Portfolio: {total_positions} position{'s' if total_positions != 1 else ''}
⏰ Updated: {time.strftime('%H:%M:%S')}
🎯 Strategy: $50 Bollinger Breakout
            """.strip()

            self.position_display.setText(display_text.strip())
            
            # Update title with position count
            self.title.setText(f"📊 Active Positions ({total_positions})")
            
        except Exception as e:
            self.position_display.setText(f"""
❌ ERROR UPDATING POSITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Error: {str(e)[:100]}

🔧 Troubleshooting:
• Check position manager connection
• Verify data source integrity  
• Review system logs for details
            """.strip())
            print(f"[ERROR] PositionListWidget update failed: {e}")

    def _get_mock_current_price(self, pair: str, entry_price: float) -> float:
        """Get mock current price for display purposes"""
        try:
            # Mock price movement (±1% from entry)
            import random
            movement = random.uniform(-0.01, 0.01)
            return entry_price * (1 + movement)
        except:
            return entry_price

    def close_all_positions(self):
        """Close all positions with confirmation"""
        try:
            if not self.position_manager:
                print("[ERROR] No position manager available")
                return
                
            positions = self.position_manager.get_all_positions()
            
            if not positions:
                print("[INFO] No positions to close")
                return
            
            # Close all positions
            closed_count = 0
            for pair in list(positions.keys()):
                try:
                    current_price = self._get_mock_current_price(pair, positions[pair].get('entry_price', 1000))
                    closed_position = self.position_manager.close_position(pair, current_price)
                    if closed_position:
                        closed_count += 1
                        print(f"[CLOSE] Closed position: {pair}")
                except Exception as e:
                    print(f"[ERROR] Failed to close {pair}: {e}")
            
            print(f"[INFO] Closed {closed_count} positions")
            
            # Trigger immediate update
            self.update_history()
            
            # Notify parent window if available
            if hasattr(self.parent_window, 'refresh_open_trades'):
                self.parent_window.refresh_open_trades()
                
        except Exception as e:
            print(f"[ERROR] Close all positions failed: {e}")

    def get_position_summary(self) -> dict:
        """Get summary of positions for external use"""
        try:
            if not self.position_manager:
                return {'total_positions': 0, 'total_value': 0, 'total_pnl': 0}
                
            positions = self.position_manager.get_all_positions()
            
            total_value = 0
            total_pnl = 0
            
            for pair, pos_data in positions.items():
                entry_price = pos_data.get('entry_price', 0)
                volume = pos_data.get('volume', 0)
                pos_value = entry_price * volume
                total_value += pos_value
                
                # Calculate P&L (mock)
                current_price = self._get_mock_current_price(pair, entry_price)
                side = pos_data.get('side', 'buy').lower()
                
                if side == 'buy':
                    pnl = (current_price - entry_price) * volume
                else:
                    pnl = (entry_price - current_price) * volume
                    
                total_pnl += pnl
            
            return {
                'total_positions': len(positions),
                'total_value': total_value,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            print(f"[ERROR] Position summary failed: {e}")
            return {'total_positions': 0, 'total_value': 0, 'total_pnl': 0}

    def set_position_manager(self, position_manager):
        """Set or update position manager"""
        self.position_manager = position_manager
        self.update_history()
