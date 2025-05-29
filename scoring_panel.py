# gui/scoring_panel.py - Javított verzió

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QHeaderView
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import time

class ScoringPanel(QWidget):
    def __init__(self, scorer, coin_data_provider, parent=None):
        super().__init__(parent)
        self.scorer = scorer
        self.coin_data_provider = coin_data_provider
        self.init_ui()
        self.refresh_interval = 10000  # 10 seconds
        self.setup_timer()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Header label
        self.label = QLabel("🎯 Mikro-Trading Érme Pontozó", self)
        self.label.setStyleSheet("font-weight: bold; font-size: 14px; color: white; padding: 5px;")
        self.layout.addWidget(self.label)

        # Main scoring table
        self.table = QTableWidget(self)
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Szimbólum", "Pontszám", "Forgalom", "RSI(14)", "MACD", 
            "Bollinger", "Volatilitás", "BTC Korr.", "Státusz", "Művelet"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2D2D30;
                color: white;
                gridline-color: #555;
                selection-background-color: #0078D4;
            }
            QHeaderView::section {
                background-color: #404040;
                color: white;
                font-weight: bold;
                border: 1px solid #555;
                padding: 4px;
            }
        """)
        self.layout.addWidget(self.table)

        # Blacklist section
        self.blacklist_label = QLabel("🚫 Blacklisted Érmék:", self)
        self.blacklist_label.setStyleSheet("font-weight: bold; color: #FF6B35; padding: 5px;")
        self.layout.addWidget(self.blacklist_label)

        self.blacklist_table = QTableWidget(self)
        self.blacklist_table.setColumnCount(3)
        self.blacklist_table.setHorizontalHeaderLabels([
            "Szimbólum", "Feloldás ideje", "Indok"
        ])
        self.blacklist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.blacklist_table.setMaximumHeight(100)
        self.blacklist_table.setStyleSheet("""
            QTableWidget {
                background-color: #3D2D2D;
                color: #FF6B35;
                gridline-color: #555;
            }
        """)
        self.layout.addWidget(self.blacklist_table)

        # Status label
        self.status_label = QLabel("Státusz: Betöltés...", self)
        self.status_label.setStyleSheet("color: #00AA00; font-weight: bold; padding: 5px;")
        self.layout.addWidget(self.status_label)

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_panel)
        self.timer.start(self.refresh_interval)

    def refresh_panel(self):
        """Scoring panel frissítése"""
        try:
            if not self.coin_data_provider:
                self.status_label.setText("Státusz: Nincs adatforrás")
                return

            coin_list = self.coin_data_provider()
            if not coin_list:
                self.status_label.setText("Státusz: Nincs coin adat")
                return

            self.table.setRowCount(len(coin_list))
            
            scored_coins = []
            pump_count = 0

            for i, coin in enumerate(coin_list):
                try:
                    # Score calculation
                    if hasattr(self.scorer, 'score_coin'):
                        score = self.scorer.score_coin(coin)
                    else:
                        score = self._calculate_fallback_score(coin)

                    scored_coins.append((coin, score))

                    # Extract coin data safely
                    symbol = coin.get('symbol', f'COIN_{i}')
                    volume_ratio = self._safe_divide(coin.get('volume_last', 0), coin.get('volume_15m_avg', 1))
                    rsi = coin.get('rsi_3m', coin.get('rsi_15m', 50))
                    
                    # Format volume display
                    volume_display = f"{coin.get('volume_last', 0):.0f} / {coin.get('volume_15m_avg', 0):.0f}"
                    
                    # RSI display
                    rsi_display = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "N/A"
                    
                    # MACD status (simplified)
                    macd_status = "📈" if coin.get('rsi_3m', 50) > coin.get('rsi_15m', 50) else "📉"
                    
                    # Bollinger status
                    close_price = coin.get('close', 0)
                    bb_upper = coin.get('boll_3m_upper', close_price * 1.02)
                    bb_lower = coin.get('boll_3m_lower', close_price * 0.98)
                    
                    if close_price > bb_upper * 0.98:
                        bb_status = "🔥 Kitörés"
                    elif close_price < bb_lower * 1.02:
                        bb_status = "❄️ Túladott"
                    else:
                        bb_status = "➡️ Középen"
                    
                    # Volatility
                    volatility = f"{volume_ratio:.1f}x" if volume_ratio > 1 else "Low"
                    if volume_ratio > 2:
                        pump_count += 1
                    
                    # BTC correlation
                    btc_corr = coin.get('correl_btc', 0)
                    btc_display = f"{btc_corr:.2f}" if isinstance(btc_corr, (int, float)) else "N/A"
                    
                    # Blacklist status
                    if hasattr(self.scorer, 'get_blacklist_status'):
                        blacklist_status, until = self.scorer.get_blacklist_status(symbol)
                        if blacklist_status:
                            status_text = f"🚫 BL ({time.strftime('%H:%M', time.localtime(until))})"
                        else:
                            status_text = "✅ OK"
                    else:
                        status_text = "✅ OK"

                    # Populate table row
                    self.table.setItem(i, 0, QTableWidgetItem(symbol))
                    
                    # Score with color coding
                    score_item = QTableWidgetItem(f"{score:.1f}")
                    if score >= 3:
                        score_item.setBackground(QColor(0, 170, 0, 100))  # Green
                    elif score >= 1:
                        score_item.setBackground(QColor(255, 165, 0, 100))  # Orange
                    else:
                        score_item.setBackground(QColor(255, 0, 0, 100))  # Red
                    self.table.setItem(i, 1, score_item)
                    
                    self.table.setItem(i, 2, QTableWidgetItem(volume_display))
                    self.table.setItem(i, 3, QTableWidgetItem(rsi_display))
                    self.table.setItem(i, 4, QTableWidgetItem(macd_status))
                    self.table.setItem(i, 5, QTableWidgetItem(bb_status))
                    self.table.setItem(i, 6, QTableWidgetItem(volatility))
                    self.table.setItem(i, 7, QTableWidgetItem(btc_display))
                    self.table.setItem(i, 8, QTableWidgetItem(status_text))
                    
                    # Action button
                    action_btn = QPushButton("📊 Részletek")
                    action_btn.clicked.connect(lambda checked, coin=coin: self.show_coin_details(coin))
                    action_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #0078D4;
                            color: white;
                            border: none;
                            padding: 3px 8px;
                            border-radius: 3px;
                            font-size: 10px;
                        }
                        QPushButton:hover {
                            background-color: #106EBE;
                        }
                    """)
                    self.table.setCellWidget(i, 9, action_btn)

                except Exception as e:
                    print(f"[SCORING] Error processing coin {i}: {e}")
                    # Fill with error data
                    self.table.setItem(i, 0, QTableWidgetItem(f"ERROR_{i}"))
                    for j in range(1, 9):
                        self.table.setItem(i, j, QTableWidgetItem("ERR"))

            # Update blacklist table
            self._update_blacklist_table()
            
            # Update status
            best_coins = [coin for coin, score in scored_coins if score >= 3]
            self.status_label.setText(
                f"📊 Érmék: {len(coin_list)} | 🔥 Pump: {pump_count} | "
                f"⭐ Kiváló (≥3): {len(best_coins)} | "
                f"🕒 Frissítve: {time.strftime('%H:%M:%S')}"
            )

        except Exception as e:
            print(f"[ERROR] Scoring panel refresh failed: {e}")
            self.status_label.setText(f"❌ Hiba: {str(e)[:50]}")

    def _update_blacklist_table(self):
        """Blacklist tábla frissítése"""
        try:
            if not hasattr(self.scorer, 'blacklist'):
                self.blacklist_table.setRowCount(0)
                return

            blacklisted = [
                (sym, until) for sym, until in self.scorer.blacklist.items() 
                if until > time.time()
            ]
            
            self.blacklist_table.setRowCount(len(blacklisted))
            
            for i, (symbol, until) in enumerate(blacklisted):
                self.blacklist_table.setItem(i, 0, QTableWidgetItem(symbol))
                self.blacklist_table.setItem(i, 1, QTableWidgetItem(
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(until))
                ))
                self.blacklist_table.setItem(i, 2, QTableWidgetItem("Auto blacklist"))

        except Exception as e:
            print(f"[ERROR] Blacklist table update failed: {e}")

    def _calculate_fallback_score(self, coin):
        """Fallback pontozás ha nincs scorer"""
        try:
            score = 0
            
            # Volume scoring
            volume_ratio = self._safe_divide(coin.get('volume_last', 0), coin.get('volume_15m_avg', 1))
            if volume_ratio > 2:
                score += 2
            elif volume_ratio > 1.5:
                score += 1
            
            # RSI scoring
            rsi = coin.get('rsi_3m', coin.get('rsi_15m', 50))
            if 30 <= rsi <= 70:
                score += 1
            
            # Price position scoring
            close = coin.get('close', 0)
            bb_upper = coin.get('boll_3m_upper', close * 1.02)
            if close > bb_upper * 0.98:
                score += 1
            
            return score
            
        except Exception as e:
            print(f"[ERROR] Fallback scoring failed: {e}")
            return 0

    def _safe_divide(self, a, b):
        """Biztonságos osztás"""
        try:
            return a / b if b != 0 else 0
        except (TypeError, ZeroDivisionError):
            return 0

    def show_coin_details(self, coin):
        """Coin részletek megjelenítése"""
        try:
            details_text = f"""
🪙 {coin.get('symbol', 'N/A')} Részletes Elemzés

📊 Alapadatok:
• Ár: ${coin.get('close', 0):.6f}
• 24h Forgalom: {coin.get('volume_last', 0):,.0f}
• Átlag forgalom (15m): {coin.get('volume_15m_avg', 0):,.0f}
• Forgalom arány: {self._safe_divide(coin.get('volume_last', 0), coin.get('volume_15m_avg', 1)):.2f}x

📈 Technikai Indikátorok:
• RSI (3m): {coin.get('rsi_3m', 'N/A')}
• RSI (15m): {coin.get('rsi_15m', 'N/A')}
• Bollinger Upper: {coin.get('boll_3m_upper', 'N/A')}
• Bollinger Lower: {coin.get('boll_3m_lower', 'N/A')}

🔗 Korrelációk:
• BTC korreláció: {coin.get('correl_btc', 'N/A')}
• ETH korreláció: {coin.get('correl_eth', 'N/A')}

⚡ Teljesítmény:
• Közelmúlt win rate: {coin.get('recent_winrate', 'N/A')}
• BTC breakout: {'✅' if coin.get('btc_is_breaking', False) else '❌'}
• ETH breakout: {'✅' if coin.get('eth_is_breaking', False) else '❌'}

🎯 Pontozás:
• Végső pontszám: {coin.get('score', 'Számítás...')}
            """

            # Create details dialog
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{coin.get('symbol', 'N/A')} - Részletes Elemzés")
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout()
            
            text_edit = QTextEdit()
            text_edit.setPlainText(details_text)
            text_edit.setReadOnly(True)
            text_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #2D2D30;
                    color: white;
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                    border: 1px solid #555;
                    border-radius: 5px;
                    padding: 10px;
                }
            """)
            layout.addWidget(text_edit)
            
            close_btn = QPushButton("Bezárás")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078D4;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
            """)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            print(f"[ERROR] Show coin details failed: {e}")

    def update_scores(self, scored_results):
        """External update method for compatibility"""
        try:
            if not scored_results:
                return
                
            self.table.setRowCount(len(scored_results))
            
            for i, result in enumerate(scored_results):
                symbol = result.get('symbol', f'UNKNOWN_{i}')
                score = result.get('score', 0)
                
                self.table.setItem(i, 0, QTableWidgetItem(symbol))
                self.table.setItem(i, 1, QTableWidgetItem(f"{score:.2f}"))
                
                # Fill other columns with available data
                self.table.setItem(i, 2, QTableWidgetItem(str(result.get('volume', 'N/A'))))
                self.table.setItem(i, 3, QTableWidgetItem("N/A"))  # RSI
                self.table.setItem(i, 4, QTableWidgetItem("N/A"))  # MACD
                self.table.setItem(i, 5, QTableWidgetItem("N/A"))  # Bollinger
                self.table.setItem(i, 6, QTableWidgetItem("N/A"))  # Volatility
                self.table.setItem(i, 7, QTableWidgetItem("N/A"))  # BTC Corr
                self.table.setItem(i, 8, QTableWidgetItem("✅ OK"))  # Status
                
            self.status_label.setText(f"📊 External update: {len(scored_results)} coins processed")
            
        except Exception as e:
            print(f"[ERROR] External score update failed: {e}")

    def get_best_coins(self, min_score=3.0):
        """Legjobb érmék lekérdezése"""
        try:
            if not self.coin_data_provider:
                return []
                
            coin_list = self.coin_data_provider()
            best_coins = []
            
            for coin in coin_list:
                if hasattr(self.scorer, 'score_coin'):
                    score = self.scorer.score_coin(coin)
                else:
                    score = self._calculate_fallback_score(coin)
                    
                if score >= min_score:
                    best_coins.append({
                        'symbol': coin.get('symbol'),
                        'score': score,
                        'data': coin
                    })
            
            # Sort by score descending
            best_coins.sort(key=lambda x: x['score'], reverse=True)
            return best_coins
            
        except Exception as e:
            print(f"[ERROR] Get best coins failed: {e}")
            return []
