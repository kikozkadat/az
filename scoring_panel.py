# gui/scoring_panel.py - THREADED VERSION

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QHeaderView
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
import time
import logging

logger = logging.getLogger(__name__)

class ScoringWorker(QThread):
    """Háttérszál a pontozási műveletekhez"""
    data_ready = pyqtSignal(list)  # Scored coins lista
    status_update = pyqtSignal(str)  # Status üzenet
    error_occurred = pyqtSignal(str)  # Hiba üzenet
    
    def __init__(self, scorer, coin_data_provider):
        super().__init__()
        self.scorer = scorer
        self.coin_data_provider = coin_data_provider
        self.running = True
        
    def run(self):
        """Háttérszálon futó pontozási logika"""
        try:
            if not self.coin_data_provider:
                self.status_update.emit("Státusz: Nincs adatforrás")
                return
                
            # Adatok lekérése
            coin_list = self.coin_data_provider()
            if not coin_list:
                self.status_update.emit("Státusz: Nincs coin adat")
                return
            
            scored_coins = []
            pump_count = 0
            
            # Pontozás végrehajtása
            for i, coin in enumerate(coin_list):
                if not self.running:  # Megszakítás ellenőrzése
                    break
                    
                try:
                    # Score számítás
                    if hasattr(self.scorer, 'score_coin'):
                        score = self.scorer.score_coin(coin)
                    else:
                        score = self._calculate_fallback_score(coin)
                    
                    # Adatok előkészítése a GUI számára
                    volume_ratio = self._safe_divide(coin.get('volume_last', 0), coin.get('volume_15m_avg', 1))
                    
                    if volume_ratio > 2:
                        pump_count += 1
                    
                    coin_data = {
                        'index': i,
                        'symbol': coin.get('symbol', f'COIN_{i}'),
                        'score': score,
                        'volume_display': f"{coin.get('volume_last', 0):.0f} / {coin.get('volume_15m_avg', 0):.0f}",
                        'rsi': coin.get('rsi_3m', coin.get('rsi_15m', 50)),
                        'volume_ratio': volume_ratio,
                        'btc_corr': coin.get('correl_btc', 0),
                        'close': coin.get('close', 0),
                        'bb_upper': coin.get('boll_3m_upper', 0),
                        'bb_lower': coin.get('boll_3m_lower', 0),
                        'raw_coin': coin  # Eredeti coin adat
                    }
                    
                    scored_coins.append(coin_data)
                    
                except Exception as e:
                    logger.error(f"Error processing coin {i}: {e}")
                    
            # Eredmények küldése a fő szálnak
            self.data_ready.emit(scored_coins)
            
            # Status update
            best_coins = [c for c in scored_coins if c['score'] >= 3]
            status_msg = (f"📊 Érmék: {len(coin_list)} | 🔥 Pump: {pump_count} | "
                         f"⭐ Kiváló (≥3): {len(best_coins)} | "
                         f"🕒 Frissítve: {time.strftime('%H:%M:%S')}")
            self.status_update.emit(status_msg)
            
        except Exception as e:
            logger.error(f"Scoring worker error: {e}")
            self.error_occurred.emit(f"❌ Hiba: {str(e)[:50]}")
    
    def stop(self):
        """Szál leállítása"""
        self.running = False
        
    def _calculate_fallback_score(self, coin):
        """Fallback pontozás"""
        try:
            score = 0
            volume_ratio = self._safe_divide(coin.get('volume_last', 0), coin.get('volume_15m_avg', 1))
            if volume_ratio > 2:
                score += 2
            elif volume_ratio > 1.5:
                score += 1
            
            rsi = coin.get('rsi_3m', coin.get('rsi_15m', 50))
            if 30 <= rsi <= 70:
                score += 1
            
            close = coin.get('close', 0)
            bb_upper = coin.get('boll_3m_upper', close * 1.02)
            if close > bb_upper * 0.98:
                score += 1
            
            return score
        except Exception:
            return 0
            
    def _safe_divide(self, a, b):
        """Biztonságos osztás"""
        try:
            return a / b if b != 0 else 0
        except (TypeError, ZeroDivisionError):
            return 0


class ScoringPanel(QWidget):
    def __init__(self, scorer, coin_data_provider, parent=None):
        super().__init__(parent)
        self.scorer = scorer
        self.coin_data_provider = coin_data_provider
        self.worker = None
        self.worker_thread = None
        self.refresh_interval = 10000  # 10 seconds
        
        self.init_ui()
        self.setup_worker()
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
        
        # Manual refresh button
        self.refresh_button = QPushButton("🔄 Kézi frissítés", self)
        self.refresh_button.clicked.connect(self.manual_refresh)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #106EBE; }
        """)
        self.layout.addWidget(self.refresh_button)

    def setup_worker(self):
        """Worker thread beállítása"""
        self.worker = ScoringWorker(self.scorer, self.coin_data_provider)
        
        # Signal kapcsolatok
        self.worker.data_ready.connect(self.update_table)
        self.worker.status_update.connect(self.update_status)
        self.worker.error_occurred.connect(self.handle_error)
        
    def setup_timer(self):
        """Automatikus frissítés időzítő"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(self.refresh_interval)
        
    def refresh_data(self):
        """Adatok frissítése háttérszálon"""
        if self.worker and not self.worker.isRunning():
            self.status_label.setText("Státusz: Frissítés folyamatban...")
            self.refresh_button.setEnabled(False)
            self.worker.start()
            
    def manual_refresh(self):
        """Kézi frissítés gomb"""
        logger.info("Manual refresh requested")
        self.refresh_data()
        
    @pyqtSlot(list)
    def update_table(self, scored_coins):
        """Táblázat frissítése a háttérszálból kapott adatokkal"""
        try:
            self.table.setRowCount(len(scored_coins))
            
            for coin_data in scored_coins:
                i = coin_data['index']
                
                # Szimbólum
                self.table.setItem(i, 0, QTableWidgetItem(coin_data['symbol']))
                
                # Pontszám színezéssel
                score_item = QTableWidgetItem(f"{coin_data['score']:.1f}")
                if coin_data['score'] >= 3:
                    score_item.setBackground(QColor(0, 170, 0, 100))  # Green
                elif coin_data['score'] >= 1:
                    score_item.setBackground(QColor(255, 165, 0, 100))  # Orange
                else:
                    score_item.setBackground(QColor(255, 0, 0, 100))  # Red
                self.table.setItem(i, 1, score_item)
                
                # Forgalom
                self.table.setItem(i, 2, QTableWidgetItem(coin_data['volume_display']))
                
                # RSI
                rsi = coin_data['rsi']
                rsi_display = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "N/A"
                self.table.setItem(i, 3, QTableWidgetItem(rsi_display))
                
                # MACD
                macd_status = "📈" if rsi > 50 else "📉"
                self.table.setItem(i, 4, QTableWidgetItem(macd_status))
                
                # Bollinger
                close_price = coin_data['close']
                bb_upper = coin_data['bb_upper']
                bb_lower = coin_data['bb_lower']
                
                if bb_upper > 0 and close_price > bb_upper * 0.98:
                    bb_status = "🔥 Kitörés"
                elif bb_lower > 0 and close_price < bb_lower * 1.02:
                    bb_status = "❄️ Túladott"
                else:
                    bb_status = "➡️ Középen"
                self.table.setItem(i, 5, QTableWidgetItem(bb_status))
                
                # Volatilitás
                volume_ratio = coin_data['volume_ratio']
                volatility = f"{volume_ratio:.1f}x" if volume_ratio > 1 else "Low"
                self.table.setItem(i, 6, QTableWidgetItem(volatility))
                
                # BTC korreláció
                btc_corr = coin_data['btc_corr']
                btc_display = f"{btc_corr:.2f}" if isinstance(btc_corr, (int, float)) else "N/A"
                self.table.setItem(i, 7, QTableWidgetItem(btc_display))
                
                # Státusz
                if hasattr(self.scorer, 'get_blacklist_status'):
                    blacklist_status, until = self.scorer.get_blacklist_status(coin_data['symbol'])
                    if blacklist_status:
                        status_text = f"🚫 BL ({time.strftime('%H:%M', time.localtime(until))})"
                    else:
                        status_text = "✅ OK"
                else:
                    status_text = "✅ OK"
                self.table.setItem(i, 8, QTableWidgetItem(status_text))
                
                # Művelet gomb
                action_btn = QPushButton("📊 Részletek")
                action_btn.clicked.connect(lambda checked, coin=coin_data['raw_coin']: self.show_coin_details(coin))
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
                
            # Blacklist tábla frissítése
            self._update_blacklist_table()
            
            # Frissítés gomb újra engedélyezése
            self.refresh_button.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Table update error: {e}")
            self.handle_error(f"Táblázat frissítési hiba: {str(e)[:50]}")
            
    @pyqtSlot(str)
    def update_status(self, status_msg):
        """Státusz frissítése"""
        self.status_label.setText(status_msg)
        self.refresh_button.setEnabled(True)
        
    @pyqtSlot(str)
    def handle_error(self, error_msg):
        """Hiba kezelése"""
        self.status_label.setText(error_msg)
        self.refresh_button.setEnabled(True)
        
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
            logger.error(f"Blacklist table update error: {e}")

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
            logger.error(f"Show coin details error: {e}")
            
    def _safe_divide(self, a, b):
        """Biztonságos osztás"""
        try:
            return a / b if b != 0 else 0
        except (TypeError, ZeroDivisionError):
            return 0

    # Kompatibilitási metódusok
    def update_scores(self, scored_results):
        """External update method for compatibility"""
        # Nem használjuk, mert a worker thread kezeli
        pass
        
    def refresh_panel(self):
        """Kompatibilitás a régi kóddal"""
        self.refresh_data()
        
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
                    score = 0
                    
                if score >= min_score:
                    best_coins.append({
                        'symbol': coin.get('symbol'),
                        'score': score,
                        'data': coin
                    })
            
            best_coins.sort(key=lambda x: x['score'], reverse=True)
            return best_coins
            
        except Exception as e:
            logger.error(f"Get best coins error: {e}")
            return []
            
    def start_auto_refresh(self):
        """Auto refresh indítása - kompatibilitás"""
        if self.timer and not self.timer.isActive():
            self.timer.start()
            
    def closeEvent(self, event):
        """Panel bezárásakor a worker thread leállítása"""
        if self.worker:
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(2000)  # Max 2 másodperc várakozás
        event.accept()
