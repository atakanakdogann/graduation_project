import sys
import json
import os
import http.server
import socketserver
import threading
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QWidget,
    QLabel,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl

PORT = 8000
HTML_FILE = "route_map.html"

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Route Visualizer")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        top_layout = QHBoxLayout()
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["BAFRS", "RFBAS", "IM"])
        self.strategy_combo.currentTextChanged.connect(self.load_data)
        top_layout.addWidget(QLabel("Strategy:"))
        top_layout.addWidget(self.strategy_combo)

        self.day_combo = QComboBox()
        self.day_combo.currentTextChanged.connect(self.day_changed)
        top_layout.addWidget(QLabel("Day:"))
        top_layout.addWidget(self.day_combo)
        self.layout.addLayout(top_layout)

        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)

        self.data = None
        self.load_data()

    def load_data(self):
        strategy = self.strategy_combo.currentText().lower()
        filename = f"{strategy}_results.json" 
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.populate_day_selector()
        except FileNotFoundError:
            print(f"Error: '{filename}' not found. ")
        except Exception as e:
            print(f"Error when loading data: {e}")

    def populate_day_selector(self):
        self.day_combo.blockSignals(True)
        self.day_combo.clear()
        if self.data:
            routes = self.data.get("analysis_results", {}).get("vns_routing", {}).get("routes", [])
            days = sorted(list(set(route["day"] for route in routes)))
            self.day_combo.addItems([f"Day {day}" for day in days])
        self.day_combo.blockSignals(False)
        self.day_changed()

    def day_changed(self):
        self.generate_html_for_day()
        self.web_view.setUrl(QUrl(f"http://localhost:{PORT}/{HTML_FILE}?day={self.day_combo.currentText()}"))

    def generate_html_for_day(self):
        if not self.data or not self.day_combo.currentText():
            return

        selected_day = int(self.day_combo.currentText().split(" ")[1])
        all_nodes = self.data["coordinates"] + [self.data["depot"]] + self.data["IFs"]
        
        day_routes = [
            r for r in self.data["analysis_results"]["vns_routing"]["routes"] 
            if r["day"] == selected_day
        ]

        animation_steps = []
        if day_routes:
            max_steps = max(len(r['nodes']) for r in day_routes) if day_routes else 0
            for step in range(max_steps):
                step_positions = []
                for route_info in day_routes:
                    node_idx = route_info['nodes'][min(step, len(route_info['nodes']) - 1)]
                    step_positions.append(all_nodes[node_idx])
                animation_steps.append(step_positions)

        truck_icon_urls = [
            'garbage_truck_green.png',
            'garbage_truck_red.png',
        ]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Route Map</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
                .controls {{ position: fixed; top: 10px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <div class="controls">
                <button id="startBtn" onclick="startAnim()">Start</button>
                <button id="pauseBtn" onclick="pauseAnim()">Pause</button>
                <input type="range" id="slider" min="0" value="0" oninput="sliderChange(this.value)" style="vertical-align: middle;">
                <span id="stepLabel">Step: 1</span>
            </div>

            <script>
                var map = L.map('map').setView({self.data['depot']}, 12);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(map);

                var animationSteps = {json.dumps(animation_steps)};
                var truckIconUrls = {json.dumps(truck_icon_urls)};
                var truckIcons = truckIconUrls.map(url => L.icon({{ iconUrl: url, iconSize: [40, 40], iconAnchor: [20, 20] }}));
                var depotIcon = L.icon({{iconUrl: 'Base_Depot.png', iconSize: [50, 50], iconAnchor: [25, 25], popupAnchor: [0, -25]}});
                var truckMarkers = [];
                var currentStep = 0;
                var animTimer = null;

                var customers = {json.dumps(self.data['coordinates'])};
                customers.forEach(c => L.circleMarker(c, {{radius: 3, color: 'blue', fill: true, fillColor: 'blue'}}).addTo(map));
                
                L.marker({self.data['depot']}, {{icon: depotIcon}}).addTo(map).bindPopup('Depot');
                var ifs = {json.dumps(self.data['IFs'])};
                ifs.forEach(i => L.marker(i).addTo(map).bindPopup('IF'));
                
                var allNodes = {json.dumps(all_nodes)};
                var routeColors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'];
                var dayRoutes = {json.dumps(day_routes)};
                dayRoutes.forEach((route, index) => {{
                    var coords = route.nodes.map(idx => allNodes[idx]);
                    var routeColor = routeColors[index % routeColors.length];
                    L.polyline(coords, {{color: routeColor, weight: 5, opacity: 0.9}}).addTo(map);
                }});

                var slider = document.getElementById('slider');
                var stepLabel = document.getElementById('stepLabel');
                if (animationSteps.length > 0) {{
                    slider.max = animationSteps.length - 1;
                    stepLabel.textContent = `Step: 1 / ${{animationSteps.length}}`;
                }} else {{
                    stepLabel.textContent = "No routes for this day";
                }}

                function updateTrucks(step) {{
                    currentStep = parseInt(step);
                    if (isNaN(currentStep) || !animationSteps[currentStep]) return;

                    truckMarkers.forEach(m => m.remove());
                    truckMarkers = [];
                    
                    animationSteps[currentStep].forEach((pos, i) => {{
                        var marker = L.marker(pos, {{icon: truckIcons[i % truckIcons.length]}}).addTo(map);
                        truckMarkers.push(marker);
                    }});
                    slider.value = currentStep;
                    stepLabel.textContent = `Step: ${{currentStep + 1}} / ${{animationSteps.length}}`;
                }}
                
                function stepForward() {{
                    if (currentStep < animationSteps.length - 1) {{
                        currentStep++;
                        updateTrucks(currentStep);
                    }} else {{
                        pauseAnim();
                    }}
                }}

                function stepBackward() {{
                    if (currentStep > 0) {{
                        currentStep--;
                        updateTrucks(currentStep);
                    }}
                }}

                function startAnim() {{
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('pauseBtn').disabled = false;
                    if(animTimer) clearInterval(animTimer);
                    animTimer = setInterval(stepForward, 1000);
                }}

                function pauseAnim() {{
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('pauseBtn').disabled = true;
                    clearInterval(animTimer);
                }}

                function sliderChange(val) {{
                    pauseAnim();
                    updateTrucks(val);
                }}
                
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'ArrowRight') {{
                        e.preventDefault();
                        pauseAnim();
                        stepForward();
                    }} else if (e.key === 'ArrowLeft') {{
                        e.preventDefault();
                        pauseAnim();
                        stepBackward();
                    }}
                }});

                if (animationSteps.length > 0) {{
                    updateTrucks(0);
                }}
                pauseAnim();
            </script>
        </body>
        </html>
        """
        try:
            with open(HTML_FILE, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            print(f"Error when trying to write HTML file: {e}")


def run_server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()

if __name__ == "__main__":
    if not os.path.exists("garbage_truck_green.png") or not os.path.exists("garbage_truck_red.png") or not os.path.exists("Base_Depot.png"):
        print("WARNING: Truck or Depot icon images not found. Please ensure they are in the same directory.")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec())