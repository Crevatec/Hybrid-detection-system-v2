# run_dashboard.py
"""
Entry point for the Streamlit dashboard.

Run:
    streamlit run run_dashboard.py

Or directly:
    python run_dashboard.py  (launches streamlit programmatically)
"""

import os
import sys
import subprocess

DASHBOARD = os.path.join(os.path.dirname(__file__), "dashboard", "dashboard_app.py")

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "streamlit", "run", DASHBOARD,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    subprocess.run(cmd)
