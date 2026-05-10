#!/usr/bin/env bash
# install.sh — Installations-Skript für den IOTA Trading Bot
# Getestet auf Ubuntu 22.04 und 24.04

set -euo pipefail

# --- Farben ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "================================================="
echo "   IOTA Trading Bot — Installation"
echo "================================================="
echo ""

# -------------------------------------------------------
# Schritt 1: Systempakete
# -------------------------------------------------------
info "Schritt 1/5: Systempakete prüfen und installieren..."

if ! command -v apt-get &>/dev/null; then
    error "Dieses Skript benötigt apt (Ubuntu/Debian). Andere Distributionen werden nicht unterstützt."
fi

sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv git curl
success "Systempakete installiert."

# -------------------------------------------------------
# Schritt 2: Python-Version prüfen
# -------------------------------------------------------
info "Schritt 2/5: Python-Version prüfen..."

PYTHON_BIN="python3"
PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 || ( "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ) ]]; then
    warn "Python $PYTHON_VERSION gefunden — mindestens 3.11 erforderlich."
    info "Installiere Python 3.11 über deadsnakes PPA..."
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-pip
    PYTHON_BIN="python3.11"
    PYTHON_VERSION=$("$PYTHON_BIN" --version | awk '{print $2}')
fi

success "Python $PYTHON_VERSION wird verwendet ($PYTHON_BIN)."

# -------------------------------------------------------
# Schritt 3: Virtuelle Umgebung
# -------------------------------------------------------
info "Schritt 3/5: Virtuelle Umgebung einrichten..."

VENV_DIR="$REPO_DIR/venv"

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtuelle Umgebung existiert bereits unter $VENV_DIR — wird übersprungen."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtuelle Umgebung erstellt: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtuelle Umgebung aktiviert."

# -------------------------------------------------------
# Schritt 4: TensorFlow-Variante wählen und installieren
# -------------------------------------------------------
info "Schritt 4/5: Python-Abhängigkeiten installieren..."
echo ""

# GPU-Erkennung
GPU_AVAILABLE=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unbekannt")
    GPU_AVAILABLE=true
    info "NVIDIA-GPU erkannt: $GPU_NAME"
fi

# Nutzer fragen
echo "  Welche TensorFlow-Variante soll installiert werden?"
echo ""
if [[ "$GPU_AVAILABLE" == "true" ]]; then
    echo "  [1] tensorflow      (GPU + CPU, nutzt die erkannte NVIDIA-GPU)"
    echo "  [2] tensorflow-cpu  (nur CPU, kein CUDA erforderlich)"
else
    echo "  [1] tensorflow-cpu  (empfohlen — kein GPU erkannt)"
    echo "  [2] tensorflow      (GPU + CPU, nur sinnvoll mit NVIDIA-GPU + CUDA)"
fi
echo ""
read -rp "  Auswahl [1/2, Standard: 1]: " TF_CHOICE
TF_CHOICE="${TF_CHOICE:-1}"

if [[ "$GPU_AVAILABLE" == "true" ]]; then
    [[ "$TF_CHOICE" == "2" ]] && TF_PKG="tensorflow-cpu" || TF_PKG="tensorflow"
else
    [[ "$TF_CHOICE" == "2" ]] && TF_PKG="tensorflow" || TF_PKG="tensorflow-cpu"
fi

echo ""
info "Installiere $TF_PKG und weitere Abhängigkeiten..."
pip install --quiet --upgrade pip
pip install --quiet "requests>=2.31.0" "pandas>=2.0.0" "schedule>=1.2.0" "numpy>=1.24.0"
pip install --quiet "${TF_PKG}>=2.13.0"
success "Abhängigkeiten installiert (${TF_PKG})."

# -------------------------------------------------------
# Schritt 5: Verzeichnisse anlegen und Installation prüfen
# -------------------------------------------------------
info "Schritt 5/5: Verzeichnisse anlegen und Installation prüfen..."

mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/models"
success "Verzeichnisse logs/ und models/ angelegt."

# Imports prüfen
python3 -c "
import sys
failures = []
for mod in ['tensorflow', 'pandas', 'requests', 'schedule', 'numpy', 'sqlite3']:
    try:
        __import__(mod)
    except ImportError:
        failures.append(mod)
if failures:
    print('FEHLER: Folgende Module fehlen: ' + ', '.join(failures))
    sys.exit(1)

import tensorflow as tf
import pandas as pd
import numpy as np
print(f'  tensorflow : {tf.__version__}')
print(f'  pandas     : {pd.__version__}')
print(f'  numpy      : {np.__version__}')
" || error "Import-Prüfung fehlgeschlagen. Überprüfe die Installation."

success "Alle Module erfolgreich importiert."

# -------------------------------------------------------
# Zusammenfassung
# -------------------------------------------------------
echo ""
echo "================================================="
echo -e "${GREEN}   Installation abgeschlossen!${NC}"
echo "================================================="
echo ""
echo "  Projektverzeichnis : $REPO_DIR"
echo "  Python             : $("$VENV_DIR/bin/python" --version)"
echo "  TensorFlow         : $TF_PKG"
echo ""
echo "  Nächste Schritte:"
echo ""
echo "  1. Virtuelle Umgebung aktivieren:"
echo "     source $REPO_DIR/venv/bin/activate"
echo ""
echo "  2. Daten sammeln (mind. ein paar Stunden laufen lassen):"
echo "     python main.py collect"
echo ""
echo "  3. Modell trainieren:"
echo "     python main.py train --timeframe 1h"
echo ""
echo "  4. Paper-Trading starten:"
echo "     python main.py paper"
echo ""
echo "  5. Backtest ausführen:"
echo "     python main.py backtest"
echo ""
