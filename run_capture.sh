#!/bin/bash

# Default values
DURATION=120
MODE="normal"
DELAY=0
OUTPUT_PREFIX=""

# Parse arguments
while getopts t:sdaw:o: flag
do
    case "${flag}" in
        t) DURATION=${OPTARG};;
        s) MODE="hop";;
        d) MODE="lock";;
        a) MODE="active";;
        w) DELAY=${OPTARG};;
        o) OUTPUT_PREFIX=${OPTARG};;
    esac
done

cd /home/krishna/Desktop/CSI_Data
mkdir -p data

# Safely catch Ctrl+C so we don't accidentally leave flood pings running forever!
trap 'echo -e "\n[ABORT] Killing background processes..."; kill $HOP_PID $PING_PID 2>/dev/null; exit 1' INT TERM

# Set up Monitor Mode (mon0) if it doesn't exist
if ! iw dev | grep -q mon0; then
    echo "Creating mon0 interface..."
    iw dev wlan0 interface add mon0 type monitor 2>/dev/null
fi
ip link set mon0 up 2>/dev/null

# Handle Modes
HOP_PID=""
PING_PID=""
if [ "$MODE" == "lock" ]; then
    echo "[Mode: DYNAMIC LOCK] Scanning wlan0 for the strongest signal..."
    # Scan via wlan0 to find the highest signal peak. Note: wlan0 must be active.
    BEST_FREQ=$(iw dev wlan0 scan 2>/dev/null | awk '/freq: /{f=$2} /signal: /{if(!freq_sigs[f] || $2>freq_sigs[f]) freq_sigs[f]=$2} END {max_f=""; max_s=-999; for(f in freq_sigs) if(freq_sigs[f]>max_s){max_s=freq_sigs[f]; max_f=f}; print max_f}')
    
    if [ -n "$BEST_FREQ" ]; then
        CLEAN_FREQ=$(echo $BEST_FREQ | cut -d'.' -f1)
        echo "Loudest frequency found: ${CLEAN_FREQ} MHz. Locking mon0 to it."
        iw dev mon0 set freq $CLEAN_FREQ 2>/dev/null
    else
        echo "WARNING: Could not determine strongest freq from scan. Defaulting to channel 6."
        iw dev mon0 set channel 6 2>/dev/null
    fi
elif [ "$MODE" == "active" ]; then
    echo "[Mode: ACTIVE FLOOD] Locking mon0 to wlan0's channel and forcing heavy router traffic."
    ACTIVE_CH=$(iw dev wlan0 info 2>/dev/null | awk '/channel/ {print $2}' | head -n 1)
    if [ -n "$ACTIVE_CH" ]; then
        echo "wlan0 is natively connected to channel $ACTIVE_CH. Destroying mon0 to bind directly to wlan0!"
        iw dev mon0 del 2>/dev/null
    else
        echo "WARNING: Concurrency failed! Are you connected to WiFi? Defaulting to channel 6."
        iw dev mon0 set channel 6 2>/dev/null
    fi

    BROADCAST_IP=$(ip -4 addr show wlan0 2>/dev/null | grep -oP '(?<=brd )\S+' | head -n 1)
    if [ -n "$BROADCAST_IP" ]; then
        echo "Network Broadcast IP detected at $BROADCAST_IP."
        echo "Forcing Router into overdrive (Targeting >100 Hz Legacy Broadcast CSI)..."
        ping -b -i 0.005 -s 1000 -q $BROADCAST_IP > /dev/null 2>&1 &
        PING_PID=$!
    else
        echo "WARNING: No Broadcast IP found to flood."
    fi
elif [ "$MODE" == "hop" ]; then
    echo "[Mode: CHANNEL HOPPING] Will cycle across 2.4GHz & 5GHz channels rapidly."
    # Define interesting broad spectrum channels
    CHANNELS=(1 6 11 36 40 44 48 149 153 157 161)
    
    # Background hopping function
    hop_loop() {
        while true; do
            for ch in "${CHANNELS[@]}"; do
                iw dev mon0 set channel $ch 2>/dev/null
                sleep 0.1
            done
        done
    }
    
    hop_loop &
    HOP_PID=$!
else
    echo "[Mode: NORMAL] Locking mon0 to default channel 6."
    iw dev mon0 set channel 6 2>/dev/null
fi

# Enable CSI and disable MAC filter
CSI_PATH=$(find /sys/kernel/debug/iwlwifi -type d -name "iwlmvm" | head -n 1)
if [ -n "$CSI_PATH" ]; then
    echo '' > "$CSI_PATH/csi_addresses" 2>/dev/null
    echo "MAC filter disabled."
    echo 0xffff > "$CSI_PATH/csi_frame_types"
    echo 1 > "$CSI_PATH/csi_enabled"
    echo "CSI enabled."
fi

# Backup old data
mv csi_stream.dat csi_stream_bak.dat 2>/dev/null
mv capture.log capture_bak.log 2>/dev/null

if [ "$DELAY" -gt 0 ]; then
    echo "Delaying start for $DELAY seconds... (Leave the room!)"
    for (( i=$DELAY; i>0; i-- )); do
        echo -ne "Starting in $i...\r"
        sleep 1
    done
    echo -e "\nStarting now!"
fi

echo "Starting ${DURATION}-second CSI capture."
timeout $DURATION stdbuf -oL ./csi_extractor > capture.log

# Clean up
if [ -n "$HOP_PID" ]; then
    echo "Stopping background channel hop (PID: $HOP_PID)..."
    kill $HOP_PID 2>/dev/null
fi
if [ -n "$PING_PID" ]; then
    echo "Stopping active RF flood (PID: $PING_PID)..."
    kill $PING_PID 2>/dev/null
fi

NUM_PACKETS=$(grep "Received CSI event" capture.log | wc -l)
echo "Total packets collected: $NUM_PACKETS"
if [ "$DURATION" -gt 0 ]; then
    RATE=$(awk "BEGIN {printf \"%.2f\", $NUM_PACKETS / $DURATION}")
    echo "Average packets per second: $RATE"
fi

# Catalog the data correctly
if [ -n "$OUTPUT_PREFIX" ]; then
    mv csi_stream.dat "data/${OUTPUT_PREFIX}.dat" 2>/dev/null
    mv capture.log "data/${OUTPUT_PREFIX}.log" 2>/dev/null
    echo "Dataset safely saved to: data/${OUTPUT_PREFIX}.dat"
else
    mv csi_stream.dat "data/csi_stream.dat" 2>/dev/null
    mv capture.log "data/capture.log" 2>/dev/null
    echo "Dataset safely saved to: data/csi_stream.dat"
fi
