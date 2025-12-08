#!/bin/bash

ROOT_DIR="/home/jannen/Documents/MAS2025/Sem_2/SEE/Assignment06/KUKA-Youbot-RobotArm_Manipulation/"   # <-- change this

echo "Searching for .bag files under: $ROOT_DIR"
find "$ROOT_DIR" -type f -name "*.bag"

read -p "Delete ALL these .bag files? (y/N): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

find "$ROOT_DIR" -type f -name "*.bag" -exec rm -f {} \;

echo "Done."
