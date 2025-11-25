#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Neutral model
echo -e "\nYou need to register at https://amass.is.tue.mpg.de"
read -p "Username (AMASS):" username
read -p "Password (AMASS):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p dataset/motions/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/DanceDB.tar.bz2' -O 'poses.tar.bz2' --no-check-certificate --continue

tar -xvjf poses.tar.bz2 DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses.npz 
mv DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses.npz ./dataset/motions/poses.npz
rm -rf DanceDB

