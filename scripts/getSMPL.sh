#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p dataset/body_models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './dataset/body_models/smplify.zip' --no-check-certificate --continue
unzip dataset/body_models/smplify.zip -d dataset/body_models/smplify
mv dataset/body_models/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_NEUTRAL.pkl
rm -rf dataset/body_models/smplify
rm -rf dataset/body_models/smplify.zip
