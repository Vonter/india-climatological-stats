#!/bin/bash

for station in $(cat stations.txt); do

    echo "Fetching data for station $station"
    mkdir -p "raw/$station"

    # Climatology normals
    for month in {01..12}; do

        if [ -f "raw/$station/$month.html" ]; then
            continue
        fi

        curl 'https://dsp.imdpune.gov.in/home_normals.php#' -X POST --data-raw "month=$month&stn=$station" > "raw/$station/$month.html"
        sleep 0.5
    done

    # Climatology extremes
    for parameter in tmax_high tmax_low tmin_high tmin_low rf; do

        if [ -f "raw/$station/extremes_$parameter.html" ]; then
            continue
        fi

        curl 'https://dsp.imdpune.gov.in/home_extremes.php#' -X POST --data-raw "parameter=$parameter&stn=$station" > "raw/$station/extremes_$parameter.html"
        sleep 0.5
    done

    # Station metadata
    if [ -f "raw/$station/metadata.html" ]; then
        continue
    fi

    curl 'https://dsp.imdpune.gov.in/home_vivaranika.php#' -X POST --data-raw "stn=$station" > "raw/$station/metadata.html"
    sleep 0.5
done
