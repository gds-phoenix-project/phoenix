#!/bin/bash

echo performance | sudo tee /sys/devices/system/cpu/*/cpufreq/scaling_governor
