# DigiFlow Python Reader

A pure Python implementation to read and visualize `.dfi` files produced by DigiFlow, a image processing tool for analyzing fluid flows including PIV processing.

## Overview

This tool provides Python support for:

- Reading `.dfi` image data (DigiFlow's native floating-point format)

- Extracting metadata such as image dimensions, intensity range, and plane names

- Displaying individual image planes using matplotlib with correct color scales

- Command-line interface for quick inspection and data exploration


## Features

- Robust `.dfi` binary file parsing (supports 32-bit and 64-bit float detection)

- Automatic decoding of DigiFlow tags, header, and structured image data

- CLI options:

  - `--info`: show image shape, intensity range, and plane labels

  - `--plot [plane_index]`: visualize a specific image plane

- Support for automatic scaling of vorticity plots using the parsed IntensityRange


## Status

- Successfully parses `.dfi` files

- Identifies bitness (32 vs 64-bit) through heuristic checks

- Handles multi-plane fields (e.g., velocity and vorticity)

- CLI interface fully operational


## Sample CLI Output

`$ python digiflowio.py sample.dfi --info \n
Image shape     : (3, 643, 1024)

Intensity range : -25.0–25.0

Planes          : {0: 'u', 1: 'v', 2: 'Vorticity'}
`
- Limited support for advanced DigiFlow rescaling modes (e.g., DataType 1100 tags)

- Metadata validation against DigiFlow PDF specs in progress
