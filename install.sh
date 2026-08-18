#!/bin/bash

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Playwright Chromium browser
echo "Installing Chromium browser..."
playwright install chromium

echo "Installation complete!"
