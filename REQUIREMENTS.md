# Overview
This repo is meant to be a custom implementation of Qwen Image: https://github.com/QwenLM/Qwen-Image-2.1

Use Python, Torch, FastAPI, etc to complete the implementation.

# Requirements
1. Core Functionality
   1. generate an image using a text prompt and/or reference image
   2. edit an image
   3. combine subjects in to a new image
   4. identify and isolate a subject in an image
      - for example, user outline-selects a character in a cartoon, and this app will remove everything else from the image, including the backgrounds
      - should work for sub-elements too, like objects a character is holding, or specific items scattered in the background
   5. quantized variations of Qwen Image should be selectable (and automatically-downloads)
2. Web UI
   1. a simple python web UI that allows the user to easily-use this implementation
   2. should allow the user to create/close tabs
      - each tab represents a workspace with all the controls available to the user (similar to if they were to open the web UI in multiple browser tabs)
3. Web API
   1. each core functionality should have an API endpoint
   2. the web UI should implement this API
4. Terminal CLI
   1. the app should support command-line invocation (i.e. for scripting)
   2. each core functionality should be accessible through a CLI