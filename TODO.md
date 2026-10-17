# Instructions
For each item, create a side branch and implement it. Do not merge anything.
Name each branch in the form `branch-title` (do not include the TODO number in the branch name).

Any new features added should have an accompanying API access point (if applicable).

Skip anything mared as "[DONE]".

# TODO
1. In the selection tools, add an "Auto" selection where the user can click on an area on screen, and the implementation uses AI to auto-select the section in the image based on context, color, etc
   - for example, the user clicks on a red square on a background, and the red square is automatically selected
   - not sure if Qwen Image supports this kind of thing - if not, find the best free open model that supports this, and use it for this implementation
2. Add a `Dockerfile` and `docker-compose.yml` that deploys this in a minimal docker container so I can easy run this on my Linux server. Include a component in the `docker-compose` stack so this can be hosted behind its own SSL cert over HTTPS.
3. Add a `Upscale` tab next to the `Background` tab:
   - user can use this to up-scale an image using AI (to a resolution and/or scale like 2.5x or 10x, etc)