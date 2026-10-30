---
name: scene-image
description: Render a picture of the current scene, character, place or moment from this conversation with the local mako image model (Qwen-Image), and show it. Use when the user asks for an image, illustration, picture or visual of what is being written, imagined or built in the session.
---

# Scene image

Turn what the conversation has established into one image prompt, render it with mako, and show the result.

1. Write one image prompt in English, 60-200 words, from the conversation so far:
   - the setting and the subjects, with every visual detail the story has fixed (appearance, clothing, colours,
     objects, time of day, weather), the same way each time so characters stay recognisable across images;
   - the composition and camera (wide shot, close-up, viewpoint), the lighting and the mood;
   - the art style the user has asked for in this session; otherwise one that suits the material, stated plainly
     ("oil painting", "cinematic photograph", "watercolour", ...);
   - no text or lettering in the image unless the user wants it; no list of things to avoid.
   If the user asked for something specific ("the tavern from the first chapter", "portrait of the captain"), use
   the details the conversation gives for it and leave out the rest.
2. Pick a size: 1024x1024, or 1344x768 for a landscape or wide scene, 768x1344 for a standing figure or tall scene
   (each side a multiple of 32).
3. Run, from this skill's directory:

   ```sh
   scripts/render.sh WIDTHxHEIGHT "<the prompt>"
   ```

   It prints the path of the PNG it saved. It takes from about 10 seconds to a minute or two; meanwhile mako has the
   chat model unloaded, and this conversation continues as before once it is back.
4. The script opens the image in the Mac's image viewer. Reply with its path and the prompt used, in a line or two.
   Do not `read` the image unless the user wants to talk about what is in it (it then goes to the model as image
   input, which needs mako started with `--mmproj`). If the user wants changes, adjust the prompt and render again.

If the script fails, report its message: mako may not be running, may have no image model (`--image-model`), or may
need an API key (`MAKO_API_KEY`).
