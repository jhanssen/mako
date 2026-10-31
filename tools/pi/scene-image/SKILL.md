---
name: scene-image
description: Render a picture of the current scene, character, place or moment from this conversation with one of the local mako image models, and show it. Use when the user asks for an image, illustration, picture or visual of what is being written, imagined or built in the session.
---

# Scene image

Turn what the conversation has established into one image prompt, render it with mako, and show the result.

0. Pick the image model (see "Models" below) unless one is already in use in this conversation.
1. Write one image prompt in English, 100-300 words, from the conversation so far, in full descriptive sentences
   rather than a list of keywords (the image models' text encoders are language models and read prose best; a
   model whose description says it takes tags can have tags in front of the sentences). The image model knows only
   what the prompt says: anything left out is filled in by guess, usually with the most generic choice. Spell every
   detail out, concretely and literally, even ones that seem obvious from the story:
   - the subjects: who or what, how many, and every visual detail the story has fixed (age, build, height, hair
     length, style and colour, eye colour, skin, face, clothing piece by piece with its colour, fit and state,
     accessories, objects held), the same words each time so characters stay recognisable across images;
   - exactly where each subject is and what each part of the body is doing: standing, sitting or lying, which way
     they face, where the hands and arms are, where they look, the expression; with several subjects, where each is
     relative to the others and to the camera, and who is touching what;
   - what is visible and what is not: which parts of the subjects are in frame, what is in front of or behind them,
     what fills the foreground and the background;
   - the setting in particular things (not "a room" but the room's walls, floor, furniture, light sources), the time
     of day and the weather;
   - the composition and camera (wide shot, close-up, viewpoint, height, lens), the lighting (direction, colour,
     hardness) and the mood;
   - the art style the user has asked for in this session; otherwise one that suits the material, stated plainly
     ("oil painting", "cinematic photograph", "watercolour", "editorial photography", ...);
   - no text or lettering in the image unless the user wants it; no list of things to avoid.
   Do not summarize or soften what the user asked for: an abstraction ("an intimate moment", "a dramatic scene", "a
   fantasy outfit") gives a generic image; say what is actually seen instead. Do not leave out a detail because it
   was mentioned long ago or seems implied; if the user describes something, it belongs in the prompt in their
   terms. Say plainly what the image should show; vague or hedged wording gives vaguer images.
   If the user asked for something specific ("the tavern from the first chapter", "portrait of the captain"), use
   every detail the conversation gives for it and leave out what does not belong to it. If the user says an image
   came out wrong, find which detail the prompt left vague or missing, write it out more explicitly, and keep the
   rest of the prompt the same.
2. Pick a size: 1024x1024, or 1344x768 for a landscape or wide scene, 768x1344 for a standing figure or tall scene
   (each side a multiple of 32).
3. Run, from this skill's directory:

   ```sh
   scripts/render.sh --model MODEL WIDTHxHEIGHT "<the prompt>"
   ```

   It prints the path of the PNG it saved. It takes from a few seconds to a few minutes (the model's description
   gives its time); meanwhile mako has the chat model unloaded, and this conversation continues as before once it is
   back.
4. The script opens the image in the Mac's image viewer. Reply with its path and the prompt used, in a line or two.
   Do not `read` the image unless the user wants to talk about what is in it (it then goes to the model as image
   input, which needs mako started with `--mmproj`). If the user wants changes, adjust the prompt and render again.

## Models

`scripts/models.sh` lists the image models mako has: each one's name, which model it is, a description of what it is
good at, how its prompts should be written and how long it takes, and its default LoRAs; one is marked "default"
(used without `--model`) and one "loaded" (rendering with it needs no reload).

- Pick by what the user asks for and by the descriptions: the style (anime and illustration, photograph, painting),
  what the image must contain (legible text, a layout), the kind of content, and the time the user will wait. A model
  the user names wins. If none fits clearly, use the default and say which you used.
- Keep the same model for the rest of the conversation, so characters and style stay consistent; change it only when
  the user asks or the material changes (a photograph after anime illustrations).
- Write the prompt the way the chosen model's description says (prose, tags, prompt weights).
- Rendering with another model than the loaded one first loads it (a few seconds more).

## LoRAs

LoRAs are add-ons to an image model (a style, a kind of content, a faster schedule), each model with its own.
`scripts/loras.sh MODEL` lists the chosen model's, by name; those marked "default" are what every render with it
uses unless told otherwise, at the scale shown.

- Without `--lora`, a render uses the model's defaults: leave `--lora` out unless you change them. With
  `--lora NAME[:SCALE]` (repeatable), it uses exactly the LoRAs given and no others, so to add one to the defaults,
  pass the defaults too:
  `scripts/render.sh --lora fast-4step:1 --lora watercolour:0.8 1024x1024 "<the prompt>"`.
  `--no-loras` renders with none.
- SCALE is the strength (1 when left out). Where the list gives a LoRA's scales, start in the middle of that range
  and stay within it. If the user finds an effect too weak or too strong, raise or lower it by 0.2-0.3.
- A model's default LoRAs are part of how it is set up (a speed LoRA such as "dmd", "lightning", "4step", "turbo"
  goes with the low step count and guidance the model is set to); the LoRAs you added are the user's choices. When
  the user asks to remove LoRAs, remove the ones you added and keep the defaults: pass the defaults alone as
  `--lora`, or leave `--lora` out if only defaults remain. Use `--no-loras` only when the user asks to drop the
  defaults too, and then, if a default was a speed LoRA, switch to the same model's folder without it instead (its
  description says which, e.g. `moss-sft` for `moss-sft-dmd`): without it the low step count gives soft, smeared
  images.
- Use a LoRA only when the user asks for it or its name clearly matches what they want. Once the user has chosen
  LoRAs, pass the same ones on every later render in the conversation: nothing is remembered between renders.
- A LoRA listed with a trigger acts only when the prompt contains it: work the trigger into the prompt, word for word
  (at the start if it ends in a colon, like "Next Scene:", else within the sentence where it fits). A description
  says what the LoRA is for. A LoRA with neither may still need words the list does not show; tell the user so if
  its effect does not show.
- A render with other LoRAs than the previous one first reloads the image model (a few seconds more). LoRAs do not
  carry over to another model: list the new model's and choose again.

If a script fails, report its message: mako may not be running, may have no image models (`--image-models`), may
need an API key (`MAKO_API_KEY`), or may not have a model or LoRA of that name (list them again).
