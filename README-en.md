# ComfyUI DashuaiTools – Node Overview

## Image_Nodes

- **ImageComparisonGIF**  
  Takes two images and creates a left–right sliding comparison GIF. Supports custom frame count, automatically saves to the ComfyUI `output` folder and avoids filename collisions by adding numeric suffixes.

- **LoadImageList**  
  Loads images from a folder in batch. Supports sorting by alphabetical order, numeric order, or file modification time (ascending/descending), and lets you set start index and maximum number of outputs. Returns both image tensors and the corresponding filenames for easier batch processing and logging.

- **SaveImageWithName**  
  Saves one or multiple images to a given path with custom filenames and optional suffix. Automatically fills in missing extensions, generates unique filenames to avoid overwriting, and handles EXIF orientation to prevent rotated outputs.

- **XY_Image (XY chart)**  
  A standalone XY Plot–style node for ComfyUI. It hooks into the entire workflow graph: you define X/Y axes as parameter overrides (prompt, CFG, seed, etc.), and it executes combinations to build comparison grids. It uses content hashing and a global cache so that only changes in upstream nodes or XY configuration trigger re‑execution; layout tweaks (gap, font size, layout mode) reuse cached results, giving a “dashboard‑like” refresh experience. Supports three layouts: “XY Plot / X:Y / Y:X”, with flexible labels and axis configuration, ideal for large workflows and parameter sweeps.

## PS_Nodes

- **MinimumFilter**  
  A Photoshop‑style minimum filter implemented with `skimage`’s rank minimum operator. Radius is configurable and it works on single images or batches, useful for pre‑processing tasks such as matte refinement or detail suppression.

## API_Nodes

- **api_caption (lite)**  
  Uses a multimodal chat API to generate descriptions/prompts for a single image. Compatible with the OpenAI‑style API schema and supports multiple providers (Siliconflow, ZhenZhen Workshop, OpenRouter). You can customize model name, temperature and base prompt, choose Chinese or English output, and get basic retry + error reporting.

- **batch_api_caption (lite)**  
  Processes all images in a folder and calls a multimodal API in parallel to generate text descriptions, saving each caption into a `.txt` file with the same basename. Allows configuring input/output folders, model, prompt, output language and concurrency level, and returns a JSON log summarizing total/succeeded/failed counts—suitable for large‑scale dataset labeling.

## Tools
## 参考代码https://github.com/PGCRT/CRT-Nodes/blob/main/py/LoraLoaderZImage.py
- **ZImageLoraModelOnly**  
  A Z‑Image–specific LoRA loader based on the official “LoRA Model‑Only Loader”. It loads Q/K/V‑fused Z‑Image LoRA weights and applies them only to the MODEL branch (leaving CLIP untouched). You can pick any existing LoRA by filename and adjust the model strength; recommended as the dedicated LoRA entry point for Z‑Image pipelines.

## Video_Nodes

- **(Reserved)**  
  Currently no video nodes are shipped; this section is reserved for future extensions.

---

More examples and tutorials (Chinese) on Bilibili:  
https://space.bilibili.com/85024828

---

