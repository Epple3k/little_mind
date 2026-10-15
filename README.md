# A Little Mind

**An expressive visual instrument for exploring language-model sampling parameters.**

[Open the browser demo](https://little-mind.onrender.com/) · [Portfolio case study](https://emitrice.com/?project=little-mind)

Temperature, Top-K, Top-P, Min-P and repetition penalties can radically change a local model's output, but numeric fields alone do little to make their mechanisms legible. A Little Mind gives those controls tactile visual counterparts and a small reactive face. The face responds during interaction and returns to its neutral idle expression when the gesture ends.

## Release status

This is an **interactive design prototype**, not an installable LM Studio extension or hosted AI model.

- The default demo simulates token probabilities and generated text using a small illustrative vocabulary.
- The demo explicitly labels simulated output.
- Experimental connectors for LM Studio, Ollama and OpenAI-compatible local backends exist in the source but are **not yet validated as production-ready**.
- A website served over HTTPS may be unable to reach a local HTTP inference server because of browser-origin and local-network protections.
- Parameter availability, order of sampling operations and actual support vary between backends.

## Run it locally

Requires a modern version of Node.js. This project uses Vite, React, Tailwind CSS and TypeScript.

```sh
npm install
npm run dev
```

Or use pnpm with the included lockfile:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Open the local URL shown by Vite. Use the **Demo (simulated)** backend to explore the controls with no model running.

To experiment with LM Studio, first enable its local API server and read the [official server settings documentation](https://lmstudio.ai/docs/developer/core/server/settings). Cross-origin requests may require CORS and authentication, and some of the app's backend parameter mappings may need to be adjusted for your model and LM Studio version.

## Hosting

This is a static Vite app:

```sh
npm run build
```

Publish the generated `dist/` directory to a static host.

### Hugging Face Spaces

This project can also be published as a **Static Space**, which is a website/demo rather than an LM Studio plug-in.

Create a new Static Space at [Hugging Face](https://huggingface.co/new-space), add the source files, and use the following YAML front matter at the top of that Space's `README.md`:

```yaml
---
title: A Little Mind
emoji: 🧠
colorFrom: gray
colorTo: blue
sdk: static
app_build_command: npm run build
app_file: dist/index.html
---
```

The Space builds and serves the front end. It does **not** automatically host an LLM or install a plugin in LM Studio.

For LM Studio's native plugin ecosystem, see its [plugin development](https://lmstudio.ai/docs/typescript/plugins) and [sharing](https://lmstudio.ai/docs/typescript/plugins/publish-plugins) documentation. Native plugin settings currently expose standard controls rather than unrestricted custom React interfaces.

## Concept

A language model may behave as though it is more rigid, exploratory or repetitive, but sampling settings are not direct equivalents of human emotions or cognitive processes. The face is deliberately a responsive interface metaphor, not a representation of the model's inner experience.

Designed and prototyped by Emit Rice.
