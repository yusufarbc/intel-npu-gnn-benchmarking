# Interactive HPEC Poster

This directory contains two Slidev presentations for the paper **“Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis.”**

| Source | Role |
|---|---|
| [`poster.md`](poster.md) | Six-slide HPEC virtual poster deployed to GitHub Pages |
| [`slides.md`](slides.md) | Longer technical deck for extended discussion |

The public poster is available at:

<https://yusufarbc.github.io/intel-npu-gnn-benchmarking/>

## Why the deployed version has six slides

HPEC recommends four to six slides for a virtual poster. The short deck is organized around the conversation visitors are most likely to have:

1. Central result
2. Challenge and motivation
3. Experimental setup
4. FP32 latency result
5. INT8 and device-assignment exceptions
6. Deployment guidance and limitations

The repository and paper provide the deeper material. During the Zoom poster session, screen-share the GitHub Pages tab and open the linked repository only when someone asks for implementation details, raw results, or reproduction instructions.

## Run locally

Install Node.js 20 or later, then:

```bash
cd showcase
npm ci
npm run dev
```

The poster will be served at <http://localhost:3030>.

To open the full technical deck instead:

```bash
npm run dev:full
```

## Build or export

```bash
# Build the six-slide poster
npm run build

# Export the poster for offline backup
npm run export

# Build or export the longer technical deck
npm run build:full
npm run export:full
```

Keep an exported PDF available during the conference in case GitHub Pages or the Zoom connection is unreliable.

## Deployment

The workflow at [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) builds `poster.md` and deploys it to GitHub Pages on every push to `main`.

Before presenting, verify that:

- the GitHub Pages link opens in a private/incognito browser window;
- the QR code resolves to the repository;
- all charts remain readable at typical Zoom screen-share resolution;
- the title, author order, software versions, and headline values match the camera-ready paper;
- an offline export is available.
