# Interactive HPEC Poster

This directory contains the Slidev presentation for the paper **“Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis.”**

| Source | Role |
|---|---|
| [`poster.md`](poster.md) | Six-slide HPEC virtual poster deployed to GitHub Pages |
| [`poster-export.pdf`](poster-export.pdf) | Six-slide offline PDF backup for screen sharing |

The public interactive poster is available at:

<https://yusufarbc.github.io/intel-npu-gnn-benchmarking/>

## Structure of the Six-Slide Poster

HPEC recommends four to six slides for a virtual poster. The deck is organized around the conversation visitors are most likely to have during the session:

1. **Central result:** Main takeaway, authors, and link to code/data
2. **Challenge and motivation:** Dense streaming NPU vs. irregular sparse GNN mismatch
3. **Experimental setup:** Meteor Lake SoC platform, backends, workloads, and protocol
4. **FP32 latency result:** Dense models benefit on NPU; iGPU leads on evaluated GNNs
5. **INT8 and device-assignment exceptions:** SGC regression, GAT compilation failures, silent CPU fallback
6. **Deployment guidance and limitations:** Practical backend decision rules and evidence boundaries

The repository and camera-ready paper provide deeper material. During the Zoom poster session, screen-share the GitHub Pages tab or offline PDF and open the linked repository when visitors request raw data, traces, or reproduction scripts.

## Run locally

Install Node.js 20 or later, then:

```bash
cd showcase
npm ci
npm run dev
```

The poster will be served at <http://localhost:3030>.

## Build or export

```bash
# Build the six-slide poster
npm run build

# Export the poster as PDF for offline backup
npm run export
```

Keep the exported PDF ([`poster-export.pdf`](poster-export.pdf)) open during the conference in case GitHub Pages or the network connection is unreliable.

## Deployment

The GitHub Actions workflow at [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) builds `poster.md` and deploys it to GitHub Pages automatically on every push to `main`.
