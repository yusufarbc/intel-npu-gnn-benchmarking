# Showcase — Academic Presentation Slides

Interactive, web-based presentation slides for the conference presentation of the paper: **"Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs"**.

Built using [Slidev](https://sli.dev/) with a custom academic light theme.

## Features

- **Academic Light Theme**: Clean, high-contrast typography and layout matching standard scientific presentation styles (`style.css`).
- **Interactive Term Notes**: A global glossary helper (`components/Glossary.vue`) displaying definitions of hardware, software, and graph theory terms directly on the slides.
- **Persistent Footers**: Slide numbers and paper citation info (`global-bottom.vue`) visible on all content slides.
- **Embedded Architecture Blueprints & Charts**: High-quality architecture diagrams and performance figures matching the paper.

## Running Locally

To run the presentation in development mode locally:

1. Navigate to the `showcase/` directory:
   ```bash
   cd showcase
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the local development server:
   ```bash
   npm run dev
   ```

The presentation will be available at `http://localhost:3030`.

## Building and Deployment

The slides are automatically compiled and deployed to **GitHub Pages** on every push to the `main` branch via the GitHub Action workflow:

- Workflow file: `.github/workflows/deploy.yml`
- Production build command: `npm run build`
- Target environment: GitHub Pages
