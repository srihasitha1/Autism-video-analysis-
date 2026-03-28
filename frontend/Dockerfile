# AutiSense Frontend - Docker Image

FROM node:20-slim

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────────────
# Copy package files first for Docker layer caching
COPY package.json ./
COPY package-lock.json* ./

# Install dependencies
RUN npm install

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Startup command ──────────────────────────────────────────────────────────
# --host 0.0.0.0 is required for Vite to be reachable from outside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
