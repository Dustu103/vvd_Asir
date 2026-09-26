# 🚀 Deploying Hoichoi ML Backend on Render

This guide explains how to deploy the **Hoichoi Problem 2 ML Model Backend** on **Render** (no Colab needed) as a 24/7 cloud service.

---

## 3-Minute Deployment Steps

### 1. Create a New Web Service on Render
1. Go to [https://dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** → **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your GitHub repo:
   `https://github.com/Dustu103/vvd_Asir`

### 2. Configure Service Settings
Render will automatically detect the `Dockerfile`:
* **Name**: `hoichoi-ml-backend`
* **Region**: Oregon (US West) or Frankfurt (EU)
* **Environment**: `Docker`
* **Health Check Path**: `/health`
* **Plan**: `Starter` (2 GB RAM recommended for Whisper Medium)

### 3. Deploy
Click **Create Web Service**.
Render will build the Docker container and start `ml_server.py`.
Once live, Render gives you a public HTTPS URL:
```
https://hoichoi-ml-backend.onrender.com
```

### 4. Connect with your Vercel Web App
1. Go to your Vercel Dashboard → Project Settings → **Environment Variables**.
2. Add:
   ```
   ML_BACKEND_URL = https://hoichoi-ml-backend.onrender.com
   ```
3. Or inside the live web app, paste the URL into the **ML Model** connection input and click **Connect**.

---

## Alternative Cloud Platforms (1-Click Docker)
Because the service uses a standard Dockerfile with `/health` and dynamic `$PORT`:
* **Railway**: Click *New Project* → *Deploy from GitHub repo* → Select `Dustu103/vvd_Asir`.
* **Fly.io**: Run `fly launch` in this directory.
* **Hugging Face Spaces**: Create a Docker Space and push repo.
